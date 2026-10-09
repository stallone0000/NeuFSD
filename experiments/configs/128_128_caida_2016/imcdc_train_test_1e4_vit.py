import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import time
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import random
import csv
from sklearn.metrics import r2_score

from model import CustomViT

# ================= 核心配置区域 =================
JSON_PATH = 'train_test_name_key.json'
PRETRAINED_ROOT = '0_pretrained_weights/ViT_10_1e4_results_1e-2'
FINETUNED_OUT_ROOT = '0_finetuned_results/ViT_10_1e4_results_1e-2' # 👈 新增这一行：独立的输出根目录
LABEL_DIR_NAME = '10_1e4_chazhi'                               
NUM_EPOCHS = int(os.environ.get('REPRO_EPOCHS', 20))                                                 
LR = 1e-2                                                       
BATCH_SIZE = 64                                                 
DATALOADER_NUM_WORKERS = int(os.environ.get("DATALOADER_NUM_WORKERS", 2))
PATIENCE = 5                                                     # ✅ 仅新增：早停耐心轮数
# ================================================

def read_bin_file(filename):
    data = np.fromfile(filename, dtype='<u4')
    if len(data) != 16384:
        print(f"⚠️ 文件长度异常: {len(data)} != 16384")
    acending_array = np.sort(data) 
    org_array = data                
    decending_array = acending_array[::-1] 
    final_array = np.concatenate((acending_array, org_array, decending_array))
    return final_array

def read_one_type_da(root_dir_name, dataset_id, label_dir):
    folder_path = f'{root_dir_name}/input/{dataset_id}'
    if not os.path.exists(folder_path):
        print(f"⚠️ 找不到数据文件夹: {folder_path}")
        return None, None

    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))
    if not files:
        return None, None

    sorted_data = [read_bin_file(os.path.join(folder_path, file)) for file in files]
    data = np.stack(sorted_data, axis=0) 
    
    label_folder_path = f'{root_dir_name}/{label_dir}/{dataset_id}.npy'
    if not os.path.exists(label_folder_path):
        print(f"⚠️ 找不到标签文件: {label_folder_path}")
        return None, None
        
    chazhi_label = np.load(label_folder_path)
    chazhi_label = chazhi_label.reshape(1, -1)
    chazhi_label = np.repeat(chazhi_label, data.shape[0], axis=0)
    
    data_tensor = torch.from_numpy(data).float()
    data_tensor = torch.reshape(data_tensor, (data_tensor.shape[0], 3, 128, 128))
    label_tensor = torch.from_numpy(chazhi_label).float() 
    
    return data_tensor, label_tensor

def run_evaluation(model, dataloader, device, desc="Evaluating"):
    model.eval()
    all_preds, all_targets = [], []
    eval_pbar = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for inputs, targets in eval_pbar:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
            
    final_preds = np.concatenate(all_preds)
    final_trues = np.concatenate(all_targets)
    
    val_mse = np.mean((final_preds - final_trues) ** 2)
    val_mae = np.mean(np.abs(final_preds - final_trues))
    val_r2 = r2_score(final_trues.ravel(), final_preds.ravel())
    val_wmrd = np.mean(np.abs(final_preds - final_trues)) / np.mean((final_preds + final_trues)/2) * 100
    
    return final_preds, final_trues, val_mse, val_mae, val_r2, val_wmrd

if __name__ == "__main__":
    print('='*80)
    print('🚀 启动在线微调与测试任务 (11到10000大流模型) ...')
    
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif os.environ.get("ALLOW_CPU") == "1":
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is required for reproduction; set ALLOW_CPU=1 only for debugging")
    print(f"Using device: {device}")

    with open(JSON_PATH, 'r') as f:
        tasks = json.load(f)

    # 👇 核心修改 1：不抛出异常，而是记录是否有预训练权重
    pretrained_models = glob.glob(os.path.join(PRETRAINED_ROOT, 'best_model_*.pth'))
    current_model_path = None
    if pretrained_models:
        current_model_path = pretrained_models[0]
        print(f"📦 发现初始预训练权重: {current_model_path}")
    else:
        print(f"⚠️ 在 {PRETRAINED_ROOT} 下未找到预训练权重，模型将随机初始化！")

    os.makedirs(FINETUNED_OUT_ROOT, exist_ok=True)

    first_dataset = list(tasks.keys())[0].split('.')[0]
    _, dummy_label = read_one_type_da('tr_ts_finetuned_continue', first_dataset, LABEL_DIR_NAME)
    out_dim = dummy_label.shape[-1]
    
    model = CustomViT(out_dim=out_dim).to(device)
    
    # 👇 核心修改 2：根据情况决定是 load_state_dict 还是直接用随机初始化的状态
    if current_model_path:
        model.load_state_dict(torch.load(current_model_path, map_location=device))
        print("✅ 预训练权重加载成功！")
    else:
        print("🌱 预训练权重不存在，从头开始训练 (Scratch)！")
        
    criterion = nn.SmoothL1Loss()
    
    task_list = list(tasks.items())
    task_pbar = tqdm(task_list, desc="总体任务进度")
    processed_train_files = set()
    current_finetune_dir = PRETRAINED_ROOT 

    for idx, (filename, is_train) in enumerate(task_pbar):
        dataset_id = filename.split('.')[0] 
        task_pbar.set_postfix({'current': dataset_id})
        
        if is_train:
            if filename in processed_train_files:
                continue

            print(f"\n" + "="*50)
            print(f"🔥 开始 Finetune (大流联合训练块，起点: {dataset_id})")
            
            train_dt_list, train_lt_list, train_names = [], [], []
            val_start_idx = len(task_list) 
            
            for j in range(idx, len(task_list)):
                next_filename, next_is_train = task_list[j]
                if not next_is_train: 
                    val_start_idx = j
                    break
                next_dataset_id = next_filename.split('.')[0]
                dt_train, lt_train = read_one_type_da('tr_ts_finetuned_continue', next_dataset_id, LABEL_DIR_NAME)
                if dt_train is not None:
                    train_dt_list.append(dt_train); train_lt_list.append(lt_train); train_names.append(next_dataset_id)
                processed_train_files.add(next_filename)

            if not train_dt_list: continue

            # 👇 这里就是破案的关键！加了这个打印，你就知道其实 5 个都塞进去了
            print(f"📥 成功聚合了 {len(train_names)} 个训练集: {train_names}")

            train_dataset = TensorDataset(torch.cat(train_dt_list, dim=0), torch.cat(train_lt_list, dim=0))
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_NUM_WORKERS)
            
            val_dt_list, val_lt_list, val_names = [], [], []
            for j in range(val_start_idx, len(task_list)):
                next_filename, next_is_train = task_list[j]
                if next_is_train: break
                next_dataset_id = next_filename.split('.')[0]
                dt_val, lt_val = read_one_type_da('tr_ts', next_dataset_id, LABEL_DIR_NAME)
                if dt_val is not None:
                    val_dt_list.append(dt_val); val_lt_list.append(lt_val); val_names.append(next_dataset_id)
            
            val_loaders = [] 
            if val_names:
                for i in range(len(val_names)):
                    ds = TensorDataset(val_dt_list[i], val_lt_list[i])
                    dl = DataLoader(ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=DATALOADER_NUM_WORKERS)
                    val_loaders.append((val_names[i], dl))
            else:
                dt_fallback, lt_fallback = read_one_type_da('tr_ts', train_names[-1], LABEL_DIR_NAME)
                val_loaders.append((train_names[-1], DataLoader(TensorDataset(dt_fallback, lt_fallback), batch_size=BATCH_SIZE*2, shuffle=False)))

            current_finetune_dir = os.path.join(PRETRAINED_ROOT, f"finetuned_block_{dataset_id}")
            os.makedirs(current_finetune_dir, exist_ok=True)
            
            model = CustomViT(out_dim=out_dim).to(device)  # one-by-one re-init: 每块独立从头
            
            optimizer = optim.Adam(model.parameters(), lr=LR)
            best_mean_wmrd = float('inf') 
            best_model_path_current = None
            patience_counter = 0  # ✅ 仅新增：早停计数器

            current_finetune_dir = os.path.join(FINETUNED_OUT_ROOT, f"finetuned_block_{dataset_id}") 
            os.makedirs(current_finetune_dir, exist_ok=True)
            csv_file = os.path.join(current_finetune_dir, 'finetune_results.csv')
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'mean_val_wmrd'])
                
            for epoch in range(NUM_EPOCHS):
                model.train()
                running_loss = 0.0
                train_pbar_inner = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train", leave=False)
                for inputs, targets in train_pbar_inner:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), targets)
                    loss.backward(); optimizer.step()
                    running_loss += loss.item()
                
                train_loss = running_loss / len(train_loader)
                
                current_epoch_wmrds, all_epoch_preds, all_epoch_trues = [], [], []
                for v_name, v_loader in val_loaders:
                    f_preds, f_trues, _, _, _, v_wmrd = run_evaluation(model, v_loader, device, desc=f"Eval {v_name}")
                    current_epoch_wmrds.append(v_wmrd); all_epoch_preds.append(f_preds); all_epoch_trues.append(f_trues)
                
                epoch_mean_wmrd = np.mean(current_epoch_wmrds)
                
                with open(csv_file, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch + 1, train_loss, epoch_mean_wmrd])
                    
                wmrd_str = " | ".join([f"{w:.2f}%" for w in current_epoch_wmrds])
                print(f"   ↳ Epoch [{epoch+1:02d}/{NUM_EPOCHS}] Loss: {train_loss:.4f} | 各测试集 WMRD: [{wmrd_str}] | 🎯 平均 WMRD: {epoch_mean_wmrd:.4f}%")

                if epoch_mean_wmrd < best_mean_wmrd:
                    if best_model_path_current and os.path.exists(best_model_path_current):
                        os.remove(best_model_path_current)
                    best_mean_wmrd = epoch_mean_wmrd
                    best_model_path_current = os.path.join(current_finetune_dir, f'best_model_{best_mean_wmrd:.6f}.pth')
                    torch.save(model.state_dict(), best_model_path_current)
                    np.save(os.path.join(current_finetune_dir, f'preds.npy'), np.concatenate(all_epoch_preds))
                    np.save(os.path.join(current_finetune_dir, f'trues.npy'), np.concatenate(all_epoch_trues))
                    patience_counter = 0  # ✅ 重置计数
                else:
                    patience_counter += 1 # ✅ 累加计数
                
                if patience_counter >= PATIENCE: # ✅ 仅新增：触发早停
                    print(f"🛑 触发早停: 连续 {PATIENCE} 轮指标未提升，停止本块训练。")
                    break

            print(f"✅ 联合训练块完成，最佳平均 WMRD: {best_mean_wmrd:.4f}%")
            model.load_state_dict(torch.load(best_model_path_current))
            
        else:
            print(f"\n🧪 正在独立测试: {dataset_id}")
            dt, lt = read_one_type_da('tr_ts', dataset_id, LABEL_DIR_NAME)
            if dt is None: continue
            test_dir = os.path.join(current_finetune_dir, "test_results", dataset_id)
            os.makedirs(test_dir, exist_ok=True)
            f_p, f_t, mse, mae, r2, wmrd = run_evaluation(model, DataLoader(TensorDataset(dt, lt), batch_size=BATCH_SIZE*2, shuffle=False, num_workers=DATALOADER_NUM_WORKERS), device)
            np.save(os.path.join(test_dir, 'preds.npy'), f_p); np.save(os.path.join(test_dir, 'trues.npy'), f_t)
            with open(os.path.join(test_dir, 'metrics.csv'), 'w', newline='') as f:
                csv.writer(f).writerow(['val_mse', 'val_mae', 'val_r2', 'val_wmrd'])
                csv.writer(f).writerow([mse, mae, r2, wmrd])
            print(f"📊 测试结果 [MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | WMRD: {wmrd:.4f}%]")

    print("\n🎉 大流模型在线学习流水线处理完毕！")