import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from scipy.interpolate import interp1d

# ================= 核心配置区域 =================
JSON_PATH = 'train_test_name_key.json'
ROOT_DIR = '0_finetuned_results'     
MODEL_NAME = 'ViT'                   
PLOT_SAVE_DIR = 'plots/pipeline_eval'
# ✅ 拼接点：神经网络只管 1 到 100，后续全交接给 EL
SPLICE_POINT = 100                 
# ================================================

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def load_csv_analysis(el_folder_idx, sep_idxs, threshold):
    df = pd.read_csv(f'EL/{el_folder_idx}/heavy_{sep_idxs}.csv')
    
    count_nums = df['count'].values
    count_nums = count_nums[count_nums > threshold] 
    count_nums.sort()
    
    value_counts = pd.Series(count_nums).value_counts().sort_index()
    values = value_counts.index.to_numpy()
    frequencies = value_counts.values
    if len(values) == 0:
        return np.empty((0, 2), dtype=float)
    flited_val_freq = np.concatenate((values, frequencies), axis=0).reshape(2,-1).T
    return flited_val_freq

def main():
    make_dir(PLOT_SAVE_DIR)
    
    with open(JSON_PATH, 'r') as f:
        tasks = json.load(f)

    test_mapping = {}
    current_train_block = None
    
    # 自动锁定连续 True 块的“首个” dataset，用于物理读取路径
    in_train_block = False  
    
    for filename, is_train in tasks.items():
        ds_name = filename.split('.')[0]
        if is_train:
            # 如果刚遇到 True（即不在训练块中），记录它作为整个 Block 的代表
            if not in_train_block:
                current_train_block = ds_name
                in_train_block = True
        else:
            # 遇到 False，断开连续的 True 状态，进入测试集映射
            in_train_block = False
            if current_train_block is not None:
                test_mapping[ds_name] = current_train_block

    mrd_avg_all = []
    wmrd_avg_all = []
    closest_preds_dict = {}
    processed_test_ids = []

    print(f"🚀 开始处理流水线画图评估，共发现 {len(test_mapping)} 个测试集...")

    for dataset_id, block_name in test_mapping.items():
        print('='*80)
        
        # 直接从 dataset_XXXX 提取最后的数字
        el_folder_idx = str(int(dataset_id.split('_')[-1]))
        
        print(f'🔥 Processing Dataset: {dataset_id} (所属物理文件夹: {block_name} | EL 文件夹: {el_folder_idx})')

        path1 = f'{ROOT_DIR}/{MODEL_NAME}_1_10_results_1e-2/finetuned_block_{block_name}/test_results/{dataset_id}/preds.npy'
        path2 = f'{ROOT_DIR}/{MODEL_NAME}_10_1e4_results_1e-2/finetuned_block_{block_name}/test_results/{dataset_id}/preds.npy'

        if not os.path.exists(path1) or not os.path.exists(path2):
            raise FileNotFoundError(f"Missing complete predictions for {dataset_id}: {path1} / {path2}")

        preds1 = np.load(path1)
        preds2 = np.load(path2)
        corr_preds = np.column_stack((preds1, preds2))

        real_trues_1 = np.load(f'tr_ts/1_10_real/{dataset_id}.npy')
        real_trues_2 = np.load(f'tr_ts/10_1e4_real/{dataset_id}.npy')
        real_trues = np.vstack((real_trues_1, real_trues_2))

        val_mrd_ls = []
        val_wmrd_ls = []
        interp_ls = []
        # EL output is identical for every seed prediction of the same dataset.
        flited_val_freq = load_csv_analysis(el_folder_idx, 0, 99)

        for sep_idx, sep_preds in enumerate(corr_preds):
            target_freqs = np.concatenate((
                np.arange(1, 1001, 1),       
                np.arange(1001, 10001, 100)  
            ))
            
            sep_preds[sep_preds < 0] = 0  
            interp_func = interp1d(target_freqs, sep_preds, kind='linear', fill_value='extrapolate')
            
            interpolated_preds = interp_func(np.arange(1, SPLICE_POINT + 1, 1))
            
            filted_tgt_freqs = flited_val_freq[:,0]
            filted_sep_preds = flited_val_freq[:,1]
            if len(filted_tgt_freqs) == 0:
                filted_interpolated_preds = np.zeros(20000 - SPLICE_POINT)
            elif len(filted_tgt_freqs) == 1:
                filted_interpolated_preds = np.full(20000 - SPLICE_POINT, filted_sep_preds[0])
            else:
                filted_interp_func = interp1d(filted_tgt_freqs, filted_sep_preds, kind='linear', fill_value='extrapolate')
                filted_interpolated_preds = filted_interp_func(np.arange(SPLICE_POINT + 1, 20001, 1))
            
            cat_interpolated_preds = np.concatenate((interpolated_preds, filted_interpolated_preds))

            final_inter_preds = cat_interpolated_preds[real_trues[:,0]-1]
            final_inter_preds = np.around(final_inter_preds, 0)
            final_inter_preds[final_inter_preds < 0] = 0  
            interp_ls.append(final_inter_preds)
            
            val_mrd = np.around(np.mean(np.abs(final_inter_preds - real_trues[:,1]) / ((final_inter_preds + real_trues[:,1])/2)), 6)
            val_wmrd = np.around(np.mean(np.abs(final_inter_preds - real_trues[:,1])) / np.mean((final_inter_preds + real_trues[:,1])/2), 6)
            val_mrd_ls.append(val_mrd)
            val_wmrd_ls.append(val_wmrd)

        wmrd_avg = np.mean(val_wmrd_ls)
        mrd_avg = np.mean(val_mrd_ls)
        mrd_avg_all.append(mrd_avg)
        wmrd_avg_all.append(wmrd_avg)
        processed_test_ids.append(dataset_id)

        differences = np.abs(np.array(val_wmrd_ls) - wmrd_avg)
        closest_idx = np.argmin(differences)

        closest_preds_dict[dataset_id] = {
            'pred': interp_ls[closest_idx],
            'mrd': val_mrd_ls[closest_idx],
            'wmrd': val_wmrd_ls[closest_idx],
            'block': block_name
        }
        print(f"   ↳ MRD: {val_mrd_ls[closest_idx]:.6f} | WMRD: {val_wmrd_ls[closest_idx]:.6f}")

    # ================= 保存总体统计报表 =================
    if not test_mapping:
        print("⚠️ 没有发现任何需要处理的测试集。")
        return

    df = pd.DataFrame({
        'dataset_id': processed_test_ids + ['OVERALL_AVG'],
        'mrd_avg': mrd_avg_all + [np.mean(mrd_avg_all)],
        'wmrd_avg': wmrd_avg_all + [np.mean(wmrd_avg_all)]
    })
    df.to_csv(f'{PLOT_SAVE_DIR}/summary_metrics.csv', index=False)
    
    print('\n' + '='*80)
    print(f'🎉 所有数据评估完毕! 整体平均 MRD: {np.mean(mrd_avg_all):.6f} | 整体平均 WMRD: {np.mean(wmrd_avg_all):.6f}')
    print(f'📊 正在绘制 2x5 网格图像到目录: {PLOT_SAVE_DIR}/')

    # ================= 绘制 2x5 网格大图 =================
    dataset_items = list(closest_preds_dict.items())
    num_items = len(dataset_items)
    items_per_fig = 10  # 更改为 10 个一组
    num_figs = int(np.ceil(num_items / items_per_fig))

    for fig_idx in range(num_figs):
        # 创建 2x5 画布，拉大宽度 (25) 和高度 (10)，防止图表挤压
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()
        
        start_idx = fig_idx * items_per_fig
        end_idx = min(start_idx + items_per_fig, num_items)
        current_chunk = dataset_items[start_idx:end_idx]
        
        for ax_idx in range(10):  # 循环 10 次填满 2x5 网格
            ax = axes[ax_idx]
            
            if ax_idx < len(current_chunk):
                dataset_id, info = current_chunk[ax_idx]
                
                real_trues_1 = np.load(f'tr_ts/1_10_real/{dataset_id}.npy')
                real_trues_2 = np.load(f'tr_ts/10_1e4_real/{dataset_id}.npy')
                real_trues = np.vstack((real_trues_1, real_trues_2))

                final_inter_preds = info['pred']
                
                ax.plot(np.log10(real_trues[:,0]), np.log10(real_trues[:,1]), label='True')
                
                drop_index = np.where(final_inter_preds == 0)
                adjusted_preds = np.delete(final_inter_preds, drop_index)
                adjusted_trues = np.delete(real_trues[:,0], drop_index) 
                        
                ax.plot(np.log10(adjusted_trues), np.log10(adjusted_preds), label='Predicted')
                
                ax.set_xlabel('log10(Flow Size)')
                ax.set_xlim(-0.05, 5)
                ax.set_ylabel('log10(Frequency)')
                ax.set_title(f'[{info["block"]}] {dataset_id}\nMRD: {info["mrd"]}, WMRD: {info["wmrd"]}')
                ax.legend()
            else:
                # 如果这组不够 10 个（比如最后一张图），隐藏多出来的空白子图框
                ax.set_visible(False)
        
        # 自动调整子图间距
        plt.tight_layout()
        plt.savefig(f'{PLOT_SAVE_DIR}/grid_plot_batch_{fig_idx + 1}.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
