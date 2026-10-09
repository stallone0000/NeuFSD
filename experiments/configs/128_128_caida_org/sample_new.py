import os
import random
import json
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "caida_1min_split"
OUTPUT_DIR = "caida_1min_split_finetune_continue"  # 统一存放在这里
KEY_SIZE = 13               # 每个 Key 固定 13 字节
SAMPLE_RATE = 0.1         # 流级采样率 20% (1/5)
MULTIPLIER = 10              # 还原倍数 (1/采样率，即放大 5 倍)

RANDOM_SEED = 42            
# ============================================

# 全局自增 ID，确保不同的训练文件之间，捏造的流 Key 绝对不会重复
global_fake_key_id = 1 

def process_fsd_synthesis(file_path: str, out_path: str):
    """单文件的 FSD 流级采样与纯伪造合成"""
    global global_fake_key_id
    
    with open(file_path, 'rb') as f:
        data = f.read()
    
    total_keys = len(data) // KEY_SIZE
    if total_keys == 0:
        return
        
    # 提取所有 Key 并统计
    keys = [data[i:i + KEY_SIZE] for i in range(0, total_keys * KEY_SIZE, KEY_SIZE)]
    flow_counter = Counter(keys)
    
    # 阶段一：流级采样 20% 的真实流
    sampled_keys = [k for k in flow_counter.keys() if random.random() < SAMPLE_RATE]
    
    # 阶段二：生成目标 FSD 统计表 (放大 5 倍)
    target_fsd = Counter()
    for key in sampled_keys:
        original_size = flow_counter[key]
        target_fsd[original_size] += MULTIPLIER

    # 阶段三：根据 FSD 纯合成流量
    packets = []
    for size, flow_count in target_fsd.items():
        for _ in range(flow_count):
            try:
                # 构造全局唯一的伪造 Key
                fake_key = global_fake_key_id.to_bytes(KEY_SIZE, byteorder='big')
                global_fake_key_id += 1
                packets.extend([fake_key] * size)
            except OverflowError:
                break
                
    # 打乱顺序
    random.shuffle(packets)
    
    # 写入纯合成文件
    with open(out_path, 'wb') as f:
        f.write(b"".join(packets))

def main():
    random.seed(RANDOM_SEED)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 确保文件按数字名严格排序
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.dat')], 
                   key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    
    if not files:
        print(f"❌ 找不到输入文件于: {INPUT_DIR}")
        return

    print(f"🚀 开始执行: 滑动窗口(5训练 -> 隔5个 -> 10测试) 划分流水线")
    print(f"⚙️ 采样率: {SAMPLE_RATE*100}% | 目标放大: {MULTIPLIER}x")
    print("-" * 40)
    
    final_json_dict = {}
    group_count = 1
    
    # 步进：外层循环每次推进 10 个文件
    for i in tqdm(range(0, len(files), 10), desc="处理组别进度"):
        # 训练集：当前起点开始的 5 个文件（例：i=0 时为 1-5，i=10 时为 11-15）
        train_range = files[i : i+5]
        # 测试集：当前起点往后推 10 个单位，取 10 个文件（例：i=0 时为 11-20，i=10 时为 21-30）
        test_range = files[i+10 : i+20]
        
        if not train_range: break
        
        # print(f"\n[第 {group_count} 组] 训练: {train_range[0]}...{train_range[-1]} | 测试: {test_range[0]}...{test_range[-1]}")
        
        # 1. 处理 5 个训练文件（独立生成 FSD 合成文件）
        for fname in train_range:
            in_path = os.path.join(INPUT_DIR, fname)
            fine_name = f"fine_{fname}" 
            out_path = os.path.join(OUTPUT_DIR, fine_name)
            
            process_fsd_synthesis(in_path, out_path)
            
            # 登记为训练集
            final_json_dict[fine_name] = True
            
        # 2. 登记测试文件（直接使用原文件进行测试）
        for tf in test_range:
            final_json_dict[tf] = False

        # 3. 数据见底截断逻辑：如果当前组的测试集不够 10 个，说明到头了
        if len(test_range) < 10:
            print(f"\n⚠️ 发现测试集不足 10 个 (仅剩 {len(test_range)} 个)。本组作为最后一组处理完毕，不再往下分组！")
            break
            
        group_count += 1

    # 保存 JSON
    json_path = "train_test_name_key.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_dict, f, indent=4)

    print("-" * 40)
    print(f"🎉 全部处理完成！共生成了 {group_count} 组滑动评估数据。")
    print(f"📂 独立的 FSD 微调流文件已存入: {OUTPUT_DIR}/")
    print(f"📄 JSON 配置已生成于当前目录: {json_path}")

if __name__ == "__main__":
    main()