import os
import random
import json
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "caida_1min_split"
OUTPUT_DIR = "caida_1min_split_finetune_continue"  # 统一存放在这里
KEY_SIZE = 13               # 每个 Key 固定 13 字节
PACKET_SIZE = 21            # 最终输出的包长度 (为了兼容你的插值脚本)
BUFFER_SIZE = 4096          # 4KB 读取缓冲区
SAMPLE_RATE = 0.1           # 流级采样率 10% (1/10)
MULTIPLIER = 10             # 还原倍数 (1/采样率，即放大 10 倍)

RANDOM_SEED = 42            
# ============================================

# 全局自增 ID，确保不同的训练文件之间，捏造的流 Key 绝对不会重复
global_fake_key_id = 1 

def read_21byte_packets(file_path):
    """按 21 字节 Packet 完整读取单个文件（保留你的缓冲区安全读取逻辑）"""
    all_packets = []
    buffer = bytearray(BUFFER_SIZE + PACKET_SIZE - 1)
    packet_offset = 0
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(BUFFER_SIZE - packet_offset)
            if not chunk: break
            
            buffer[packet_offset : packet_offset + len(chunk)] = chunk
            total_bytes = packet_offset + len(chunk)
            
            num_packets = total_bytes // PACKET_SIZE
            for i in range(num_packets):
                start = i * PACKET_SIZE
                # 提取完整的 PACKET_SIZE (21字节)
                packet = bytes(buffer[start : start + PACKET_SIZE])
                all_packets.append(packet)
            
            packet_offset = total_bytes % PACKET_SIZE
            if packet_offset > 0:
                remainder_start = num_packets * PACKET_SIZE
                buffer[0:packet_offset] = buffer[remainder_start:total_bytes]
                
    if packet_offset != 0:
        print(f"⚠️ 文件 {os.path.basename(file_path)} 末尾有 {packet_offset} 字节不足一个完整包，已忽略")
        
    return all_packets

def process_fsd_synthesis(in_path: str, out_path: str):
    """单文件的 FSD 流级采样与纯伪造合成（保证最终落盘为 21 字节格式）"""
    global global_fake_key_id
    
    # 1. 使用安全的 21 字节读取逻辑
    raw_packets = read_21byte_packets(in_path)
    if not raw_packets:
        return
        
    # 提取所有包的前 13 字节作为 Key 进行统计
    keys = [p[:KEY_SIZE] for p in raw_packets]
    flow_counter = Counter(keys)
    
    # 阶段一：流级采样
    sampled_keys = [k for k in flow_counter.keys() if random.random() < SAMPLE_RATE]
    
    # 阶段二：生成目标 FSD 统计表 (按倍数放大)
    target_fsd = Counter()
    for key in sampled_keys:
        original_size = flow_counter[key]
        target_fsd[original_size] += MULTIPLIER

    # 阶段三：根据 FSD 纯合成流量
    packets = []
    # 构造 8 字节的零填充作为 Payload，保证最终写入包符合 21 字节规范
    payload_padding = b'\x00' * (PACKET_SIZE - KEY_SIZE) 
    
    for size, flow_count in target_fsd.items():
        for _ in range(flow_count):
            try:
                # 构造全局唯一的伪造 Key (13 字节)
                fake_key = global_fake_key_id.to_bytes(KEY_SIZE, byteorder='big')
                global_fake_key_id += 1
                
                # 拼接成符合要求的 21 字节 Packet (13字节Key + 8字节Padding)
                fake_packet = fake_key + payload_padding
                packets.extend([fake_packet] * size)
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
    print(f"⚙️ 采样率: {SAMPLE_RATE*100}% | 目标放大: {MULTIPLIER}x | 包存储: {PACKET_SIZE}字节")
    print("-" * 40)
    
    final_json_dict = {}
    group_count = 1
    
    # 步进：外层循环每次推进 10 个文件
    for i in tqdm(range(0, len(files), 10), desc="处理组别进度"):
        # 训练集：当前起点开始的 5 个文件
        train_range = files[i : i+5]
        # 测试集：当前起点往后推 10 个单位，取 10 个文件
        test_range = files[i+10 : i+20]
        
        if not train_range: break
        
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
    print(f"📂 独立的 21 字节 FSD 微调流文件已存入: {OUTPUT_DIR}/")
    print(f"📄 JSON 配置已生成于当前目录: {json_path}")

if __name__ == "__main__":
    main()