import os
import random
import json
from collections import Counter
from tqdm import tqdm

# ================= 配置区域 =================
RANDOM_SEED = 42
INPUT_DIR = "caida_1min_split"
OUTPUT_DIR = f"caida_1min_split_finetune_continue"  # 统一存放在这里
PACKET_SIZE = 16            # 每个原始数据包 16 字节
KEY_OFFSET = 8              # 密钥从第 8 字节开始
KEY_SIZE = 8                # 提取出的 Key 长度为 8 字节
BUFFER_SIZE = 4096          # 4KB 读取缓冲区

SAMPLE_RATE = 0.1          # 流级采样率 10% (1/10) 
MULTIPLIER = 10              # 还原倍数 (放大 10 倍)
# ============================================

# 全局自增 ID，确保不同的训练文件之间，捏造的流 Key 绝对不会重复
global_fake_key_id = 1 

def read_keys_from_16byte_packets(file_path):
    """保留你的 C 风格读取逻辑：读取 16 字节，提取后 8 字节作为 Key"""
    all_keys = []
    buffer = bytearray(BUFFER_SIZE + PACKET_SIZE - 1)
    packet_offset = 0  
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(BUFFER_SIZE - packet_offset)
            if not chunk:
                break
            
            buffer[packet_offset : packet_offset + len(chunk)] = chunk
            total_bytes = packet_offset + len(chunk)
            
            num_packets = total_bytes // PACKET_SIZE
            for i in range(num_packets):
                start = i * PACKET_SIZE
                end = start + PACKET_SIZE
                key = bytes(buffer[start + KEY_OFFSET : end])
                all_keys.append(key)
            
            packet_offset = total_bytes % PACKET_SIZE
            if packet_offset > 0:
                buffer[:packet_offset] = buffer[total_bytes - packet_offset : total_bytes]
                
    if packet_offset != 0:
        pass # 静默忽略

    return all_keys

def process_fsd_synthesis(in_path: str, out_path: str):
    """单文件的 FSD 流级采样与纯伪造合成（生成完整的 16 字节 Packet）"""
    global global_fake_key_id
    
    keys = read_keys_from_16byte_packets(in_path)
    if not keys:
        return
        
    flow_counter = Counter(keys)
    
    # 阶段一：流级采样 10%
    sampled_keys = [k for k in flow_counter.keys() if random.random() < SAMPLE_RATE]
    
    # 阶段二：生成目标 FSD 统计表 (放大 10 倍)
    target_fsd = Counter()
    for key in sampled_keys:
        original_size = flow_counter[key]
        target_fsd[original_size] += MULTIPLIER

    # 阶段三：根据 FSD 纯合成流量
    packets = []
    
    # ======== ✨ 核心修复 ✨ ========
    # 生成 8 字节的无意义 0，用来占位充当“时间戳”
    dummy_timestamp = b'\x00' * KEY_OFFSET 
    # ================================

    for size, flow_count in target_fsd.items():
        for _ in range(flow_count):
            try:
                # 构造 8 字节伪造 Key
                fake_key = global_fake_key_id.to_bytes(KEY_SIZE, byteorder='big')
                global_fake_key_id += 1
                
                # 拼接成 16 字节完整包：[8字节Dummy时间戳] + [8字节假Key]
                fake_packet = dummy_timestamp + fake_key 
                
                # 放入列表
                packets.extend([fake_packet] * size)
            except OverflowError:
                break
                
    # 打乱顺序，模拟真实到达情况
    random.shuffle(packets)
    
    # 写入纯合成文件
    with open(out_path, 'wb') as f:
        f.write(b"".join(packets))

def main():
    random.seed(RANDOM_SEED)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.dat')], 
                   key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    
    if not files:
        print(f"❌ 找不到输入文件于: {INPUT_DIR}")
        return

    print(f"🚀 开始执行: 滑动窗口(5训练 -> 隔5个 -> 10测试) 划分流水线")
    print(f"⚙️ 模式: 16字节包 (前8充当时间戳，后8为Key) | 采样率: {SAMPLE_RATE*100}% | 放大: {MULTIPLIER}x")
    print("-" * 40)
    
    final_json_dict = {}
    group_count = 1
    
    for i in tqdm(range(0, len(files), 10), desc="处理组别进度"):
        train_range = files[i : i+5]
        test_range = files[i+10 : i+20]
        
        if not train_range: break
        
        for fname in train_range:
            in_path = os.path.join(INPUT_DIR, fname)
            fine_name = f"fine_{fname}" 
            out_path = os.path.join(OUTPUT_DIR, fine_name)
            
            process_fsd_synthesis(in_path, out_path)
            final_json_dict[fine_name] = True
            
        for tf in test_range:
            final_json_dict[tf] = False

        if len(test_range) < 10:
            break
            
        group_count += 1

    json_path = "train_test_name_key.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_dict, f, indent=4)

    print("-" * 40)
    print(f"🎉 全部处理完成！")
    print(f"📂 独立的 16 字节完整格式流文件已存入: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()