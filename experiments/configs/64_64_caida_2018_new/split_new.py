import os
from typing import List

# --- Caida 2018 配置参数 ---
PACKET_SIZE = 21             # 2018 数据集每个包 21 字节
TARGET_PACKETS = 1_000_000   # 每个 1min 窗口的包数
TARGET_BYTES = TARGET_PACKETS * PACKET_SIZE  # 每个 1min 窗口的字节数

# 缓冲区大小，建议为 4MB
BUFFER_SIZE = 4 * 1024 * 1024 

def merge_split_and_clean_2018(input_files: List[str], output_dir: str):
    """
    顺序拼接 2018 的原始文件，按 1min 窗口切分，并删除最后的不足额尾巴
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_index = 0
    current_out_bytes = 0
    
    # 辅助函数：生成输出文件名
    def get_out_path(idx):
        return os.path.join(output_dir, f"dataset_{idx:04d}.dat")

    # 初始化第一个输出文件
    out_filename = get_out_path(output_index)
    out_f = open(out_filename, 'wb')

    print(f"开始流式切分 Caida 2018 数据 (每包 21 字节)...")

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"  -> [跳过] 文件不存在: {file_path}")
            continue
            
        print(f"正在处理: {file_path}")
        with open(file_path, 'rb') as in_f:
            while True:
                # 计算当前输出文件还需要多少字节写满
                bytes_needed = TARGET_BYTES - current_out_bytes
                
                # 限制读取量，确保不会跨过 1min 窗口的边界
                read_size = min(BUFFER_SIZE, bytes_needed)
                chunk = in_f.read(read_size)

                if not chunk:
                    break  # 当前输入文件读完，换下一个

                out_f.write(chunk)
                current_out_bytes += len(chunk)

                # 如果当前输出文件刚好达到了 一个 1min 窗口
                if current_out_bytes == TARGET_BYTES:
                    out_f.close()
                    output_index += 1
                    
                    # 准备写下一个数据集文件
                    out_filename = get_out_path(output_index)
                    out_f = open(out_filename, 'wb')
                    current_out_bytes = 0

    # 循环结束，关闭最后一个正在写入的文件
    out_f.close()

    # --- 删尾巴逻辑 ---
    # 如果最后一个文件的大小不足 TARGET_BYTES，说明它是不足完整 1min 窗口的尾巴
    if current_out_bytes < TARGET_BYTES:
        if os.path.exists(out_filename):
            os.remove(out_filename)
            print(f"\n[清理] 已删除末尾残余文件 (包含 {current_out_bytes // PACKET_SIZE} 个 packet)。")
        final_count = output_index
    else:
        final_count = output_index

    print("-" * 30)
    print(f"处理完成！")
    print(f"共生成完整的 1min 窗口数据集: {final_count} 个")
    print(f"保存路径: {output_dir}")

if __name__ == "__main__":
    # 1. 设置输入文件路径（按照 0.dat 到 9.dat 的顺序）
    input_files = [f'caida2018_new/{i}.dat' for i in range(10)]
    
    # 2. 设置输出文件夹
    output_dir = "caida2018_1min_split"
    
    # 3. 执行
    merge_split_and_clean_2018(input_files, output_dir)