import os
from typing import List

# --- 配置参数 ---
PACKET_SIZE = 21             # Caida 2018 每个数据包 21 字节
TARGET_PACKETS = 1_000_000   # 每个 1min 窗口的包数
TARGET_BYTES = TARGET_PACKETS * PACKET_SIZE # 每个 1min 窗口的字节数

# 缓冲区大小（建议为 PACKET_SIZE 的倍数，但流式读写下非倍数也不影响准确性）
BUFFER_SIZE = 4 * 1024 * 1024 

def merge_and_split_2018(input_files: List[str], output_dir: str) -> None:
    """针对 21 字节包的拼接、切分及去尾处理"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_index = 0
    current_out_bytes = 0
    
    # 构造输出路径函数
    def get_out_path(idx):
        return os.path.join(output_dir, f"dataset_{idx:04d}.dat")

    out_filename = get_out_path(output_index)
    out_f = open(out_filename, 'wb')

    print(f"开始处理 2018 数据集 (21 bytes/packet)...")

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"  -> [警告] 文件不存在: {file_path}，跳过。")
            continue

        print(f"正在读取拼接: {file_path}")
        with open(file_path, 'rb') as in_f:
            while True:
                # 计算当前输出文件还需要多少字节
                bytes_needed = TARGET_BYTES - current_out_bytes
                
                # 读取 chunk
                read_size = min(BUFFER_SIZE, bytes_needed)
                chunk = in_f.read(read_size)

                if not chunk:
                    break  # 当前输入文件读完

                out_f.write(chunk)
                current_out_bytes += len(chunk)

                # 如果刚好填满一个 1min 窗口
                if current_out_bytes == TARGET_BYTES:
                    out_f.close()
                    output_index += 1
                    
                    # 准备写下一个
                    out_filename = get_out_path(output_index)
                    out_f = open(out_filename, 'wb')
                    current_out_bytes = 0

    # 所有原始文件读完，关闭当前的输出文件
    out_f.close()

    # --- 剔除尾巴逻辑 ---
    # 如果最后一个文件没有写满 TARGET_BYTES，直接删掉
    if current_out_bytes < TARGET_BYTES:
        if os.path.exists(out_filename):
            os.remove(out_filename)
            print(f"\n[清理] 最后一个文件不满完整 1min 窗口，已剔除。")
        final_count = output_index
    else:
        final_count = output_index

    print(f"处理完成！")
    print(f"总共生成了 {final_count} 个完整的 1min 窗口数据集。")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    # 1. 定义 2018 数据集的 10 个原始文件路径
    # 注意：根据你提供的代码，路径是 caida2018/{0-9}.dat
    input_files_list = [f'caida2018/{i}.dat' for i in range(10)]
    
    # 2. 定义输出文件夹
    output_directory = "caida2018_1min_split"
    
    # 3. 执行
    merge_and_split_2018(input_files_list, output_directory)