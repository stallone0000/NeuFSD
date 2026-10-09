import os
from typing import List

# --- 新任务配置参数 ---
KEY_SIZE = 13                # 每个 Key 固定 13 字节
TARGET_KEYS = 1_000_000      # 每个 1min 窗口的 Key 数
TARGET_BYTES = TARGET_KEYS * KEY_SIZE  # 每个 1min 窗口的字节数

# 缓冲区大小 4MB
BUFFER_SIZE = 4 * 1024 * 1024 

def merge_split_and_clean_caida(input_files: List[str], output_dir: str):
    """
    顺序拼接 caida 文件夹下的原始文件，按 1min 窗口切分，并删除末尾不足额文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_index = 0
    current_out_bytes = 0
    
    def get_out_path(idx):
        return os.path.join(output_dir, f"dataset_{idx:04d}.dat")

    # 初始化第一个输出文件
    out_filename = get_out_path(output_index)
    out_f = open(out_filename, 'wb')

    print(f"开始流式切分数据 (每 Key 13 字节)...")

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
                    break

                out_f.write(chunk)
                current_out_bytes += len(chunk)

                # 如果当前输出文件刚好达到一个 1min 窗口
                if current_out_bytes == TARGET_BYTES:
                    out_f.close()
                    output_index += 1
                    
                    # 准备写下一个数据集文件
                    out_filename = get_out_path(output_index)
                    out_f = open(out_filename, 'wb')
                    current_out_bytes = 0

    # 关闭最后一个文件
    out_f.close()

    # --- 删尾巴逻辑 ---
    # 检查最后一个文件是否写满一个 1min 窗口
    if current_out_bytes < TARGET_BYTES:
        if os.path.exists(out_filename):
            os.remove(out_filename)
            print(f"\n[清理] 最后一个文件不满完整 1min 窗口，已删除。")
        final_count = output_index
    else:
        final_count = output_index

    print("-" * 30)
    print(f"处理完成！")
    print(f"共生成完整的 1min 窗口数据集: {final_count} 个")
    print(f"保存路径: {output_dir}")

if __name__ == "__main__":
    # 1. 根据你的截图，输入文件在 caida/ 目录下，从 0.dat 到 10.dat
    # 这里使用列表生成式严格保证 0-10 的顺序
    input_files_list = [f'caida/{i}.dat' for i in range(11)]
    
    # 2. 设置输出文件夹
    output_directory = "caida_1min_split"
    
    # 3. 执行拼接与切分
    merge_split_and_clean_caida(input_files_list, output_directory)