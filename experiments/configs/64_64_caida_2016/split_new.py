import os
from typing import List

PACKET_SIZE = 16
TARGET_PACKETS = 1_000_000
# 每个输出文件严格 16,000,000 字节
TARGET_BYTES = TARGET_PACKETS * PACKET_SIZE 

# 缓冲区大小 4MB
BUFFER_SIZE = 4 * 1024 * 1024 

def merge_and_split_stream(input_files: List[str], output_dir: str) -> None:
    """顺序拼接文件并按 1min 窗口切分，自动剔除末尾不足完整 1min 窗口的残余文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_index = 0
    current_out_bytes = 0
    
    # 准备写入第一个文件
    def get_out_path(idx):
        return os.path.join(output_dir, f"dataset_{idx:04d}.dat")

    out_filename = get_out_path(output_index)
    out_f = open(out_filename, 'wb')

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"  -> [警告] 跳过不存在的文件: {file_path}")
            continue
            
        print(f"正在读取并拼接: {file_path}")
        with open(file_path, 'rb') as in_f:
            while True:
                bytes_needed = TARGET_BYTES - current_out_bytes
                read_size = min(BUFFER_SIZE, bytes_needed)
                chunk = in_f.read(read_size)

                if not chunk:
                    break

                out_f.write(chunk)
                current_out_bytes += len(chunk)

                # 当前文件写满一个 1min 窗口
                if current_out_bytes == TARGET_BYTES:
                    out_f.close()
                    output_index += 1
                    
                    # 准备下一个文件
                    out_filename = get_out_path(output_index)
                    out_f = open(out_filename, 'wb')
                    current_out_bytes = 0

    # 所有输入文件读取完毕，关闭当前输出文件
    out_f.close()

    # --- 核心修改：删掉最后一点尾巴 ---
    # 如果 current_out_bytes < TARGET_BYTES，说明最后一个文件没写满
    if current_out_bytes < TARGET_BYTES:
        if os.path.exists(out_filename):
            os.remove(out_filename)
        print(f"\n[清理] 已删除末尾不足完整 1min 窗口的碎片文件。")
        final_count = output_index
    else:
        # 恰好整除的情况
        final_count = output_index

    print(f"全部处理完成！")
    print(f"最终生成的完整数据集文件总数: {final_count}")
    print(f"文件存储在: {output_dir}")


if __name__ == "__main__":
    # 原始文件列表
    input_files_list = [f'caida2016/formatted0{i}.dat' for i in range(10)]
    
    # 输出目录
    output_directory = "caida2016_1min_split"
    
    merge_and_split_stream(input_files_list, output_directory)