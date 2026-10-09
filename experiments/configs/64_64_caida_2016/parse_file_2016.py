import os
from collections import defaultdict
from typing import DefaultDict, Tuple

PACKET_SIZE = 16       # 每个数据包16字节
BUFFER_SIZE = 4096     # 4KB读取缓冲区
KEY_OFFSET = 8         # 密钥从第8字节开始

def parse_data_file_c_style(file_path: str) -> DefaultDict[bytes, int]:
    """与C版本完全一致的读取方式实现的频率统计"""
    frequency_dict = defaultdict(int)
    packet_offset = 0  # 未完成数据包的长度
    buffer = bytearray(BUFFER_SIZE + PACKET_SIZE - 1)  # 带余量的缓冲区

    with open(file_path, 'rb') as f:
        while True:
            # 读取数据并填充到缓冲区尾部
            chunk = f.read(BUFFER_SIZE - packet_offset)
            if not chunk:
                break
            
            # 将新数据拼接到缓冲区尾部
            buffer[packet_offset:packet_offset + len(chunk)] = chunk
            total_bytes = packet_offset + len(chunk)
            
            # 计算完整包数量
            packets = total_bytes // PACKET_SIZE
            for i in range(packets):
                start = i * PACKET_SIZE
                end = start + PACKET_SIZE
                key = bytes(buffer[start + KEY_OFFSET : end])  # 提取后8字节
                frequency_dict[key] += 1
            
            # 保存未完成数据到缓冲区头部
            packet_offset = total_bytes % PACKET_SIZE
            if packet_offset > 0:
                buffer[:packet_offset] = buffer[total_bytes - packet_offset:total_bytes]

    # 处理文件尾部的残留数据（与C版本逻辑一致）
    if packet_offset != 0:
        print(f"警告：末尾有 {packet_offset} 字节不完整数据，已忽略")

    return frequency_dict
