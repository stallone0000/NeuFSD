from collections import Counter

def parse_data_file_c_style(file_name: str, key_size: int = 13):
    """
    极速读取二进制 dat 文件，按 key_size 切分，并返回 {Key: 频次} 的统计字典。
    """
    try:
        with open(file_name, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_name}")
        return Counter()

    total_keys = len(data) // key_size
    if total_keys == 0:
        return Counter()

    # 直接在内存中切片提取所有的 13 字节 Key
    keys = [data[i:i + key_size] for i in range(0, total_keys * key_size, key_size)]
    
    # 使用底层为 C 哈希表的 Counter 进行极速频次统计
    frequency_dict = Counter(keys)
    
    return frequency_dict