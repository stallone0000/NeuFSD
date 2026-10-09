from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from parse_file_2018 import parse_data_file_c_style
import os
import glob
import re  # 新增：用于正则提取数字进行排序

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def process_dat_file(file_name, base_name, real_dir, chazhi_dir):
    print(f"正在处理: {base_name} ...")
    
    # 统计每个 key 的出现次数
    frequency_dict = parse_data_file_c_style(file_name)

    # 统计每个频率对应的 key 的数量
    frequency_count = defaultdict(int)
    for count in frequency_dict.values():
        frequency_count[count] += 1

    # 按纵坐标（频率）从小到大排序
    sorted_data = sorted([(freq, num_keys) for freq, num_keys in frequency_count.items()], key=lambda x: x[0])

    real_data = [(item[0], item[1]) for item in sorted_data if item[0] <= 10]
    real_data = np.array(real_data)

    # 分解x（频率）和y（num_keys）
    x = np.array([item[0] for item in sorted_data])
    y = np.array([item[1] for item in sorted_data])

    # 创建插值函数，允许外推 (加上了 bounds_error 防止因为边界数据缺失导致报错)
    interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')

    target_freqs = np.arange(1, 11, 1)

    # 计算对应的num_keys
    interpolated_values = interp_func(target_freqs)

    # 结果组合为元组列表，格式 [(频率, 插值后的num_keys), ...]
    result = list(zip(target_freqs, interpolated_values))
    final_result = np.array([key_num for freq_num, key_num in result])
    
    # 保存时使用提取出的 base_name
    np.save(os.path.join(real_dir, f'{base_name}.npy'), real_data)
    np.save(os.path.join(chazhi_dir, f'{base_name}.npy'), final_result)
    

if __name__ == "__main__":
    # 1. 更新为最新的统一输入目录
    import sys as _sys
    input_base_dir = _sys.argv[1] if len(_sys.argv) > 1 else 'caida_1min_split_finetune_continue'
    out_base = _sys.argv[2] if len(_sys.argv) > 2 else 'tr_ts_finetuned_continue'
    
    real_dir = f'{out_base}/1_10_real'
    chazhi_dir = f'{out_base}/1_10_chazhi'
    make_dir(real_dir)
    make_dir(chazhi_dir)
    
    # 2. 放宽匹配规则：获取目录下所有 .dat 文件
    dat_files = glob.glob(os.path.join(input_base_dir, '*.dat'))
    
    if not dat_files:
        print(f"❌ 未在 {input_base_dir} 中找到任何 .dat 文件！")
        exit()
        
    # 3. 使用正则提取文件名中的数字进行排序，保证处理顺序稳定
    dat_files.sort(key=lambda x: int(''.join(re.findall(r'\d+', os.path.basename(x))) or 0))
    
    for file_path in dat_files:
        # 提取文件名（如 dataset_0011 或 fine1_5_key）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 传入 base_name 替代原来的 i, k
        process_dat_file(file_path, base_name, real_dir, chazhi_dir)
        
    print("🎉 全部 1_10 数据插值处理完成！")


    # 1. 更新为最新的统一输入目录
    input_base_dir = 'caida_1min_split'
    
    real_dir = 'tr_ts/1_10_real'
    chazhi_dir = 'tr_ts/1_10_chazhi'
    make_dir(real_dir)
    make_dir(chazhi_dir)
    
    # 2. 放宽匹配规则：获取目录下所有 .dat 文件
    dat_files = glob.glob(os.path.join(input_base_dir, '*.dat'))
    
    if not dat_files:
        print(f"❌ 未在 {input_base_dir} 中找到任何 .dat 文件！")
        exit()
        
    # 3. 使用正则提取文件名中的数字进行排序，保证处理顺序稳定
    dat_files.sort(key=lambda x: int(''.join(re.findall(r'\d+', os.path.basename(x))) or 0))
    
    for file_path in dat_files:
        # 提取文件名（如 dataset_0011 或 fine1_5_key）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 传入 base_name 替代原来的 i, k
        process_dat_file(file_path, base_name, real_dir, chazhi_dir)
        
    print("🎉 全部 1_10 数据插值处理完成！")