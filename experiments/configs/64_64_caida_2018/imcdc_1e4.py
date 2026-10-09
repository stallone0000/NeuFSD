from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from parse_file_2018 import parse_data_file_c_style
import os
import glob
import pandas as pd
import re

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def process_dat_file(file_name, base_name, is_first_file, real_dir, chazhi_dir):
    print(f"正在处理: {base_name} ...")
    
    # 统计每个 key 的出现次数
    frequency_dict = parse_data_file_c_style(file_name)
    # 统计每个频率对应的 key 的数量
    frequency_count = defaultdict(int)
    for count in frequency_dict.values():
        frequency_count[count] += 1

    # 按纵坐标（频率）从小到大排序
    sorted_data = sorted([(freq, num_keys) for freq, num_keys in frequency_count.items()], key=lambda x: x[0])
    real_data = [(item[0], item[1]) for item in sorted_data if item[0] > 10 and item[0] <= 10000]
    real_data = np.array(real_data)
    
    # 原来 i==0 and k==0 的逻辑，现在改为判断是否是第一个处理的文件
    if is_first_file:
        df = pd.DataFrame(real_data, columns=['x', 'y'])
        # 改用 base_name 命名
        df.to_csv(os.path.join(real_dir, f'{base_name}.csv'), index=False)
    
    # 分解x（频率）和y（num_keys）
    x = np.array([item[0] for item in sorted_data])
    y = np.array([item[1] for item in sorted_data])

    # 创建插值函数，允许外推
    interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')

    target_freq_1 = np.arange(11, 1001, 1)
    target_freq_2 = np.arange(1001, 10001, 100)
    target_freqs = np.concatenate((target_freq_1, target_freq_2))

    # 计算对应的num_keys
    interpolated_values = interp_func(target_freqs)

    # 结果组合为元组列表，格式 [(频率, 插值后的num_keys), ...]
    result = list(zip(target_freqs, interpolated_values))
    final_result = np.array([key_num for freq_num, key_num in result])

    # 保存 npy 文件，使用提取出的文件名 
    np.save(os.path.join(real_dir, f'{base_name}.npy'), real_data)
    np.save(os.path.join(chazhi_dir, f'{base_name}.npy'), final_result)


if __name__ == "__main__":
    # 1. 更新为最新的统一输入目录
    import sys as _sys
    input_base_dir = _sys.argv[1] if len(_sys.argv) > 1 else 'caida_1min_split_finetune_continue'
    out_base = _sys.argv[2] if len(_sys.argv) > 2 else 'tr_ts_finetuned_continue'
    
    real_dir = f'{out_base}/10_1e4_real'
    chazhi_dir = f'{out_base}/10_1e4_chazhi'
    make_dir(real_dir)
    make_dir(chazhi_dir)
    
    # 2. 获取目录下所有 .dat 文件 (不再限制 dataset_ 前缀)
    dat_files = glob.glob(os.path.join(input_base_dir, '*.dat'))
    
    if not dat_files:
        print(f"❌ 未在 {input_base_dir} 中找到任何 .dat 文件！")
        exit()
        
    # 3. 使用正则提取文件名中的数字进行排序，保证 dataset_ 和 fine_ 混合时也能大致按顺序处理
    dat_files.sort(key=lambda x: int(''.join(re.findall(r'\d+', os.path.basename(x))) or 0))
    
    is_first_file = True
    for file_path in dat_files:
        # 提取不带后缀的文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 传入 base_name 和 is_first_file 标志
        process_dat_file(file_path, base_name, is_first_file, real_dir, chazhi_dir)
        
        # 处理完第一个文件后，将标志位置为 False
        is_first_file = False
        
    print("🎉 全部数据插值处理完成！")
    
    
    
    input_base_dir = 'caida_1min_split'
    
    real_dir = 'tr_ts/10_1e4_real'
    chazhi_dir = 'tr_ts/10_1e4_chazhi'
    make_dir(real_dir)
    make_dir(chazhi_dir)
    
    # 2. 获取目录下所有 .dat 文件 (不再限制 dataset_ 前缀)
    dat_files = glob.glob(os.path.join(input_base_dir, '*.dat'))
    
    if not dat_files:
        print(f"❌ 未在 {input_base_dir} 中找到任何 .dat 文件！")
        exit()
        
    # 3. 使用正则提取文件名中的数字进行排序，保证 dataset_ 和 fine_ 混合时也能大致按顺序处理
    dat_files.sort(key=lambda x: int(''.join(re.findall(r'\d+', os.path.basename(x))) or 0))
    
    is_first_file = True
    for file_path in dat_files:
        # 提取不带后缀的文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 传入 base_name 和 is_first_file 标志
        process_dat_file(file_path, base_name, is_first_file, real_dir, chazhi_dir)
        
        # 处理完第一个文件后，将标志位置为 False
        is_first_file = False
        
    print("🎉 全部数据插值处理完成！")