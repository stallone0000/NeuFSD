#!/usr/bin/env python3
# 复现 prev 里的 obtain_counter_plot.py：把 counter 数组当成 3 通道"图"渲染成 PNG，
# 直观展示 ViT 看到的输入(升序/原始/降序 三通道)。
# 用法: python obtain_counter_plot.py <input_dir> <out_dir> <img_size> [num_samples]
#   input_dir = .../tr_ts/input/<dataset_id>   (里面是 0.bin,1.bin,... 每个 16384 或 4096 个 uint32)
#   img_size  = 128 (counter 16384) 或 64 (counter 4096)
import os, sys
import numpy as np
from PIL import Image

def read_bin_file(filename, width):
    data = np.fromfile(filename, dtype='<u4')
    if len(data) != width:
        print(f"⚠️ 文件长度异常: {len(data)} != {width}")
    asc = np.sort(data); org = data; desc = np.sort(data)[::-1]
    return np.concatenate((asc, org, desc))   # 3*width

def main():
    input_dir = sys.argv[1]
    out_dir   = sys.argv[2]
    img_size  = int(sys.argv[3])              # 128 或 64
    n         = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    width = img_size * img_size               # 16384 或 4096
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(os.listdir(input_dir), key=lambda x: int(x.split('.')[0]))[:n]
    for i, f in enumerate(files):
        arr = read_bin_file(os.path.join(input_dir, f), width)     # (3*width,)
        img = arr.reshape(3, img_size, img_size).astype(np.float64)
        mn, mx = img.min(), img.max()
        norm = (img - mn) / (mx - mn) if mx != mn else np.zeros_like(img)
        u8 = (norm * 255).astype(np.uint8).transpose(1, 2, 0)      # (H,W,3)
        Image.fromarray(u8, 'RGB').save(os.path.join(out_dir, f'sample_{i}.png'))
    print(f"✅ counter 图已保存 {len(files)} 张到 {out_dir}/ (img_size={img_size})")

if __name__ == '__main__':
    main()
