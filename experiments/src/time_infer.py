#!/usr/bin/env python3
# 复现 prev 里训练脚本内嵌的"计时"指标：ViT 每样本推理耗时(model(inputs) 的前向时间)。
# prev 是在训练 eval 时累计 time.time()/(sample_num*epochs);这里做成独立脚本:
# 直接加载训练好的 ckpt + 真实测试 counter,跑前向并测每样本耗时(加 warmup + cuda.synchronize 更准)。
# 用法(PYTHONPATH 指向对应 config 以拿到 model.py):
#   python time_infer.py <RES> <run_exp_dir>     RES=128_128|64_64
import os, sys, time, glob
import numpy as np
import torch
from model import CustomViT

def read_bin_file(filename, width):
    d = np.fromfile(filename, dtype='<u4')
    asc = np.sort(d); desc = asc[::-1]
    return np.concatenate((asc, d, desc))

def main():
    res     = sys.argv[1]                      # 128_128 / 64_64
    run_exp = sys.argv[2]                       # run/<res>_<trace>_exp
    width = 16384 if res == '128_128' else 4096
    img   = 128 if res == '128_128' else 64
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 取一个有 counter 的测试数据集,堆成一个 batch
    in_root = os.path.join(run_exp, 'tr_ts', 'input')
    ds_list = [d for d in sorted(os.listdir(in_root)) if os.listdir(os.path.join(in_root, d))]
    ds = ds_list[0]
    folder = os.path.join(in_root, ds)
    files = sorted(os.listdir(folder), key=lambda x: int(x.split('.')[0]))
    data = np.stack([read_bin_file(os.path.join(folder, f), width) for f in files], 0)
    x = torch.from_numpy(data).float().reshape(-1, 3, img, img).to(dev)
    N = x.shape[0]
    print(f"=== 计时 {res} | 数据集 {ds} | {N} 个 counter 样本 | 输入 3x{img}x{img} | {dev} ===")

    per_head = {}
    for head in ['1_10', '10_1e4']:
        cks = glob.glob(os.path.join(run_exp, '0_finetuned_results',
                        f'ViT_{head}_results_1e-2', 'finetuned_block_*', 'best_model_*.pth'))
        if not cks:
            print(f"  ⚠️ {head} 没找到 ckpt,跳过"); continue
        sd = torch.load(cks[0], map_location=dev)
        out_dim = sd['head.4.weight'].shape[0]            # 从 ckpt 末层推回 out_dim (1_10→10, 10_1e4→1080)
        model = CustomViT(out_dim=out_dim).to(dev); model.load_state_dict(sd); model.eval()
        bs = 64
        with torch.no_grad():
            for _ in range(3): model(x[:bs])             # warmup
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for i in range(0, N, bs): model(x[i:i+bs])
            if dev.type == 'cuda': torch.cuda.synchronize()
            t1 = time.time()
        ps = (t1 - t0) / N
        per_head[head] = ps
        print(f"  {head:6s} head (out_dim={out_dim}): {ps*1e3:.4f} ms/样本")
    total = sum(per_head.values())
    if total:
        print(f"完整方法(两 head 各一次前向)每样本推理: {total*1e3:.4f} ms  (~{1/total:,.0f} 样本/秒)")
        # 落盘
        with open(os.path.join(run_exp, '..', f'{os.path.basename(run_exp).replace("_exp","")}_infer_time.csv'), 'w') as f:
            f.write("head,ms_per_sample\n")
            for h, v in per_head.items(): f.write(f"{h},{v*1e3:.4f}\n")
            f.write(f"total_two_heads,{total*1e3:.4f}\n")

if __name__ == '__main__':
    main()
