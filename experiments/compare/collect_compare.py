#!/usr/bin/env python3
# 汇总跨方法对比结果 -> 4 个 plot_compare.py 直接读取的 csv:
#   com_error_mrd.csv / com_error_wmrd.csv   (Memory, elastic, mrac, array, hash, Ours)
#   decode_time.csv  / insert_time.csv        (同上, 单位 ms)
#  - elastic/mrac: 解析 sketch_test 的 stdout txt (MRD / WMRD / Insertion time / decode time)
#  - array/hash : WMRD/MRD/Insert 来自 <KB>_{array,hash}_results.csv; decode 来自 <KB>_{array,hash}.txt
#                 (传统采样器 get_distribution 是 O(1) 直接读, decode≈0ms -> 留空, log 图自动略过)
#  - Ours       : 16KB<-64x64, 64KB<-128x128
#                 误差 = RUN_ROOT 内本次实验的 OVERALL_AVG
#                 decode = 每样本推理时间 (time_infer, res-only)
#                 insert = 留空；需要单独测量 NeuFSD 插入路径，不借用基线时间
# 用法: 在 compare/<trace>/ 下  python ../collect_compare.py <trace>
import sys, re, os, csv, math
from pathlib import Path
import pandas as pd

trace = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get('RUN_ROOT', ROOT / 'run'))
MEMS_KB = [16, 32, 64, 128, 256]
# Counter memory (KB) -> resolution; timings come from the reader's run.
OURS_MAP = {16: '64_64', 64: '128_128'}

def parse_sketch(method, kb):
    """sketch_test stdout -> (mrd, wmrd, decode_ms, insert_ms, packets)"""
    f = f'{method}_{kb*1024}.txt'
    if not os.path.exists(f):
        return (None, None, None, None, None)
    t = open(f).read()
    g = lambda p: (float(re.search(p, t).group(1)) if re.search(p, t) else None)
    return (g(r'MRD:\s*([0-9.eE+-]+)'),
            g(r'WMRD:\s*([0-9.eE+-]+)'),
            g(r'decode\)\s*time:\s*([0-9.eE+-]+)\s*ms'),
            g(r'Insertion time:\s*([0-9.eE+-]+)\s*ms'),
            g(r'Total packets:\s*([0-9]+)'))

def parse_trad(method, kb):
    """traditional_sample -> (mrd, wmrd, decode_ms, insert_ms, packets)
    误差/insert 取 csv (-o), decode 取 stdout txt 的 'Distribution calculation time'."""
    f = f'{kb}_{method}_results.csv'
    if not os.path.exists(f):
        return (None, None, None, None, None)
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    # 列: packets number, memory, method, WMRD, ARE(=MRD), Insert time
    mrd, wmrd, ins = df['ARE'].mean(), df['WMRD'].mean(), df['Insert time'].mean()
    dec = None
    tf = f'{kb}_{method}.txt'
    if os.path.exists(tf):
        m = re.search(r'Distribution calculation time:\s*([0-9.eE+-]+)\s*ms', open(tf).read())
        if m:
            v = float(m.group(1))
            dec = v if v > 0 else None   # 0ms (instant) -> 留空, 避免 log(0)
    packets = df['packets number'].mean() if 'packets number' in df else None
    return (mrd, wmrd, dec, ins, packets)

def throughput_mpps(packets, insert_ms):
    if packets is None or insert_ms is None:
        return None
    if not math.isfinite(packets) or not math.isfinite(insert_ms) or packets < 0 or insert_ms <= 0:
        return None
    return packets / (insert_ms * 1000)

def ours_err(kb, which):  # which: 'mrd'|'wmrd'
    if kb not in OURS_MAP:
        return None
    res = OURS_MAP[kb]
    sm = RUN_ROOT / f'{res}_{trace}_final' / 'plots' / 'pipeline_eval' / 'summary_metrics.csv'
    if not sm.exists():
        return None
    row = [r for r in csv.reader(open(sm)) if r and r[0] == 'OVERALL_AVG'][0]
    return float(row[1]) if which == 'mrd' else float(row[2])  # mrd_avg, wmrd_avg

def ours_decode(kb):
    if kb not in OURS_MAP:
        return None
    path = RUN_ROOT / f'{OURS_MAP[kb]}_{trace}_infer_time.csv'
    if not path.exists():
        return None
    with path.open(newline='') as handle:
        measured = {row['head']: float(row['ms_per_sample'])
                    for row in csv.DictReader(handle)}
    # A partially measured pair is not a complete NeuFSD decode measurement.
    if not {'1_10', '10_1e4', 'total_two_heads'} <= measured.keys():
        return None
    return measured['total_two_heads']

# 先把每个内存的 sketch/trad 结果解析出来
S, T = {}, {}
for kb in MEMS_KB:
    S[kb] = {'elastic': parse_sketch('elastic', kb), 'mrac': parse_sketch('mrac', kb)}
    T[kb] = {'array': parse_trad('array', kb), 'hash': parse_trad('hash', kb)}

rows = {'mrd': [], 'wmrd': [], 'decode': [], 'insert': [], 'throughput': []}
for kb in MEMS_KB:
    e_mrd, e_wmrd, e_dec, e_ins, e_packets = S[kb]['elastic']
    m_mrd, m_wmrd, m_dec, m_ins, m_packets = S[kb]['mrac']
    a_mrd, a_wmrd, a_dec, a_ins, a_packets = T[kb]['array']
    h_mrd, h_wmrd, h_dec, h_ins, h_packets = T[kb]['hash']
    o_dec = ours_decode(kb)
    o_ins = None
    rows['mrd'].append([kb, e_mrd, m_mrd, a_mrd, h_mrd, ours_err(kb, 'mrd')])
    rows['wmrd'].append([kb, e_wmrd, m_wmrd, a_wmrd, h_wmrd, ours_err(kb, 'wmrd')])
    rows['decode'].append([kb, e_dec, m_dec, a_dec, h_dec, o_dec])
    rows['insert'].append([kb, e_ins, m_ins, a_ins, h_ins, o_ins])
    rows['throughput'].append([
        kb, throughput_mpps(e_packets, e_ins), throughput_mpps(m_packets, m_ins),
        throughput_mpps(a_packets, a_ins), throughput_mpps(h_packets, h_ins), None,
    ])

for metric, fn in [('mrd', 'com_error_mrd.csv'), ('wmrd', 'com_error_wmrd.csv'),
                   ('decode', 'decode_time.csv'), ('insert', 'insert_time.csv'),
                   ('throughput', 'throughput_mpps.csv')]:
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Memory', 'elastic', 'mrac', 'array', 'hash', 'Ours'])
        for r in rows[metric]:
            w.writerow([x if x is not None else '' for x in r])
    print(f"  wrote {fn}")

# comb_ViT_1.csv = 我方 128x128 每个测试窗口的 mrd/wmrd (供分布图 paint_MRD_WMRD_distribution)
src_sm = RUN_ROOT / f'128_128_{trace}_final' / 'plots' / 'pipeline_eval' / 'summary_metrics.csv'
if src_sm.exists():
    df = pd.read_csv(src_sm)
    df = df[df['dataset_id'] != 'OVERALL_AVG']
    df.to_csv('comb_ViT_1.csv', index=False)
    print(f"  wrote comb_ViT_1.csv ({len(df)} 个窗口)")
print(f"✅ collected {trace}: elastic/mrac/array/hash + Ours "
      f"-> com_error_{{mrd,wmrd}}.csv + decode_time.csv + insert_time.csv")
