import pandas as pd
import matplotlib.pyplot as plt

# 定义键、标签、线条样式和颜色等属性
keys = ["elastic", "mrac", "array", "hash","Ours"]

labels = {
    "elastic": "Elastic", 
    "mrac": "MRAC", 
    "array": "Array",
    "hash": "Hashmap",
    "Ours": "Ours",
}

markers = {"Ours": "o", "elastic": "s", "mrac": "^", "array": "D", "hash": "h"}
tick_size = 19
label_size = 24
marker_size = {"Ours": 14, "elastic": 12, "mrac": 12, "array": 12, "hash": 12}
legend_args = {"bbox_to_anchor": (0.48, 1), "loc": "lower center", "ncol": 2, "handlelength": 2, "prop": {"size": 23, "weight": "bold"}, "frameon": False, "columnspacing": 0.5, "handletextpad": 0.3, "borderpad": 0}
legend_args_tmp = {"bbox_to_anchor": (0.48, 1), "loc": "lower center", "ncol": 3, "handlelength": 1.3, "prop": {"size": 23, "weight": "bold"}, "frameon": False, "columnspacing": 0.4, "handletextpad": 0.2, "borderpad": 0}
savefig_args = {"bbox_inches": "tight", "pad_inches": 0}


def paint_WMRD():
    # 读取数据
    data_path = './com_error_wmrd.csv'
    data = pd.read_csv(data_path)


    # 读取x坐标和各算法误差
    x = data['Memory']
    xtick = list(x)

    # 初始化绘图设置
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='DejaVu Sans')
    plt.figure(figsize=(6, 4))  # 增加高度以容纳上方图例

    # 绘制坐标轴和刻度
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tick_params(labelsize=tick_size)
    plt.xlabel('Memory (KB)', fontweight='bold', fontsize=label_size)
    plt.ylabel('WMRD', fontweight='bold', fontsize=label_size)
    
    # 明确设置X轴刻度位置（根据图片）
    plt.xticks([16, 32, 64, 128, 256])
    
    # 绘制三条曲线（根据图片标记）
    for key in keys:
        y = data[key]
        plt.plot(xtick, y, label=labels[key], linestyle='-', alpha=1, linewidth=2.5, 
                 marker=markers[key], markersize=marker_size[key], 
                 markeredgewidth=2, markerfacecolor='none')

    # ==== 关键修正：正确移动图例到上方 ====
    plt.legend(loc='lower center', 
               bbox_to_anchor=(0.5, 1.05),  # 垂直移动到轴上方，水平居中
               ncol=3, 
               handlelength=3,
               frameon=False
               )  # 无边框
    
    # 设置图例文本格式
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)
    
    # 添加网格线
    plt.grid(True, linestyle='--', axis='y')
    plt.grid(True, linestyle='--', axis='x')
    
    # 设置Y轴范围（根据图片10⁻¹⁰到10⁰范围）
    
    # 调整布局确保图例完整显示
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 顶部留出空间给图例
    plt.savefig("./caida_wmrd_vs_memory.pdf", bbox_inches='tight')
    plt.close()
    plt.close()

####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
def paint_MRD():
    # 读取数据
    data_path = './com_error_mrd.csv'
    data = pd.read_csv(data_path)


    # 读取x坐标和各算法误差
    x = data['Memory']
    xtick = list(x)

    # 初始化绘图设置
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='DejaVu Sans')
    plt.figure(figsize=(6, 4))  # 增加高度以容纳上方图例

    # 绘制坐标轴和刻度
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tick_params(labelsize=tick_size)
    plt.xlabel('Memory (KB)', fontweight='bold', fontsize=label_size)
    plt.ylabel('MRD', fontweight='bold', fontsize=label_size)
    
    # 明确设置X轴刻度位置（根据图片）
    plt.xticks([16, 32, 64, 128, 256])
    plt.ylim((0.2,2.5))
    
    # 绘制三条曲线（根据图片标记）
    for key in keys:
        y = data[key]
        plt.plot(xtick, y, label=labels[key], linestyle='-', alpha=1, linewidth=2.5, 
                 marker=markers[key], markersize=marker_size[key], 
                 markeredgewidth=2, markerfacecolor='none')

    # ==== 关键修正：正确移动图例到上方 ====
    plt.legend(loc='lower center', 
               bbox_to_anchor=(0.5, 1.05),  # 垂直移动到轴上方，水平居中
               ncol=3, 
               handlelength=3,
               frameon=False
               )  # 无边框
    
    # 设置图例文本格式
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)
    
    # 添加网格线
    plt.grid(True, linestyle='--', axis='y')
    plt.grid(True, linestyle='--', axis='x')
    
    # 设置Y轴范围（根据图片10⁻¹⁰到10⁰范围）
    
    # 调整布局确保图例完整显示
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 顶部留出空间给图例
    plt.savefig("./caida_mrd_vs_memory.pdf", bbox_inches='tight')
    plt.close()
    plt.close()


def paint_DECODE_TIME():
    # 读取数据
    data_path = './decode_time.csv'
    data = pd.read_csv(data_path)


    # 读取x坐标和各算法误差
    x = data['Memory']
    xtick = list(x)

    # 初始化绘图设置（匹配图片样式）
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='DejaVu Sans')
    plt.figure(figsize=(6, 4))  # 增加高度以容纳上方图例

    # 设置坐标轴
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tick_params(labelsize=tick_size)
    plt.xlabel('Memory (KB)', fontweight='bold', fontsize=label_size)
    plt.ylabel('Time (s)', fontweight='bold', fontsize=label_size)
    
    # 设置X轴刻度位置（根据图片）
    plt.xticks([16, 32, 64, 128, 256])
    
    # 绘制各算法曲线（转换为秒）
    for key in keys:
        y = data[key] / 1000
        plt.plot(xtick, y, label=labels[key], linestyle='-', alpha=1, linewidth=2.5, 
                 marker=markers[key], markersize=marker_size[key], 
                 markeredgewidth=2, markerfacecolor='none')

    # 移动图例到上方居中
    plt.legend(loc='lower center', 
               bbox_to_anchor=(0.5, 1.05),
               ncol=3, 
               handlelength=3,
               frameon=False)  # 无边框
    
    # 设置图例文本格式
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=20)
    
    # 添加网格线
    plt.grid(True, linestyle='--', axis='y')
    plt.grid(True, linestyle='--', axis='x')
       
    # 调整布局确保图例完整显示
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 顶部留出空间给图例
    plt.savefig("./caida_decode_time_vs_memory.pdf", bbox_inches='tight')
    plt.close()
    plt.close()

def paint_MRD_WMRD_distribution():
    # 加载数据
    data = pd.read_csv('./comb_ViT_1.csv')  # 替换为实际文件路径
    
    # 过滤掉最后一行（平均值）
    data = data[data['dataset_id'] != 'OVERALL_AVG']
    
    # 将dataset_id转换为整数序列 (1-40)
    data['id'] = list(range(1, len(data)+1))
    
    # 设置绘图风格
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='DejaVu Sans')
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale('log')
    
    # 绘制两条曲线
    ax.plot(data['id'], data['mrd_avg'], 
            label='MRD', linestyle='-', linewidth=2.5, 
            color='tab:blue', marker='o', markersize=4, 
            markeredgecolor='tab:blue', markerfacecolor='white')
    
    ax.plot(data['id'], data['wmrd_avg'], 
            label='WMRD', linestyle='-', linewidth=2.5, 
            color='tab:red', marker='s', markersize=4,
            markeredgecolor='tab:red', markerfacecolor='white')
    
    # 设置坐标轴和标签
    ax.set_xlabel('Time', fontweight='bold', fontsize=20)
    ax.set_ylabel('Metric Value', fontweight='bold', fontsize=20)
    ax.set_xticks(range(0, len(data)+1, max(1,len(data)//8)))
    ax.set_xlim(0, len(data)+1)
    ax.tick_params(axis='both', labelsize=12)
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # 添加图表标题
    plt.title('MRD and WMRD Distribution', 
              fontsize=20, fontweight='bold', pad=15)
    
    # 优化布局并保存
    plt.tight_layout()
    plt.savefig("./caida_mrd_wmrd_distri.pdf")
    plt.close()

def paint_MPPS():
    # 读取数据
    data_path = './throughput_mpps.csv'
    data = pd.read_csv(data_path)

    # 读取x坐标
    x = data['Memory']
    xtick = list(x)

    # 初始化绘图设置（匹配图片样式）
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rc('font', family='DejaVu Sans')
    plt.figure(figsize=(6, 4))  # 使用原始尺寸

    # 设置坐标轴
    plt.yscale('log')  # Y轴对数刻度
    plt.xscale('log', base=2)  # X轴log2刻度
    plt.tick_params(labelsize=tick_size)
    plt.xlabel('Memory (KB)', fontweight='bold', fontsize=22)
    plt.ylabel('Throughput (Mpps)', fontweight='bold', fontsize=22)
    
    # 设置X轴刻度位置
    plt.xticks([16, 32, 64, 128, 256])
    
    # 绘制各算法曲线
    for key in keys:
        mpps = data[key]
        
        # 对于Ours算法，只绘制点不连线
        if key == 'Ours':
            plt.plot(xtick, mpps, label=labels[key], linestyle='', alpha=1, linewidth=4.5, 
                     marker=markers[key], markersize=marker_size[key], 
                     markeredgewidth=2, markerfacecolor='none')
        else:
            plt.plot(xtick, mpps, label=labels[key], linestyle='-', alpha=1, linewidth=2.5, 
                     marker=markers[key], markersize=marker_size[key], 
                     markeredgewidth=2, markerfacecolor='none')

    # 移动图例到上方居中
    plt.legend(loc='lower center', 
               bbox_to_anchor=(0.5, 1.05),
               ncol=3, 
               handlelength=3,
               frameon=False)  # 无边框
    
    # 设置图例文本格式
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontweight='bold', fontsize=18)
    
    # 添加网格线
    plt.grid(True, linestyle='--', axis='y')
    plt.grid(True, linestyle='--', axis='x')
       
    # 调整布局确保图例完整显示
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # 顶部留出空间给图例
    
    # 保存图片
    plt.savefig("./throughput_vs_memory.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    paint_DECODE_TIME()
    paint_WMRD()
    paint_MRD()
    from pathlib import Path
    if Path("comb_ViT_1.csv").is_file():
        paint_MRD_WMRD_distribution()
    paint_MPPS()
