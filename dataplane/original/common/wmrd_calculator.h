// wmrd_calculator.h
#ifndef WMRD_CALCULATOR_H
#define WMRD_CALCULATOR_H

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>

/**
 * 计算WMRD (Weighted Mean Relative Difference)
 * @param real_distribution 实际流量大小分布，索引是流大小，值是该大小的流数量
 * @param estimated_distribution 估计的流量大小分布，索引是流大小，值是该大小的流数量
 * @return WMRD值
 */
double calculate_wmrd(const std::vector<double>& real_distribution, const std::vector<double>& estimated_distribution) {
    // 确定最大流大小
    size_t max_size = std::max(real_distribution.size(), estimated_distribution.size());
    
    double sum_abs_diff = 0.0;
    double sum_weights = 0.0;
    
    // 计算WMRD
    for(size_t i = 1; i < max_size; i++) {
        double real_val = (i < real_distribution.size()) ? real_distribution[i] : 0.0;
        double est_val = (i < estimated_distribution.size()) ? estimated_distribution[i] : 0.0;
        
        // 计算权重 (n_i + n'_i)/2
        double weight = (real_val + est_val) / 2.0;
        
        if(weight > 0) {  // 避免除零
            // 累加 |n_i - n'_i|
            sum_abs_diff += fabs(real_val - est_val);
            // 累加权重
            sum_weights += weight;
        }
    }
    
    // 如果没有有效数据，返回0
    if(sum_weights == 0) {
        return 0.0;
    }
    
    // 计算最终WMRD
    return sum_abs_diff / sum_weights;
}

/**
 * 计算 MRD (Mean Relative Difference, 各流大小桶的相对误差的非加权平均)
 * 与 plot_final.py 的 val_mrd 定义一致: mean( |n_i - n'_i| / ((n_i+n'_i)/2) )
 */
double calculate_mrd(const std::vector<double>& real_distribution, const std::vector<double>& estimated_distribution) {
    size_t max_size = std::max(real_distribution.size(), estimated_distribution.size());
    double sum_rel = 0.0;
    long count = 0;
    for(size_t i = 1; i < max_size; i++) {
        double real_val = (i < real_distribution.size()) ? real_distribution[i] : 0.0;
        double est_val  = (i < estimated_distribution.size()) ? estimated_distribution[i] : 0.0;
        double weight = (real_val + est_val) / 2.0;
        if(weight > 0) { sum_rel += std::fabs(real_val - est_val) / weight; count++; }
    }
    return count ? sum_rel / count : 0.0;
}

/**
 * 从哈希表转换为流大小分布向量
 * @param flow_frequencies 流ID到流大小的映射
 * @return 流大小分布向量，索引是流大小，值是该大小的流数量
 */
std::vector<double> convert_to_distribution(const std::unordered_map<std::string, int>& flow_frequencies) {
    // 找出最大流大小
    int max_flow_size = 0;
    for(const auto& pair : flow_frequencies) {
        max_flow_size = std::max(max_flow_size, pair.second);
    }
    
    // 初始化分布向量
    std::vector<double> distribution(max_flow_size + 1, 0);
    
    // 统计每个大小的流数量
    for(const auto& pair : flow_frequencies) {
        int flow_size = pair.second;
        distribution[flow_size]++;
    }
    
    return distribution;
}

#endif // WMRD_CALCULATOR_H
