#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <memory>
#include <algorithm>
#include <chrono>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string>
#include <functional>
#include <fstream>
#include <iomanip>
#include "src/common/BOBHash32.h"

using namespace std;

static constexpr double FIXED_FLOW_SAMPLE_RATE = 0.01;
static constexpr uint32_t SAMPLE_DENOMINATOR = 1000000;
static constexpr uint32_t SAMPLE_THRESHOLD = 10000; // 1% of SAMPLE_DENOMINATOR.

// 定义五元组结构
struct FIVE_TUPLE { 
    char key[13]; 
};

// 全局变量
vector<vector<FIVE_TUPLE>> traces;
ofstream csv_file; // CSV文件输出流

// 读取数据文件的函数
void ReadInTraces(const char *trace_prefix, int start_file_no, int end_file_no) {
    traces.resize(end_file_no - start_file_no + 1);
    
    for(int datafileCnt = start_file_no; datafileCnt <= end_file_no; ++datafileCnt) {
        char datafileName[4096];
        snprintf(datafileName, sizeof(datafileName), "%s%d.dat", trace_prefix, datafileCnt);
        FILE *fin = fopen(datafileName, "rb");

        if (!fin) {
            printf("Failed to open file: %s\n", datafileName);
            exit(1);
        }

        // 定义21字节读取缓冲区 (13字节五元组 + 8字节时间戳)
        uint8_t buffer[21];
        traces[datafileCnt - start_file_no].clear();

        while(fread(buffer, 1, 13, fin) == 13) {
            FIVE_TUPLE tmp_five_tuple;
            memcpy(tmp_five_tuple.key, buffer, 13); // 拷贝13字节五元组
            traces[datafileCnt - start_file_no].push_back(tmp_five_tuple);
        }
        fclose(fin);

        printf("Successfully read in %s, %ld packets\n", datafileName, traces[datafileCnt - start_file_no].size());
    }
    printf("\n");
}

// 基础采样器接口
class BaseSampler {
public:
    virtual ~BaseSampler() {}
    virtual void insert(const FIVE_TUPLE& key) = 0;
    virtual int query(const FIVE_TUPLE& key) = 0;
    virtual void get_distribution(vector<double>& dist) = 0;
    virtual size_t memory_usage() const = 0;
    virtual const char* name() const = 0;
    virtual void print_stats() const = 0;
    virtual double get_sample_rate() const = 0;
};

// 基于数组的采样器（方案1）- 使用BOBHash32
class ArraySampler : public BaseSampler {
private:
    struct Bucket {
        FIVE_TUPLE key;
        int count;
        bool occupied;
    };
    
    vector<Bucket> buckets;
    size_t bucket_size;
    double sample_rate;
    size_t total_inserts;
    size_t collisions;
    BOBHash32* bob_hash;
    BOBHash32* sample_hash;
    
public:
    ArraySampler(size_t memory_bytes, size_t total_flows)
        : total_inserts(0), collisions(0) {
        
        // 计算桶数量：内存(位) / (104位键 + 32位计数器)
        bucket_size = (memory_bytes * 8) / (104 + 32);
        if (bucket_size == 0) bucket_size = 1;
        
        sample_rate = FIXED_FLOW_SAMPLE_RATE;
        
        buckets.resize(bucket_size);
        
        for (auto& bucket : buckets) {
            bucket.occupied = false;
            bucket.count = 0;
        }
        
        bob_hash = new BOBHash32(131);
        sample_hash = new BOBHash32(17);
    }
    ~ArraySampler() {
        delete bob_hash;
        delete sample_hash;
    }
    
    double get_sample_rate() const override {
        return sample_rate;
    }
    
    // 使用BOBHash32计算哈希值
    size_t hash_function(const FIVE_TUPLE& key) const {
        return bob_hash->run(key.key, 13) % bucket_size;
    }

    bool sampled_flow(const FIVE_TUPLE& key) const {
        return (sample_hash->run(key.key, 13) % SAMPLE_DENOMINATOR) < SAMPLE_THRESHOLD;
    }
    
    void insert(const FIVE_TUPLE& key) override {
        if (!sampled_flow(key)) return;
        
        total_inserts++;
        
        // 使用哈希函数计算初始位置
        size_t index = hash_function(key);
        size_t original_index = index;
        
        // 线性探测解决冲突
        do {
            if (!buckets[index].occupied) {
                buckets[index].key = key;
                buckets[index].count = 1;
                buckets[index].occupied = true;
                return;
            }
            
            if (memcmp(buckets[index].key.key, key.key, 13) == 0) {
                buckets[index].count++;
                return;
            }
            
            index = (index + 1) % bucket_size;
        } while (index != original_index);
        
        // 所有桶都满了 - 碰撞
        collisions++;
    }
    
    int query(const FIVE_TUPLE& key) override {
        // 使用哈希函数计算初始位置
        size_t index = hash_function(key);
        size_t original_index = index;
        
        // 线性探测查找
        do {
            if (buckets[index].occupied && 
                memcmp(buckets[index].key.key, key.key, 13) == 0) {
                return buckets[index].count / sample_rate;
            }
            index = (index + 1) % bucket_size;
        } while (index != original_index);
        
        return 0;
    }
    
    void get_distribution(vector<double>& dist) override {
        unordered_map<int, int> freq_count;
        int total_sampled = 0;
        
        for (const auto& bucket : buckets) {
            if (bucket.occupied) {
                freq_count[bucket.count]++;
                total_sampled++;
            }
        }
        
        if (total_sampled == 0) return;
        
        // 找到最大频率以确定分布大小
        int max_freq = 0;
        for (const auto& pair : freq_count) {
            if (pair.first > max_freq) max_freq = pair.first;
        }
        
        dist.resize(max_freq + 1, 0.0);
        for (const auto& pair : freq_count) {
            dist[pair.first] = static_cast<double>(pair.second) / sample_rate;
        }
    }
    
    size_t memory_usage() const override {
        // 每个桶：104位键 + 32位计数器 = 136位 = 17字节
        return bucket_size * 17;
    }
    
    const char* name() const override {
        return "array";
    }
    
    void print_stats() const override {
        size_t occupied_buckets = 0;
        for (const auto& bucket : buckets) {
            if (bucket.occupied) occupied_buckets++;
        }
        
        cout << "Array Sampler Statistics:" << endl;
        cout << "  Bucket size: " << bucket_size << endl;
        cout << "  Occupied buckets: " << occupied_buckets << endl;
        cout << "  Total inserts: " << total_inserts << endl;
        cout << "  Collisions: " << collisions << endl;
        cout << "  Memory usage: " << memory_usage() << " bytes" << endl;
        cout << "  Sample rate: " << sample_rate << endl;
    }
};

// 基于哈希表的采样器（方案2）- 使用BOBHash32
class HashMapSampler : public BaseSampler {
private:
    struct Hash {
        BOBHash32* bob_hash;
        
        Hash() {
            bob_hash = new BOBHash32(rand() % 1229); // MAX_PRIME32=1229
        }
        
        ~Hash() {
            delete bob_hash;
        }
        
        size_t operator()(const FIVE_TUPLE& key) const {
            return bob_hash->run(key.key, 13);
        }
    };
    
    struct Equal {
        bool operator()(const FIVE_TUPLE& a, const FIVE_TUPLE& b) const {
            return memcmp(a.key, b.key, 13) == 0;
        }
    };
    
    unordered_map<FIVE_TUPLE, int, Hash, Equal> hash_map;
    double sample_rate;
    size_t total_inserts;
    size_t max_entries;
    size_t dropped_new_flows;
    BOBHash32* sample_hash;
    
public:
    HashMapSampler(size_t memory_bytes, size_t total_flows)
        : total_inserts(0), max_entries(0), dropped_new_flows(0) {
        
        // 计算最大元素数：内存(位) / (104位键 + 32位计数器 + 200位开销)
        max_entries = (memory_bytes * 8) / (104 + 32 + 200);
        if (max_entries == 0) max_entries = 1;
        
        sample_rate = FIXED_FLOW_SAMPLE_RATE;
        
        // 预留空间以提高性能
        hash_map.reserve(max_entries);
        sample_hash = new BOBHash32(17);
    }

    ~HashMapSampler() {
        delete sample_hash;
    }
    
    double get_sample_rate() const override {
        return sample_rate;
    }

    bool sampled_flow(const FIVE_TUPLE& key) const {
        return (sample_hash->run(key.key, 13) % SAMPLE_DENOMINATOR) < SAMPLE_THRESHOLD;
    }
    
    void insert(const FIVE_TUPLE& key) override {
        if (!sampled_flow(key)) return;
        
        total_inserts++;
        auto it = hash_map.find(key);
        if (it != hash_map.end()) {
            it->second++;
            return;
        }
        if (hash_map.size() < max_entries) {
            hash_map.emplace(key, 1);
        } else {
            dropped_new_flows++;
        }
    }
    
    int query(const FIVE_TUPLE& key) override {
        auto it = hash_map.find(key);
        if (it != hash_map.end()) {
            return it->second / sample_rate;
        }
        return 0;
    }
    
    void get_distribution(vector<double>& dist) override {
        unordered_map<int, int> freq_count;
        int total_sampled = 0;
        
        for (const auto& pair : hash_map) {
            freq_count[pair.second]++;
            total_sampled++;
        }
        
        if (total_sampled == 0) return;
        
        // 找到最大频率以确定分布大小
        int max_freq = 0;
        for (const auto& pair : freq_count) {
            if (pair.first > max_freq) max_freq = pair.first;
        }
        
        dist.resize(max_freq + 1, 0.0);
        for (const auto& pair : freq_count) {
            dist[pair.first] = static_cast<double>(pair.second) / sample_rate;
        }
    }
    
    size_t memory_usage() const override {
        // 估算内存使用：每个条目大约 104位键 + 32位计数器 + 200位开销 = 336位 = 42字节
        return max_entries * 42;
    }
    
    const char* name() const override {
        return "hash";
    }
    
    void print_stats() const override {
        cout << "Hash Map Sampler Statistics:" << endl;
        cout << "  Unique flows: " << hash_map.size() << endl;
        cout << "  Max entries: " << max_entries << endl;
        cout << "  Dropped new flows: " << dropped_new_flows << endl;
        cout << "  Total inserts: " << total_inserts << endl;
        cout << "  Memory usage: " << memory_usage() << " bytes" << endl;
        cout << "  Sample rate: " << sample_rate << endl;
    }
};

// 计算真实频率分布
void real_distribution(const vector<FIVE_TUPLE>& trace, vector<double>& dist) {
    unordered_map<string, int> freq_map;
    unordered_map<int, int> freq_count;
    int max_freq = 0;
    
    for (const auto& packet : trace) {
        string key(packet.key, 13);
        freq_map[key]++;
    }
    
    for (const auto& pair : freq_map) {
        freq_count[pair.second]++;
        if (pair.second > max_freq) max_freq = pair.second;
    }
    
    dist.resize(max_freq + 1, 0.0);
    for (const auto& pair : freq_count) {
        dist[pair.first] = static_cast<double>(pair.second);
    }
}

// 修改后的WMRD计算函数（匹配您提供的计算公式）
double calculate_wmrd(const vector<double>& real_distribution, const vector<double>& estimated_distribution) {
    // 确定最大流大小
    size_t max_size = max(real_distribution.size(), estimated_distribution.size());
    
    double sum_abs_diff = 0.0;
    double sum_weights = 0.0;
    
    // 计算WMRD（从1开始忽略频率0）
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

double calculate_mrd(const vector<double>& real_distribution, const vector<double>& estimated_distribution) {
    size_t max_size = max(real_distribution.size(), estimated_distribution.size());
    double total_error = 0.0;
    int count = 0;

    for(size_t i = 1; i < max_size; i++) {
        double real_val = (i < real_distribution.size()) ? real_distribution[i] : 0.0;
        double est_val = (i < estimated_distribution.size()) ? estimated_distribution[i] : 0.0;
        double denominator = (real_val + est_val) / 2.0;

        if (denominator > 0) {
            double error = fabs(real_val - est_val) / denominator;
            total_error += error;
            count++;
        }
    }

    if (count == 0) return 0.0;
    return total_error / count;
}

// 评估采样器性能
void evaluate_sampler(BaseSampler& sampler, const vector<FIVE_TUPLE>& trace, 
                      const vector<double>& real_dist, int file_index) {
    auto start = chrono::high_resolution_clock::now();
    
    // 插入所有数据包
    for (const auto& packet : trace) {
        sampler.insert(packet);
    }
    
    auto insert_end = chrono::high_resolution_clock::now();
    
    // 获取分布
    vector<double> est_dist;
    sampler.get_distribution(est_dist);
    
    auto dist_end = chrono::high_resolution_clock::now();
    
    // 计算WMRD
    double wmrd = calculate_wmrd(real_dist, est_dist);
    
    double are = calculate_mrd(real_dist, est_dist);
    
    // 计算查询时间
    auto query_start = chrono::high_resolution_clock::now();
    int query_count = min(1000, static_cast<int>(trace.size()));
    for (int i = 0; i < query_count; ++i) {
        sampler.query(trace[i]);
    }
    auto query_end = chrono::high_resolution_clock::now();
    
    // 输出结果到控制台
    cout << "=== " << sampler.name() << " Evaluation for file " << file_index << " ===" << endl;
    cout << "Memory usage: " << sampler.memory_usage() << " bytes" << endl;
    cout << "Sample rate: " << sampler.get_sample_rate() << endl;
    cout << "WMRD: " << wmrd << endl;
    cout << "MRD: " << are << endl;
    
    chrono::duration<double, milli> insert_duration = insert_end - start;
    chrono::duration<double, milli> dist_duration = dist_end - insert_end;
    auto query_duration = chrono::duration_cast<chrono::microseconds>(query_end - query_start);
    
    cout << fixed << setprecision(6);
    cout << "Insert time: " << insert_duration.count() << " ms" << endl;
    cout << "Distribution calculation time: " << dist_duration.count() << " ms" << endl;
    cout << "Average query time: " 
         << static_cast<double>(query_duration.count()) / query_count << " μs" << endl;
    
    // 输出特定统计信息
    sampler.print_stats();
    
    cout << endl;
    
    // 输出结果到CSV文件
    if (csv_file.is_open()) {
        csv_file << trace.size() << ",";                 // packets number
        csv_file << sampler.memory_usage() << ",";        // memory
        csv_file << sampler.name() << ",";                // method (array/hash)
        csv_file << fixed << setprecision(6) << wmrd << ","; // WMRD
        csv_file << fixed << setprecision(6) << are << ",";  // ARE (实际上是MRD)
        csv_file << insert_duration.count() << ",";        // Insert time (ms)
        csv_file << dist_duration.count() << endl;         // Decode time (ms)
    }
}

// 程序选项结构
struct ProgramOptions {
    string sampler_type = "array";
    string trace_prefix = "data/";
    int start_file_no = 10;
    int end_file_no = 10;
    size_t memory_limit = 1024 * 1024; // 1MB
    string output_csv = "";            // CSV输出文件路径
};

void print_usage() {
    cout << "Usage: sampler_test [options]\n"
         << "Options:\n"
         << "  -t, --type <type>          Sampler type: 'array' or 'hash' (default: array)\n"
         << "  -d, --data <path>          Path prefix to data files (default: data/)\n"
         << "  -s, --start <num>          Start file number (default: 10)\n"
         << "  -e, --end <num>            End file number (default: 10)\n"
         << "  -m, --memory <bytes>       Memory limit in bytes (default: 1048576)\n"
         << "  -o, --output <file>        Output CSV file path\n"
         << "  -h, --help                 Show this help message\n";
}

ProgramOptions parse_args(int argc, char* argv[]) {
    ProgramOptions options;
    
    static struct option long_options[] = {
        {"type", required_argument, 0, 't'},
        {"data", required_argument, 0, 'd'},
        {"start", required_argument, 0, 's'},
        {"end", required_argument, 0, 'e'},
        {"memory", required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:s:e:m:o:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 't':
                options.sampler_type = optarg;
                break;
            case 'd':
                options.trace_prefix = optarg;
                break;
            case 's':
                options.start_file_no = atoi(optarg);
                break;
            case 'e':
                options.end_file_no = atoi(optarg);
                break;
            case 'm':
                options.memory_limit = atoi(optarg);
                break;
            case 'o':
                options.output_csv = optarg;
                break;
            case 'h':
                print_usage();
                exit(0);
            default:
                print_usage();
                exit(1);
        }
    }

    return options;
}

int main(int argc, char* argv[]) {
    ProgramOptions options = parse_args(argc, argv);
    
    // 打印配置信息
    cout << "Running with configuration:\n"
         << "Sampler type: " << options.sampler_type << "\n"
         << "Data path: " << options.trace_prefix << "\n"
         << "File range: " << options.start_file_no << " to " << options.end_file_no << "\n"
         << "Memory limit: " << options.memory_limit << " bytes\n";
    
    // 打开CSV文件
    if (!options.output_csv.empty()) {
        csv_file.open(options.output_csv);
        if (!csv_file.is_open()) {
            cerr << "Failed to open CSV file: " << options.output_csv << endl;
            return 1;
        }
        cout << "Output CSV: " << options.output_csv << "\n";
        
        // 写入CSV表头
        csv_file << "packets number,memory,method,WMRD,ARE,Insert time,Decode time\n";
    }
    cout << endl;

    // 读取数据
    ReadInTraces(options.trace_prefix.c_str(), options.start_file_no, options.end_file_no);
    
    // 评估每个文件
    for (int i = 0; i < traces.size(); ++i) {
        // 计算总流数
        unordered_map<string, int> real_freq;
        for (const auto& packet : traces[i]) {
            string key(packet.key, 13);
            real_freq[key]++;
        }
        size_t total_flows = real_freq.size();
        
        cout << "File " << (options.start_file_no + i) << ": "
             << traces[i].size() << " packets, "
             << total_flows << " flows\n";
        
        // 创建采样器
        unique_ptr<BaseSampler> sampler;
        if (options.sampler_type == "array") {
            sampler.reset(new ArraySampler(options.memory_limit, total_flows));
        } else if (options.sampler_type == "hash") {
            sampler.reset(new HashMapSampler(options.memory_limit, total_flows));
        } else {
            cerr << "Unknown sampler type: " << options.sampler_type << endl;
            return 1;
        }
        
        // 计算真实分布
        vector<double> real_dist;
        real_distribution(traces[i], real_dist);
        
        // 评估采样器
        evaluate_sampler(*sampler, traces[i], real_dist, options.start_file_no + i);
    }
    
    // 关闭CSV文件
    if (csv_file.is_open()) {
        csv_file.close();
    }
    
    return 0;
}
