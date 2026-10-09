#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include "el.h"
#include "BOBHash.h"

#define PACKET_SIZE   16    // 改为16字节数据包
#define IP_PAIR_SIZE  8     // srcIP和dstIP共8字节
#define BUFFER_SIZE   4096
#define ALIGNMENT     16

// 内存对齐的缓冲区定义
struct AlignedBuffer {
    uint8_t data[BUFFER_SIZE + ALIGNMENT - 1];
};

// Heavy Part记录器
class HeavyRecorder {
public:
    struct Record {
        uint32_t bucket;
        uint32_t slot;
        uint32_t fingerprint;
        uint32_t count;
        uint32_t flag;
    };
    
    HeavyRecorder() = default;
    
    void add_record(uint32_t b, uint32_t s, uint32_t fp, uint32_t c, uint32_t f) {
        records_.push_back({b, s, fp, c, f});
    }
    
    bool save_csv(const std::string& path) {
        if (records_.empty()) {
            std::cerr << "❌ 无Heavy Part数据可保存" << std::endl;
            return false;
        }
        
        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "❌ 无法创建输出文件: " << path << " - " << strerror(errno) << std::endl;
            return false;
        }
        
        // 写入CSV头部
        out << "bucket,slot,fingerprint,count,flag\n";
        
        // 写入记录
        for (const auto& rec : records_) {
            out << rec.bucket << ","
                << rec.slot << ","
                << rec.fingerprint << ","
                << rec.count << ","
                << rec.flag << "\n";
        }
        
        out.close();
        return true;
    }
    
    size_t count() const { return records_.size(); }
    
    void clear() { records_.clear(); }
    
private:
    std::vector<Record> records_;
};

// 处理数据文件
void process_dat(const char* input_path, 
                std::unique_ptr<ElasticSketch>& es, 
                HeavyRecorder& hr) 
{
    FILE* fp = fopen(input_path, "rb");
    if (!fp) {
        perror("❌ 文件打开失败");
        return;
    }
    
    AlignedBuffer buffer;
    size_t packet_offset = 0;
    size_t total_packets = 0;
    
    while (true) {
        uint8_t* aligned_ptr = (uint8_t*)(((uintptr_t)buffer.data + ALIGNMENT - 1) & ~(ALIGNMENT - 1));
        size_t read_size = fread(aligned_ptr + packet_offset, 1, 
                                BUFFER_SIZE - packet_offset, fp);
        
        if (read_size == 0) break;
        
        const size_t total_bytes = read_size + packet_offset;
        const size_t packets = total_bytes / PACKET_SIZE;
        
        // 处理每个数据包 - 使用最后8字节(srcIP + dstIP)作为密钥
        for (size_t i = 0; i < packets; ++i) {
            // 获取数据包起始位置
            const uint8_t* pkt = aligned_ptr + i * PACKET_SIZE;
            // 移动到IP对位置（跳过前8字节时间戳）
            const uint8_t* ip_pair = pkt + 8;
            es->Insert((const char*)ip_pair, IP_PAIR_SIZE);
            total_packets++;
        }
        
        // 保存未完成数据包
        packet_offset = total_bytes % PACKET_SIZE;
        if (packet_offset > 0) {
            memmove(aligned_ptr, aligned_ptr + packets * PACKET_SIZE, packet_offset);
        }
    }
    
    fclose(fp);
    
    // 记录Heavy Part
    for (int b = 0; b < es->get_M1(); ++b) {
        for (int s = 0; s < es->get_buckets(); ++s) {
            auto& slot = es->get_heavy_part()[b][s];
            if (slot.pvote > 0) {
                hr.add_record(b, s, slot.FP, slot.pvote, slot.Flag);
            }
        }
    }
    
    // 获取文件信息
    struct stat st;
    if (stat(input_path, &st) != 0) {
        perror("❌ 文件状态获取失败");
        return;
    }
    
    const size_t expected_packets = st.st_size / PACKET_SIZE;
    
    std::cout << "✅ 处理完成" << std::endl;
    std::cout << "  输入文件: " << input_path << std::endl;
    std::cout << "  总数据包: " << expected_packets << std::endl;
    std::cout << "  处理数据包: " << total_packets << std::endl;
    std::cout << "  提取Heavy流: " << hr.count() << std::endl;
}

// 创建目录（如果不存在）
bool ensure_directory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "❌ 无法创建目录: " << path << " - " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    // 参数解析
    static struct option long_options[] = {
        {"input",       required_argument, 0, 'i'},
        {"output-dir",  required_argument, 0, 'd'},
        {"start-seed",  required_argument, 0, 'b'},
        {"end-seed",    required_argument, 0, 'e'},
        {"m1",          required_argument, 0, '1'}, // ES Heavy桶大小
        {"m2",          required_argument, 0, '2'}, // ES Light桶大小
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    // 参数默认值
    struct {
        const char* input = nullptr;
        const char* output_dir = nullptr;
        int start_seed = 0;
        int end_seed = 0;
        int es_m1 = 100000;   // 默认ElasticSketch Heavy桶大小
        int es_m2 = 200000;   // 默认ElasticSketch Light桶大小
    } args;

    int opt;
    while ((opt = getopt_long(argc, argv, "i:d:b:e:1:2:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'i': args.input = optarg; break;
            case 'd': args.output_dir = optarg; break;
            case 'b': args.start_seed = atoi(optarg); break;
            case 'e': args.end_seed = atoi(optarg); break;
            case '1': args.es_m1 = atoi(optarg); break;
            case '2': args.es_m2 = atoi(optarg); break;
            case 'h':
                std::cout << "用法: " << argv[0] << " -i <输入文件> -d <输出目录> [选项]\n";
                std::cout << "选项:\n";
                std::cout << "  -i, --input <路径>      输入.dat文件路径\n";
                std::cout << "  -d, --output-dir <路径> 输出目录\n";
                std::cout << "  -b, --start-seed <数字>  起始种子 (默认: 0)\n";
                std::cout << "  -e, --end-seed <数字>    结束种子 (默认: 0)\n";
                std::cout << "  --m1 <数字>             Heavy桶大小 (默认: 100000)\n";
                std::cout << "  --m2 <数字>             Light桶大小 (默认: 200000)\n";
                std::cout << "  -h, --help              显示帮助信息\n";
                return 0;
            default: 
                std::cerr << "❌ 未知选项: " << opt << std::endl;
                return 1;
        }
    }

    // 参数验证
    if (!args.input) {
        std::cerr << "❌ 必须指定输入文件 (-i)" << std::endl;
        return 1;
    }
    
    if (!args.output_dir) {
        std::cerr << "❌ 必须指定输出目录 (-d)" << std::endl;
        return 1;
    }
    
    // 检查输入文件是否存在
    struct stat st;
    if (stat(args.input, &st) != 0) {
        std::cerr << "❌ 输入文件不存在: " << args.input << " - " << strerror(errno) << std::endl;
        return 1;
    }
    
    // 创建输出目录
    if (!ensure_directory(args.output_dir)) {
        return 1;
    }
    
    // 确保种子范围有效
    if (args.start_seed > args.end_seed) {
        std::swap(args.start_seed, args.end_seed);
    }
    
    std::cout << "⚙️ 开始处理\n";
    std::cout << "  输入文件: " << args.input << std::endl;
    std::cout << "  输出目录: " << args.output_dir << std::endl;
    std::cout << "  种子范围: " << args.start_seed << " 到 " << args.end_seed << std::endl;
    std::cout << "  M1 (Heavy桶): " << args.es_m1 << std::endl;
    std::cout << "  M2 (Light桶): " << args.es_m2 << std::endl;
    
    // 处理所有种子
    for (int seed = args.start_seed; seed <= args.end_seed; ++seed) {
        // 构建输出文件路径
        char csv_path[PATH_MAX];
        snprintf(csv_path, sizeof(csv_path), "%s/heavy_%d.csv", args.output_dir, seed);
        
        std::cout << "\n🔧 处理种子: " << seed << std::endl;
        std::cout << "  输出文件: " << csv_path << std::endl;
        
        // 初始化ElasticSketch和记录器
        auto es = std::make_unique<ElasticSketch>(args.es_m1, args.es_m2);
        HeavyRecorder hr;
        
        // 处理数据
        process_dat(args.input, es, hr);
        
        // 保存结果
        if (hr.save_csv(csv_path)) {
            std::cout << "💾 Heavy Part保存成功" << std::endl;
        } else {
            std::cerr << "❌ Heavy Part保存失败" << std::endl;
        }
    }
    
    std::cout << "\n✅ 所有种子处理完成" << std::endl;
    return 0;
}