#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>
#include <libgen.h>   // 添加缺失的头文件
#include <limits.h>   // 添加PATH_MAX定义

#define DEFAULT_COUNTERS  4096    // 2^14哈希表大小
#define PACKET_SIZE       16       // 每个数据包16字节
#define BUFFER_SIZE       4096     // 4KB读取缓冲区
#define ALIGNMENT         16       // 内存对齐要求
#define PATH_MAX 4096 // 添加PATH_MAX定义

// 内存对齐的缓冲区定义
typedef struct {
    uint8_t data[BUFFER_SIZE + ALIGNMENT - 1];
} AlignedBuffer;


// MurmurHash3 32位优化版（内存对齐处理）
uint32_t MurmurHash3_x86_32(const void *key, int len, uint32_t seed) {
    const uint8_t *data = (const uint8_t *)key;
    const int nblocks = len / 4;

    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // 处理4字节块（强制内存对齐）
    const uint32_t *blocks = (const uint32_t *)(data + nblocks*4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;

        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }

    // 处理剩余字节
    const uint8_t *tail = data + nblocks*4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = (k1 << 15) | (k1 >> 17);
                k1 *= c2;
                h1 ^= k1;
    }

    // 最终处理
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}

typedef struct {
    uint32_t* counters;
    size_t total_packets;
    size_t lost_bytes;
} ProcessingResult;

ProcessingResult process_dat(const char* path, uint32_t seed) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        perror("❌ 文件打开失败");
        return (ProcessingResult){NULL, 0, 0};
    }

    // 内存对齐分配
    uint32_t* counters = aligned_alloc(ALIGNMENT, DEFAULT_COUNTERS * sizeof(uint32_t));
    memset(counters, 0, DEFAULT_COUNTERS * sizeof(uint32_t));
    
    AlignedBuffer buffer;
    size_t packet_offset = 0;
    size_t total_processed = 0;

    while (1) {
        uint8_t* aligned_ptr = (uint8_t*)(((uintptr_t)buffer.data + ALIGNMENT - 1) & ~(ALIGNMENT - 1));
        size_t read_size = fread(aligned_ptr + packet_offset, 1, 
                                BUFFER_SIZE - packet_offset, fp);
        
        if (read_size == 0) break;

        const size_t total_bytes = read_size + packet_offset;
        const size_t packets = total_bytes / PACKET_SIZE;
        
        // 处理每个数据包
        for (size_t i = 0; i < packets; ++i) {
            const uint8_t* pkt = aligned_ptr + i * PACKET_SIZE;
            const uint32_t hash = MurmurHash3_x86_32(pkt + 8, 8, seed);
            counters[hash % DEFAULT_COUNTERS]++;
            total_processed++;
        }

        // 保存未完成数据包
        packet_offset = total_bytes % PACKET_SIZE;
        if (packet_offset > 0) {
            memmove(aligned_ptr, aligned_ptr + packets * PACKET_SIZE, packet_offset);
        }
    }

    fclose(fp);
    
    // 验证数据完整性
    struct stat st;
    stat(path, &st);
    const size_t expected_packets = st.st_size / PACKET_SIZE;
    const size_t lost = expected_packets - total_processed;
    
    printf("✅ 处理完成 | 总数据包: %zu | 成功处理: %zu (%.2f%%) | 丢失: %zu\n",
          expected_packets, total_processed, 
          (total_processed * 100.0) / expected_packets, lost);

    return (ProcessingResult){counters, total_processed, packet_offset};
}

int save_binary(const char* path, const uint32_t* counters) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return 0;
    
    const size_t written = fwrite(counters, sizeof(uint32_t), DEFAULT_COUNTERS, fp);
    fclose(fp);
    return written == DEFAULT_COUNTERS;
}

void validate_arguments(const char* input, const char* output_dir) {
    struct stat path_stat;
    if (stat(input, &path_stat) != 0) {
        fprintf(stderr, "❌ 输入文件不存在: %s\n", input);
        exit(EXIT_FAILURE);
    }
    
    if (access(output_dir, W_OK) != 0 && mkdir(output_dir, 0755) != 0) {
        fprintf(stderr, "❌ 无法创建输出目录: %s\n", output_dir);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    // 参数解析优化
    static struct option long_options[] = {
        {"input",       required_argument, 0, 'i'},
        {"output-dir",  required_argument, 0, 'd'},
        {"start-seed",  required_argument, 0, 'b'},
        {"end-seed",    required_argument, 0, 'e'},
        {"single",      no_argument,       0, 's'},
        {"output",      required_argument, 0, 'o'},
        {"seed",        required_argument, 0, 'S'},
        {0, 0, 0, 0}
    };

    // 参数默认值
    struct {
        const char* input;
        const char* output_dir;
        int start_seed;
        int end_seed;
        int single_mode;
        const char* output_file;
        uint32_t seed;
    } args = {NULL, NULL, 0, 2000, 0, NULL, 0};

    int opt;
    while ((opt = getopt_long(argc, argv, "i:d:b:e:so:S:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'i': args.input = optarg; break;
            case 'd': args.output_dir = optarg; break;
            case 'b': args.start_seed = atoi(optarg); break;
            case 'e': args.end_seed = atoi(optarg); break;
            case 's': args.single_mode = 1; break;
            case 'o': args.output_file = optarg; break;
            case 'S': args.seed = atoi(optarg); break;
            default: exit(EXIT_FAILURE);
        }
    }

    // 参数验证强化（修复dirname使用）
    if (args.single_mode) {
        if (!args.input || !args.output_file || args.seed == 0) {
            fprintf(stderr, "❌ 单种子模式需要 --input, --output 和 --seed\n");
            exit(EXIT_FAILURE);
        }
        
        // 修复dirname使用方式
        char* output_path_copy = strdup(args.output_file);
        char* dir = dirname(output_path_copy);
        validate_arguments(args.input, dir);
        free(output_path_copy);
        
        ProcessingResult res = process_dat(args.input, args.seed);
        if (res.counters && save_binary(args.output_file, res.counters)) {
            printf("💾 成功保存至: %s\n", args.output_file);
        }
        free(res.counters);
    } else {
        if (!args.input || !args.output_dir) {
            fprintf(stderr, "❌ 批处理模式需要 --input 和 --output-dir\n");
            exit(EXIT_FAILURE);
        }
        
        validate_arguments(args.input, args.output_dir);
        
        // 批量处理种子（修复PATH_MAX使用）
        for (int seed = args.start_seed; seed < args.end_seed; ++seed) {
            char path[PATH_MAX];  // 使用limits.h定义的PATH_MAX
            snprintf(path, sizeof(path), "%s/%d.bin", args.output_dir, seed);
            
            ProcessingResult res = process_dat(args.input, seed);
            if (res.counters) {
                if (save_binary(path, res.counters)) {
                    printf("🔧 种子 %d 处理完成 | 目录: %s\n", seed, path);
                }
                free(res.counters);
            }
        }
    }

    return EXIT_SUCCESS;
}