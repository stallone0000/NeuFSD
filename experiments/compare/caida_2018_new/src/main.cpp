#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <getopt.h>
#include <string>
#include <memory>
#include <iostream>
#include <chrono> // 添加计时库
#include <fstream> // 添加文件操作库
#include "Sketchs/elastic/ElasticSketch.h"
#include "Sketchs/MRAC/MRAC.h"
#include <cstring>
// #include "Sketchs/DaVinci/DaVinci.h"
// #include "Sketchs/FCMelastic/FCMelastic.h"
#include "Sketchs/Eviction/Eviction.h"
#include "common/BOBHash32.h"
#include "common/wmrd_calculator.h"

using namespace std;

#define MAX_PRIME32 1229

// 移除了所有硬编码的 #define

struct FIVE_TUPLE { char key[13]; };
typedef vector<FIVE_TUPLE> TRACE;

// === 新增：用于返回计时结果的结构体 ===
struct TimingResult {
    double insert_time_ms;
    double decode_time_ms;
};

struct ProgramOptions {
    string sketch_type = "elastic";    // elastic, mrac, or method
    string trace_prefix = "data/";
    int start_file_no = 1;
    int end_file_no = 1;
    double sample_rate = 0.1;
    int memory_size = 16384; // === 新增：内存大小作为运行时参数 ===
};

void print_usage() {
    cout << "Usage: sketch_test [options]\n"
         << "Options:\n"
         << "  -t, --type <type>          Sketch type: 'elastic', 'method' or 'mrac' (default: elastic)\n"
         << "  -d, --data <path>          Path prefix to data files (default: ../data/)\n"
         << "  -s, --start <num>          Start file number (default: 1)\n"
         << "  -e, --end <num>            End file number (default: 1)\n"
         << "  -r, --sample-rate <rate>   Sample rate for MRAC/Method (default: 0.1)\n"
         << "  -m, --memory <bytes>       Total memory in bytes (default: 16384)\n" // === 新增 ===
         << "  -h, --help                 Show this help message\n";
}

ProgramOptions parse_args(int argc, char* argv[]) {
    ProgramOptions options;
    
    static struct option long_options[] = {
        {"type", required_argument, 0, 't'},
        {"data", required_argument, 0, 'd'},
        {"start", required_argument, 0, 's'},
        {"end", required_argument, 0, 'e'},
        {"sample-rate", required_argument, 0, 'r'},
        {"memory", required_argument, 0, 'm'}, // === 新增 ===
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "t:d:s:e:r:m:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 't':
                options.sketch_type = optarg;
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
            case 'r':
                options.sample_rate = atof(optarg);
                break;
            case 'm': // === 新增 ===
                options.memory_size = atoi(optarg);
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

vector<TRACE> traces;

void ReadInTraces(const char *trace_prefix, int start_file_no, int end_file_no) {
    traces.resize(end_file_no - start_file_no + 1);
    
    for(int datafileCnt = start_file_no; datafileCnt <= end_file_no; ++datafileCnt) {
        char datafileName[100];
        sprintf(datafileName, "%s%d.dat", trace_prefix, datafileCnt - 1);
        FILE *fin = fopen(datafileName, "rb");

        if (!fin) {
            printf("Failed to open file: %s\n", datafileName);
            exit(1);
        }

        uint8_t buffer[21];
        traces[datafileCnt - start_file_no].clear();

        while(fread(buffer, 1, 21, fin) == 21) {
            FIVE_TUPLE tmp_five_tuple;
            memcpy(&tmp_five_tuple, buffer, 13);
            
            traces[datafileCnt - start_file_no].push_back(tmp_five_tuple);
        }
        fclose(fin);

        printf("Successfully read in %s, %ld packets\n", datafileName, traces[datafileCnt - start_file_no].size());
    }
    printf("\n");
}

void GetRealFreq(int datafileCnt, unordered_map<string, int>& Real_Freq) {
    int packet_cnt = (int)traces[datafileCnt].size();
    for(int i = 0; i < packet_cnt; ++i) {
        string str((const char*)(traces[datafileCnt][i].key), 4);
        Real_Freq[str]++;
    }
}

class BaseSketch {
public:
    virtual ~BaseSketch() {}
    virtual void insert(uint8_t* key) = 0;
    virtual int query(uint8_t* key) = 0;
    virtual void get_distribution(vector<double>& dist) = 0;
    virtual const char* get_name() const = 0;
    virtual bool is_sampled(uint8_t* key) { return true; }
    virtual void output() {return ;}
};

template<int BUCKET_NUM, int TOT_MEM_IN_BYTES>
class ElasticSketchImpl : public BaseSketch {
private:
    ElasticSketch<BUCKET_NUM, TOT_MEM_IN_BYTES>* sketch;
public:
    ElasticSketchImpl() { sketch = new ElasticSketch<BUCKET_NUM, TOT_MEM_IN_BYTES>(); }
    ~ElasticSketchImpl() { delete sketch; }
    void insert(uint8_t* key) override { sketch->insert(key); }
    int query(uint8_t* key) override { return sketch->query(key); }
    void get_distribution(vector<double>& dist) override { sketch->get_distribution(dist); }
    const char* get_name() const override { return "ElasticSketch"; }
};

template<int TOT_MEM_IN_BYTES>
class MRACImpl : public BaseSketch {
private:
    MRAC<4, TOT_MEM_IN_BYTES>* sketch;
    BOBHash32* bob_hash;
    double sample_rate;
public:
    MRACImpl(double p) : sample_rate(p) {
        sketch = new MRAC<4, TOT_MEM_IN_BYTES>();
        bob_hash = new BOBHash32(rand() % MAX_PRIME32);
    }
    ~MRACImpl() { delete sketch; delete bob_hash; }
    void insert(uint8_t* key) override { if (is_sampled(key)) { sketch->insert(key); } }
    int query(uint8_t* key) override { return sketch->query(key) / sample_rate; }
    void get_distribution(vector<double>& dist) override {
        sketch->get_distribution(dist);
        for(size_t i = 0; i < dist.size(); i++) { dist[i] = dist[i] / sample_rate; }
    }
    const char* get_name() const override { return "MRAC"; }
    bool is_sampled(uint8_t* key) override {
        uint32_t hash_val = bob_hash->run((const char*)key, 4);
        return (hash_val % 100) < sample_rate * 100;
    }
};

template<int TOT_MEM_IN_BYTES>
class MethodImpl : public BaseSketch {
private:
    MRAC<4, TOT_MEM_IN_BYTES*3/4>* sketch;
    Eviction<4, TOT_MEM_IN_BYTES/4>* eviction;
    BOBHash32* bob_hash;
    double sample_rate;
public:
    MethodImpl(double p) : sample_rate(p) {
        sketch = new MRAC<4, TOT_MEM_IN_BYTES*3/4>();
        eviction = new Eviction<4, TOT_MEM_IN_BYTES/4>();
        bob_hash = new BOBHash32(rand() % MAX_PRIME32);
    }
    ~MethodImpl() { delete sketch; delete eviction; delete bob_hash; }
    void insert(uint8_t* key) override {
        if (is_sampled(key)) { sketch->insert(key); eviction->insert(key); }
    }
    int query(uint8_t* key) override { return sketch->query(key) / sample_rate; }
    void get_distribution(vector<double>& dist) override {
        sketch->get_distribution(dist);
        eviction->get_distribution(dist);
        for(size_t i = 0; i < dist.size(); i++) { dist[i] = dist[i] / sample_rate; }
    }
    const char* get_name() const override { return "OUR_METHOD"; }
    bool is_sampled(uint8_t* key) override {
        uint32_t hash_val = bob_hash->run((const char*)key, 4);
        return (hash_val % 100) < sample_rate * 100;
    }
    void output() override { eviction->output(); }
};

unique_ptr<BaseSketch> create_sketch(const ProgramOptions& options) {
    if (options.sketch_type == "elastic") {
        // 根据内存大小计算桶数量，并实例化对应的模板
        int heavy_mem = options.memory_size / 32;
        int bucket_num = heavy_mem / 64;
        if (options.memory_size == 16384) return unique_ptr<BaseSketch>(new ElasticSketchImpl<8, 16384>());
        else if (options.memory_size == 32768) return unique_ptr<BaseSketch>(new ElasticSketchImpl<16, 32768>());
        else if (options.memory_size == 65536) return unique_ptr<BaseSketch>(new ElasticSketchImpl<32, 65536>());
        else if (options.memory_size == 131072) return unique_ptr<BaseSketch>(new ElasticSketchImpl<64, 131072>());
        else if (options.memory_size == 262144) return unique_ptr<BaseSketch>(new ElasticSketchImpl<128, 262144>());
        else { cerr << "Unsupported memory size for ElasticSketch: " << options.memory_size << endl; return nullptr; }
    } else if (options.sketch_type == "mrac") {
        if (options.memory_size == 16384) return unique_ptr<BaseSketch>(new MRACImpl<16384>(options.sample_rate));
        else if (options.memory_size == 32768) return unique_ptr<BaseSketch>(new MRACImpl<32768>(options.sample_rate));
        else if (options.memory_size == 65536) return unique_ptr<BaseSketch>(new MRACImpl<65536>(options.sample_rate));
        else if (options.memory_size == 131072) return unique_ptr<BaseSketch>(new MRACImpl<131072>(options.sample_rate));
        else if (options.memory_size == 262144) return unique_ptr<BaseSketch>(new MRACImpl<262144>(options.sample_rate));
        else { cerr << "Unsupported memory size for MRAC: " << options.memory_size << endl; return nullptr; }
    } else if (options.sketch_type == "method") {
        if (options.memory_size == 16384) return unique_ptr<BaseSketch>(new MethodImpl<16384>(options.sample_rate));
        else if (options.memory_size == 32768) return unique_ptr<BaseSketch>(new MethodImpl<32768>(options.sample_rate));
        else if (options.memory_size == 65536) return unique_ptr<BaseSketch>(new MethodImpl<65536>(options.sample_rate));
        else if (options.memory_size == 131072) return unique_ptr<BaseSketch>(new MethodImpl<131072>(options.sample_rate));
        else if (options.memory_size == 262144) return unique_ptr<BaseSketch>(new MethodImpl<262144>(options.sample_rate));
        else { cerr << "Unsupported memory size for Method: " << options.memory_size << endl; return nullptr; }
    }
    cerr << "Unknown sketch type: " << options.sketch_type << endl;
    return nullptr;
}



TimingResult EvaluateSketch(BaseSketch* sketch, int datafileCnt) {
    unordered_map<string, int> Real_Freq;
    GetRealFreq(datafileCnt, Real_Freq);
    
    int packet_cnt = (int)traces[datafileCnt].size();

    auto insert_start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < packet_cnt; ++i) {
        sketch->insert((uint8_t*)(traces[datafileCnt][i].key));
    }
    auto insert_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> insert_duration_ms = insert_end - insert_start;
    
    vector<double> real_dist = convert_to_distribution(Real_Freq);
    vector<double> est_dist;
    
    auto decode_start = std::chrono::high_resolution_clock::now();
    sketch->get_distribution(est_dist);
    auto decode_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decode_duration_ms = decode_end - decode_start;

    double wmrd = calculate_wmrd(real_dist, est_dist);
    double mrd = calculate_mrd(real_dist, est_dist);
    printf("=== %s Evaluation ===\n", sketch->get_name());
    printf("Total packets: %d, Total flows: %d\n", packet_cnt, (int)Real_Freq.size());
    printf("MRD: %.6f\n", mrd);
    printf("WMRD: %.6f\n", wmrd);
    printf("Insertion time: %.6f ms\n", insert_duration_ms.count());
    printf("Distribution calculation (decode) time: %.6f ms\n", decode_duration_ms.count());
    printf("\n");

    return {insert_duration_ms.count(), decode_duration_ms.count()};
}

int main(int argc, char* argv[]) {
    ProgramOptions options = parse_args(argc, argv);
    
    cout << "Running with configuration:\n"
         << "Sketch type: " << options.sketch_type << "\n"
         << "Memory size: " << options.memory_size << " bytes\n"
         << "Data path: " << options.trace_prefix << "\n"
         << "File range: " << options.start_file_no << " to " << options.end_file_no << "\n";
    if (options.sketch_type == "mrac" || options.sketch_type == "method") {
        cout << "Sample rate: " << options.sample_rate << "\n";
    }
    cout << "\n";

    string insert_csv_filename = options.sketch_type + "_" + to_string(options.memory_size) + "_insert.csv";
    string decode_csv_filename = options.sketch_type + "_" + to_string(options.memory_size) + "_decode.csv";
    
    ofstream insert_csv_file(insert_csv_filename, ios::app);
    ofstream decode_csv_file(decode_csv_filename, ios::app);
    if (!insert_csv_file.is_open() || !decode_csv_file.is_open()) {
        cerr << "Error: Could not open CSV files for writing." << endl;
        return 1;
    }
    cout << "Appending results to " << insert_csv_filename << " and " << decode_csv_filename << endl;

    ReadInTraces(options.trace_prefix.c_str(), options.start_file_no, options.end_file_no);
    
    // === 使用工厂函数创建sketch ===
    unique_ptr<BaseSketch> sketch = create_sketch(options);
    if (!sketch) {
        return 1; // Error message already printed by factory
    }

    for(int datafileCnt = 0; datafileCnt < options.end_file_no - options.start_file_no + 1; ++datafileCnt) {
        TimingResult result = EvaluateSketch(sketch.get(), datafileCnt);
        insert_csv_file << result.insert_time_ms << "\n";
        decode_csv_file << result.decode_time_ms << "\n";
    }
    
    insert_csv_file.close();
    decode_csv_file.close();
    
    return 0;
}
