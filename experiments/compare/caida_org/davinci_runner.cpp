#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/Sketchs/DaVinci/DaVinci.h"

using namespace std;

struct PacketKey {
    char key[13];
};

struct Options {
    string data_prefix = "data/";
    int start_file_no = 1;
    int end_file_no = 1;
    int memory_size = 16384;
    string output_csv;
    bool united_decode = false;
    bool query_all_flows = false;
    int extra_em_iters = 0;
};

static void usage() {
    cout << "Usage: davinci_runner [options]\n"
         << "  -d, --data <path>      Path prefix to data files\n"
         << "  -s, --start <num>      Start file number; reads <num-1>.dat\n"
         << "  -e, --end <num>        End file number\n"
         << "  -m, --memory <bytes>   Total memory in bytes\n"
         << "  -o, --output <file>    Optional CSV output\n"
         << "      --united-decode    Use DaVinci united_decode path\n"
         << "      --query-all-flows  Also query every true flow after FSD decode\n"
         << "      --extra-em-iters N Run N extra Tower EM iterations for timing\n"
         << "  -h, --help             Show help\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    static option long_opts[] = {
        {"data", required_argument, 0, 'd'},
        {"start", required_argument, 0, 's'},
        {"end", required_argument, 0, 'e'},
        {"memory", required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"united-decode", no_argument, 0, 1000},
        {"query-all-flows", no_argument, 0, 1001},
        {"extra-em-iters", required_argument, 0, 1002},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };
    int c;
    while ((c = getopt_long(argc, argv, "d:s:e:m:o:h", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'd': opt.data_prefix = optarg; break;
            case 's': opt.start_file_no = atoi(optarg); break;
            case 'e': opt.end_file_no = atoi(optarg); break;
            case 'm': opt.memory_size = atoi(optarg); break;
            case 'o': opt.output_csv = optarg; break;
            case 1000: opt.united_decode = true; break;
            case 1001: opt.query_all_flows = true; break;
            case 1002: opt.extra_em_iters = atoi(optarg); break;
            case 'h': usage(); exit(0);
            default: usage(); exit(1);
        }
    }
    return opt;
}

static vector<PacketKey> read_trace(const string& prefix, int file_no) {
    char name[4096];
    snprintf(name, sizeof(name), "%s%d.dat", prefix.c_str(), file_no - 1);
    FILE* fin = fopen(name, "rb");
    if (!fin) {
        cerr << "Failed to open file: " << name << endl;
        exit(1);
    }
    vector<PacketKey> packets;
    PacketKey key;
    while (fread(&key, 1, 13, fin) == 13) {
        packets.push_back(key);
    }
    fclose(fin);
    cout << "Successfully read in " << name << ", " << packets.size() << " packets\n";
    return packets;
}

static vector<double> distribution_from_freq(const unordered_map<string, int>& freq) {
    int max_size = 0;
    for (const auto& kv : freq) {
        max_size = max(max_size, kv.second);
    }
    vector<double> dist(max_size + 1, 0.0);
    for (const auto& kv : freq) {
        dist[kv.second] += 1.0;
    }
    return dist;
}

static vector<double> real_distribution(const vector<PacketKey>& packets) {
    unordered_map<string, int> freq;
    freq.reserve(packets.size() / 2);
    for (const auto& p : packets) {
        freq[string(p.key, 4)] += 1;
    }
    return distribution_from_freq(freq);
}

static double wmrd(const vector<double>& real, const vector<double>& est) {
    size_t n = max(real.size(), est.size());
    double diff = 0.0;
    double denom = 0.0;
    for (size_t i = 1; i < n; ++i) {
        double r = i < real.size() ? real[i] : 0.0;
        double e = i < est.size() ? est[i] : 0.0;
        diff += fabs(r - e);
        denom += (r + e) / 2.0;
    }
    return denom > 0 ? diff / denom : 0.0;
}

static double mrd(const vector<double>& real, const vector<double>& est) {
    size_t n = max(real.size(), est.size());
    double rel = 0.0;
    int count = 0;
    for (size_t i = 1; i < n; ++i) {
        double r = i < real.size() ? real[i] : 0.0;
        double e = i < est.size() ? est[i] : 0.0;
        double denom = (r + e) / 2.0;
        if (denom > 0) {
            rel += fabs(r - e) / denom;
            count += 1;
        }
    }
    return count ? rel / count : 0.0;
}

static void scaled_davinci_memory(int memory, int& heavy_buckets, int& fermat_mem, int& tower_mem) {
    const int heavy_bucket_bytes = sizeof(Bucket);
    const int fermat_array_num = 3;
    const int fermat_entry_bytes = 6;  // id uint32 + signed counter int16, no fingerprint.
    const int fermat_unit = fermat_array_num * fermat_entry_bytes;

    int heavy_budget = max(heavy_bucket_bytes, memory * 24 / 100);
    heavy_buckets = min(2400, max(1, heavy_budget / heavy_bucket_bytes));
    int heavy_actual = heavy_buckets * heavy_bucket_bytes;

    int fermat_budget = max(fermat_unit, memory * 18 / 100);
    int max_fermat_budget = max(fermat_unit, memory - heavy_actual - 2);
    fermat_budget = min(fermat_budget, max_fermat_budget);
    int fermat_entries = max(1, fermat_budget / fermat_unit);
    fermat_mem = fermat_entries * fermat_unit;

    int tower_budget = max(3, memory - heavy_actual - fermat_mem);
    tower_mem = max(1, tower_budget / 3);
}

static int davinci_actual_memory(int heavy_buckets, int fermat_mem, int tower_mem) {
    // TowerSketch(w_d) stores 2-bit/4-bit/8-bit arrays with widths
    // 4*w_d/2*w_d/w_d, i.e., 3*w_d bytes total.
    return heavy_buckets * (int)sizeof(Bucket) + fermat_mem + 3 * tower_mem;
}

int main(int argc, char** argv) {
    Options opt = parse_args(argc, argv);
    ofstream csv;
    if (!opt.output_csv.empty()) {
        csv.open(opt.output_csv);
        csv << "file,memory,mrd,wmrd,insert_ms,decode_ms,fermat_decode_ms,tower_copy_ms,tower_em_ms,postprocess_ms,"
            << "extra_em_ms,query_all_ms,conservative_decode_ms,tower_width,tower_nonzero,tower_max_counter,"
            << "tower_counter_bits,tower_cap_counter,tower_cap_fraction,tower_mid_width,tower_mid_nonzero,"
            << "tower_mid_cap_counter,tower_mid_cap_fraction\n";
    }

    cout << "Running DaVinci with memory " << opt.memory_size << " bytes\n";
    for (int file_no = opt.start_file_no; file_no <= opt.end_file_no; ++file_no) {
        vector<PacketKey> packets = read_trace(opt.data_prefix, file_no);
        vector<double> real = real_distribution(packets);

        int heavy_buckets, fermat_mem, tower_mem;
        scaled_davinci_memory(opt.memory_size, heavy_buckets, fermat_mem, tower_mem);
        int actual_memory = davinci_actual_memory(heavy_buckets, fermat_mem, tower_mem);
        cout << "DaVinci memory breakdown: heavy=" << heavy_buckets * (int)sizeof(Bucket)
             << " bytes, fermat=" << fermat_mem
             << " bytes, tower=" << 3 * tower_mem
             << " bytes, actual_total=" << actual_memory << " bytes\n";
        DaVinci<2400> sketch(opt.memory_size, fermat_mem, heavy_buckets, tower_mem, 3, USE_FING, 37, false, CM);

        auto insert_start = chrono::high_resolution_clock::now();
        for (auto& p : packets) {
            sketch.insert(p.key);
        }
        auto insert_end = chrono::high_resolution_clock::now();
        unordered_map<string, int> true_freq;
        if (opt.query_all_flows) {
            true_freq.reserve(packets.size() / 2);
            for (const auto& p : packets) {
                true_freq[string(p.key, 4)] += 1;
            }
        }

        vector<double> est;
        auto decode_start = chrono::high_resolution_clock::now();
        sketch.decode(opt.united_decode);
        auto fermat_decode_end = chrono::high_resolution_clock::now();
        sketch.get_distribution(est);
        auto decode_end = chrono::high_resolution_clock::now();
        auto extra_em_start = chrono::high_resolution_clock::now();
        for (int i = 0; i < opt.extra_em_iters && sketch.em_tower; ++i) {
            sketch.em_tower->next_epoch();
        }
        auto extra_em_end = chrono::high_resolution_clock::now();
        uint64_t query_checksum = 0;
        auto query_start = chrono::high_resolution_clock::now();
        if (opt.query_all_flows) {
            for (const auto& kv : true_freq) {
                query_checksum += sketch.query(kv.first.data());
            }
        }
        auto query_end = chrono::high_resolution_clock::now();

        chrono::duration<double, milli> insert_ms = insert_end - insert_start;
        chrono::duration<double, milli> decode_ms = decode_end - decode_start;
        chrono::duration<double, milli> fermat_decode_ms = fermat_decode_end - decode_start;
        chrono::duration<double, milli> extra_em_ms = extra_em_end - extra_em_start;
        chrono::duration<double, milli> query_all_ms = query_end - query_start;
        double conservative_decode_ms = decode_ms.count() + extra_em_ms.count() + query_all_ms.count();
        double out_mrd = mrd(real, est);
        double out_wmrd = wmrd(real, est);
        double tower_cap_fraction = sketch.last_tower_width > 0
                                        ? (double)sketch.last_tower_cap_counter / sketch.last_tower_width
                                        : 0.0;
        double tower_mid_cap_fraction = sketch.last_tower_mid_width > 0
                                            ? (double)sketch.last_tower_mid_cap_counter / sketch.last_tower_mid_width
                                            : 0.0;

        cout << "=== DaVinci Evaluation ===\n"
             << "Total packets: " << packets.size() << "\n"
             << "MRD: " << out_mrd << "\n"
             << "WMRD: " << out_wmrd << "\n"
             << "Insertion time: " << insert_ms.count() << " ms\n"
             << "Distribution calculation (decode) time: " << decode_ms.count() << " ms\n"
             << "Fermat decode time: " << fermat_decode_ms.count() << " ms\n"
             << "Tower copy/init time: " << sketch.last_tower_copy_ms << " ms\n"
             << "Tower EM time: " << sketch.last_tower_em_ms << " ms\n"
             << "Postprocess time: " << sketch.last_postprocess_ms << " ms\n"
             << "Tower counters: width=" << sketch.last_tower_width
             << " bits=" << sketch.last_tower_counter_bits
             << " nonzero=" << sketch.last_tower_nonzero
             << " max=" << sketch.last_tower_max_counter
             << " capped=" << sketch.last_tower_cap_counter
             << " cap_fraction=" << tower_cap_fraction << "\n"
             << "Tower 4-bit counters: width=" << sketch.last_tower_mid_width
             << " nonzero=" << sketch.last_tower_mid_nonzero
             << " capped15=" << sketch.last_tower_mid_cap_counter
             << " cap_fraction=" << tower_mid_cap_fraction << "\n"
             << "Extra Tower EM time: " << extra_em_ms.count() << " ms for " << opt.extra_em_iters << " iterations\n"
             << "Query-all true flows time: " << query_all_ms.count() << " ms checksum=" << query_checksum << "\n"
             << "Conservative decode time: " << conservative_decode_ms << " ms\n\n";

        if (csv.is_open()) {
            csv << (file_no - 1) << "," << opt.memory_size << "," << out_mrd << "," << out_wmrd
                << "," << insert_ms.count() << "," << decode_ms.count()
                << "," << fermat_decode_ms.count() << "," << sketch.last_tower_copy_ms
                << "," << sketch.last_tower_em_ms << "," << sketch.last_postprocess_ms
                << "," << extra_em_ms.count() << "," << query_all_ms.count()
                << "," << conservative_decode_ms
                << "," << sketch.last_tower_width << "," << sketch.last_tower_nonzero
                << "," << sketch.last_tower_max_counter
                << "," << sketch.last_tower_counter_bits
                << "," << sketch.last_tower_cap_counter
                << "," << tower_cap_fraction
                << "," << sketch.last_tower_mid_width
                << "," << sketch.last_tower_mid_nonzero
                << "," << sketch.last_tower_mid_cap_counter
                << "," << tower_mid_cap_fraction << "\n";
        }
    }
    return 0;
}
