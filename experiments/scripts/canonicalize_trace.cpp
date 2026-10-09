#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

uint32_t fnv1a32(const unsigned char* data, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint32_t>(data[i]);
        h *= 16777619u;
    }
    return h;
}

void usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s <input> <output> <record_size> <key_offset> <key_len>\n",
                 prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        usage(argv[0]);
        return 1;
    }

    const std::string input = argv[1];
    const std::string output = argv[2];
    const int record_size = std::atoi(argv[3]);
    const int key_offset = std::atoi(argv[4]);
    const int key_len = std::atoi(argv[5]);

    if (record_size <= 0 || key_offset < 0 || key_len <= 0 ||
        key_offset + key_len > record_size) {
        std::fprintf(stderr, "Invalid record/key layout\n");
        return 1;
    }

    FILE* fin = std::fopen(input.c_str(), "rb");
    if (!fin) {
        std::perror(input.c_str());
        return 1;
    }
    FILE* fout = std::fopen(output.c_str(), "wb");
    if (!fout) {
        std::perror(output.c_str());
        std::fclose(fin);
        return 1;
    }

    std::vector<unsigned char> record(static_cast<size_t>(record_size));
    unsigned char out[13];
    std::memset(out, 0, sizeof(out));
    uint64_t packets = 0;

    while (std::fread(record.data(), 1, record.size(), fin) == record.size()) {
        const uint32_t h = fnv1a32(record.data() + key_offset, static_cast<size_t>(key_len));
        out[0] = static_cast<unsigned char>(h & 0xffu);
        out[1] = static_cast<unsigned char>((h >> 8) & 0xffu);
        out[2] = static_cast<unsigned char>((h >> 16) & 0xffu);
        out[3] = static_cast<unsigned char>((h >> 24) & 0xffu);
        if (std::fwrite(out, 1, sizeof(out), fout) != sizeof(out)) {
            std::perror(output.c_str());
            std::fclose(fin);
            std::fclose(fout);
            return 1;
        }
        ++packets;
    }

    std::fclose(fin);
    std::fclose(fout);
    std::fprintf(stderr, "canonicalized %llu packets: %s -> %s\n",
                 static_cast<unsigned long long>(packets), input.c_str(), output.c_str());
    return 0;
}
