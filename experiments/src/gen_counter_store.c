#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static uint32_t murmur3_x86_32(const void *key, int len, uint32_t seed) {
    const uint8_t *data = (const uint8_t *)key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const uint32_t *blocks = (const uint32_t *)(data + nblocks * 4);

    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t *tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3:
        k1 ^= tail[2] << 16;
    case 2:
        k1 ^= tail[1] << 8;
    case 1:
        k1 ^= tail[0];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
    }

    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    return h1;
}

typedef struct {
    const char *input;
    const char *output;
    int start_seed;
    int end_seed;
    int counters;
    int packet_size;
    int key_offset;
    int key_size;
} Args;

static void usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --input IN --output OUT --start-seed N --end-seed N "
            "--counters N --packet-size N --key-offset N --key-size N\n",
            prog);
}

static Args parse_args(int argc, char **argv) {
    static struct option opts[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"start-seed", required_argument, 0, 'b'},
        {"end-seed", required_argument, 0, 'e'},
        {"counters", required_argument, 0, 'c'},
        {"packet-size", required_argument, 0, 'p'},
        {"key-offset", required_argument, 0, 'k'},
        {"key-size", required_argument, 0, 'l'},
        {0, 0, 0, 0},
    };
    Args args = {0};
    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:b:e:c:p:k:l:", opts, NULL)) != -1) {
        switch (opt) {
        case 'i': args.input = optarg; break;
        case 'o': args.output = optarg; break;
        case 'b': args.start_seed = atoi(optarg); break;
        case 'e': args.end_seed = atoi(optarg); break;
        case 'c': args.counters = atoi(optarg); break;
        case 'p': args.packet_size = atoi(optarg); break;
        case 'k': args.key_offset = atoi(optarg); break;
        case 'l': args.key_size = atoi(optarg); break;
        default:
            usage(argv[0]);
            exit(2);
        }
    }
    if (!args.input || !args.output || args.end_seed <= args.start_seed || args.counters <= 0 ||
        args.packet_size <= 0 || args.key_offset < 0 || args.key_size <= 0 ||
        args.key_offset + args.key_size > args.packet_size) {
        usage(argv[0]);
        exit(2);
    }
    return args;
}

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    const int seed_count = args.end_seed - args.start_seed;
    uint32_t *counters = calloc((size_t)seed_count * (size_t)args.counters, sizeof(uint32_t));
    if (!counters) {
        perror("calloc counters");
        return 1;
    }

    struct stat st;
    if (stat(args.input, &st) != 0) {
        perror(args.input);
        free(counters);
        return 1;
    }
    if (st.st_size <= 0) {
        fprintf(stderr, "empty input: %s\n", args.input);
        free(counters);
        return 1;
    }
    size_t file_size = (size_t)st.st_size;
    if ((off_t)file_size != st.st_size) {
        fprintf(stderr, "input too large for this platform: %s\n", args.input);
        free(counters);
        return 1;
    }

    FILE *in = fopen(args.input, "rb");
    if (!in) {
        perror(args.input);
        free(counters);
        return 1;
    }

    uint8_t *buffer = malloc(file_size);
    if (!buffer) {
        perror("malloc buffer");
        fclose(in);
        free(counters);
        return 1;
    }

    size_t read_n = fread(buffer, 1, file_size, in);
    fclose(in);
    if (read_n != file_size) {
        fprintf(stderr, "short read: %zu/%zu\n", read_n, file_size);
        free(buffer);
        free(counters);
        return 1;
    }

    size_t packets = file_size / (size_t)args.packet_size;
#pragma omp parallel for schedule(static)
    for (int s = 0; s < seed_count; ++s) {
        uint32_t *row = counters + (size_t)s * (size_t)args.counters;
        uint32_t seed = (uint32_t)(args.start_seed + s);
        for (size_t i = 0; i < packets; ++i) {
            const uint8_t *pkt = buffer + i * (size_t)args.packet_size + args.key_offset;
            uint32_t h = murmur3_x86_32(pkt, args.key_size, seed);
            row[h % (uint32_t)args.counters]++;
        }
    }
    free(buffer);

    FILE *out = fopen(args.output, "wb");
    if (!out) {
        perror(args.output);
        free(counters);
        return 1;
    }
    size_t want = (size_t)seed_count * (size_t)args.counters;
    size_t wrote = fwrite(counters, sizeof(uint32_t), want, out);
    fclose(out);
    free(counters);
    if (wrote != want) {
        fprintf(stderr, "short write: %zu/%zu\n", wrote, want);
        return 1;
    }
    fprintf(stderr, "processed %zu packets -> %s (%d seeds x %d counters)\n",
            packets, args.output, seed_count, args.counters);
    return 0;
}
