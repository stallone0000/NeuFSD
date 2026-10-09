#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <time.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <sched.h>
#include <pthread.h>

#define BATCH_SIZE 64
#define ETH_HEADER_LEN 14
#define IP_HEADER_LEN 20
#define UDP_HEADER_LEN 8
#define MAX_THREADS 16

// Structs
struct raw_packet {
    struct ethhdr eth;
    struct iphdr ip;
    struct udphdr udp;
    uint8_t payload[1500];
};

struct dataset_entry {
    uint32_t src_ip; uint32_t dst_ip;
    uint16_t src_port; uint16_t dst_port;
    uint8_t proto;
};

// Thread Arguments
struct thread_data {
    int thread_id;
    int cpu_core;
    int socket_fd;
    struct dataset_entry *entries;
    long start_idx;
    long count;
    int target_len;
    int payload_len;
    char iface[IFNAMSIZ];
    uint64_t total_sent; // Output stat
};

// Shared Global Data
struct dataset_entry *global_entries;
long total_tuples_global;
uint8_t src_mac[6] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
uint8_t dst_mac[6] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x02};

void *sender_thread(void *arg) {
    struct thread_data *t_data = (struct thread_data *)arg;
    
    // 1. Pin Thread
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(t_data->cpu_core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    // 2. Setup Local Buffers
    struct mmsghdr msgs[BATCH_SIZE];
    struct iovec iovecs[BATCH_SIZE];
    struct raw_packet pkts[BATCH_SIZE];
    struct sockaddr_ll saddr;

    // 3. Setup Socket (Each thread needs its own to reduce locking contention)
    int sock = socket(AF_PACKET, SOCK_RAW, IPPROTO_RAW);
    if (sock < 0) { perror("Socket"); return NULL; }
    
    // Huge buffer per thread
    int sndbuf = 64 * 1024 * 1024; 
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    struct ifreq if_idx;
    memset(&if_idx, 0, sizeof(struct ifreq));
    strncpy(if_idx.ifr_name, t_data->iface, IFNAMSIZ-1);
    ioctl(sock, SIOCGIFINDEX, &if_idx);

    memset(&saddr, 0, sizeof(saddr));
    saddr.sll_family = AF_PACKET;
    saddr.sll_ifindex = if_idx.ifr_ifindex;
    saddr.sll_halen = ETH_ALEN;

    // 4. Pre-fill Templates
    for (int i = 0; i < BATCH_SIZE; i++) {
        memset(&pkts[i], 0, sizeof(struct raw_packet));
        memcpy(pkts[i].eth.h_dest, dst_mac, 6);
        memcpy(pkts[i].eth.h_source, src_mac, 6);
        pkts[i].eth.h_proto = htons(ETH_P_IP);

        pkts[i].ip.ihl = 5; pkts[i].ip.version = 4;
        pkts[i].ip.tot_len = htons(t_data->target_len - ETH_HEADER_LEN);
        pkts[i].ip.ttl = 64; pkts[i].ip.check = 0;
        pkts[i].udp.len = htons(UDP_HEADER_LEN + t_data->payload_len);

        iovecs[i].iov_base = &pkts[i];
        iovecs[i].iov_len = t_data->target_len;
        msgs[i].msg_hdr.msg_iov = &iovecs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
        msgs[i].msg_hdr.msg_name = &saddr;
        msgs[i].msg_hdr.msg_namelen = sizeof(saddr);
    }

    // 5. Sending Loop
    long current = 0;
    long end = t_data->count;
    
    while(current < end) {
        int n_pkts = BATCH_SIZE;
        if (current + n_pkts > end) n_pkts = end - current;

        for (int i = 0; i < n_pkts; i++) {
            // Calculate global index: start_idx + current + i
            struct dataset_entry *e = &global_entries[t_data->start_idx + current + i];
            pkts[i].ip.saddr = e->src_ip;
            pkts[i].ip.daddr = e->dst_ip;
            pkts[i].ip.protocol = e->proto;
            pkts[i].udp.source = e->src_port;
            pkts[i].udp.dest = e->dst_port;
        }

        int ret = sendmmsg(sock, msgs, n_pkts, 0);
        if (ret > 0) t_data->total_sent += ret;
        current += n_pkts;
    }
    
    close(sock);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <interface> <dataset> <len> <num_threads>\n", argv[0]);
        return 1;
    }

    char *iface = argv[1];
    char *dataset = argv[2];
    int len = atoi(argv[3]);
    int num_threads = atoi(argv[4]);
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    // Load Data
    FILE *fp = fopen(dataset, "rb");
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    total_tuples_global = fsize / 13;
    
    uint8_t *raw = malloc(fsize);
    global_entries = malloc(total_tuples_global * sizeof(struct dataset_entry));
    printf("[*] Loading %ld tuples...\n", total_tuples_global);
    fread(raw, 1, fsize, fp);
    fclose(fp);
    
    uint8_t *ptr = raw;
    for (long i = 0; i < total_tuples_global; i++) {
        memcpy(&global_entries[i].src_ip, ptr, 4);
        memcpy(&global_entries[i].dst_ip, ptr + 4, 4);
        memcpy(&global_entries[i].src_port, ptr + 8, 2);
        memcpy(&global_entries[i].dst_port, ptr + 10, 2);
        global_entries[i].proto = ptr[12];
        ptr += 13;
    }
    free(raw);

    // Prepare Threads
    pthread_t threads[MAX_THREADS];
    struct thread_data t_args[MAX_THREADS];
    long chunk_size = total_tuples_global / num_threads;
    
    int payload_len = len - 14 - 20 - 8;
    if (payload_len < 0) payload_len = 0;

    printf("[*] Launching %d threads...\n", num_threads);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        t_args[i].thread_id = i;
        // Pin to cores 1, 2, 3... (Leaving Core 0 for Interrupts)
        t_args[i].cpu_core = i + 1; 
        t_args[i].start_idx = i * chunk_size;
        t_args[i].count = (i == num_threads - 1) ? (total_tuples_global - (i * chunk_size)) : chunk_size;
        t_args[i].target_len = len;
        t_args[i].payload_len = payload_len;
        strncpy(t_args[i].iface, iface, IFNAMSIZ);
        t_args[i].total_sent = 0;

        pthread_create(&threads[i], NULL, sender_thread, &t_args[i]);
    }

    // Join Threads
    uint64_t total_sent_all = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_sent_all += t_args[i].total_sent;
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Done. Sent %ld packets in %.4fs\n", total_sent_all, duration);
    printf("Rate: %.2f Mpps | %.2f Mbps\n", (total_sent_all/duration)/1e6, (total_sent_all*len*8.0/duration)/1e6);

    return 0;
}
