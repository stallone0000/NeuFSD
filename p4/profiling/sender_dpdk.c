#include <stdint.h>
#include <inttypes.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <time.h>
#include <string.h>
#include <getopt.h>

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 64  
#define TX_RING_SIZE 2048

// Globals
static char *TARGET_FILE = NULL;
static int TARGET_LEN = 64;
static int TARGET_PORT = -1;

static struct rte_ether_addr src_mac = {{0x00, 0x00, 0x00, 0x00, 0x00, 0x01}};
static struct rte_ether_addr dst_mac = {{0x00, 0x00, 0x00, 0x00, 0x00, 0x02}};

struct dataset_entry {
    uint32_t src_ip; uint32_t dst_ip;
    uint16_t src_port; uint16_t dst_port;
    uint8_t proto;
};

struct dataset_entry *entries;
long total_tuples;
struct rte_mempool *mbuf_pool;

// Stats per core
struct worker_stats {
    uint64_t sent;
    double duration;
} stats[RTE_MAX_LCORE];

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int load_dataset(const char *filename) {
    printf("[*] Loading Dataset: %s\n", filename);
    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    total_tuples = fsize / 13;
    entries = malloc(total_tuples * sizeof(struct dataset_entry));
    uint8_t *raw_data = malloc(fsize);

    if (!entries || !raw_data) return -1;
    if (fread(raw_data, 1, fsize, fp) != fsize) return -1;
    fclose(fp);

    uint8_t *ptr = raw_data;
    for (long i = 0; i < total_tuples; i++) {
        memcpy(&entries[i].src_ip, ptr, 4);
        memcpy(&entries[i].dst_ip, ptr + 4, 4);
        memcpy(&entries[i].src_port, ptr + 8, 2);
        memcpy(&entries[i].dst_port, ptr + 10, 2);
        entries[i].proto = ptr[12];
        ptr += 13;
    }
    free(raw_data);
    return 0;
}

// --- WORKER FUNCTION ---
static int worker_main(void *arg) {
    uint16_t port_id = (uint16_t)(uintptr_t)arg;
    unsigned lcore_id = rte_lcore_id();
    
    // Each core gets its own TX Queue ID.
    // We assume queue_id maps 1:1 to lcore index for simplicity in this example
    // But safely, we just use a static counter or pass it in. 
    // Here we use the logical core index (0, 1, 2...)
    // Note: This requires Lcore IDs to be contiguous 0..N-1 usually, 
    // or we need to map lcore_id to queue_id. 
    // Let's rely on rte_lcore_index(lcore_id).
    unsigned queue_id = rte_lcore_index(lcore_id);

    struct rte_mbuf *bufs[BURST_SIZE];
    
    // Calculate split
    unsigned int n_workers = rte_lcore_count();
    long chunk_size = total_tuples / n_workers;
    long start_idx = queue_id * chunk_size;
    long end_idx = (queue_id == n_workers - 1) ? total_tuples : start_idx + chunk_size;
    long count = end_idx - start_idx;

    // Header Setup
    uint16_t ip_len = TARGET_LEN - sizeof(struct rte_ether_hdr);
    uint16_t udp_len = ip_len - sizeof(struct rte_ipv4_hdr);
    
    printf("Core %u (Queue %u) starting. Range: %ld -> %ld (%ld pkts)\n", 
           lcore_id, queue_id, start_idx, end_idx, count);

    double start_time = get_time_sec();
    long current = 0;
    long total_sent = 0;

    while (current < count) {
        if (unlikely(rte_pktmbuf_alloc_bulk(mbuf_pool, bufs, BURST_SIZE) < 0)) continue;

        int this_burst = BURST_SIZE;
        if (current + this_burst > count) this_burst = count - current;

        for (int i = 0; i < this_burst; i++) {
            struct dataset_entry *e = &entries[start_idx + current + i];
            struct rte_mbuf *m = bufs[i];
            m->pkt_len = TARGET_LEN; m->data_len = TARGET_LEN;

            struct rte_ether_hdr *eth = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
            rte_ether_addr_copy(&src_mac, &eth->src_addr);
            rte_ether_addr_copy(&dst_mac, &eth->dst_addr);
            eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);

            struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(eth + 1);
            ip->version_ihl = 0x45; ip->total_length = rte_cpu_to_be_16(ip_len);
            ip->next_proto_id = e->proto;
            ip->src_addr = e->src_ip; ip->dst_addr = e->dst_ip;

            struct rte_udp_hdr *udp = (struct rte_udp_hdr *)(ip + 1);
            udp->src_port = e->src_port; udp->dst_port = e->dst_port;
            udp->dgram_len = rte_cpu_to_be_16(udp_len);
        }

        uint16_t nb_tx = rte_eth_tx_burst(port_id, queue_id, bufs, this_burst);

        if (unlikely(nb_tx < this_burst)) {
            for (int k = nb_tx; k < BURST_SIZE; k++) rte_pktmbuf_free(bufs[k]);
        } else {
             if (this_burst < BURST_SIZE) for (int k = this_burst; k < BURST_SIZE; k++) rte_pktmbuf_free(bufs[k]);
        }
        
        // DPDK's tx_burst returns packets *enqueued*, not necessarily sent on wire yet.
        // We count them as sent.
        current += nb_tx;
        total_sent += nb_tx;
    }

    stats[lcore_id].duration = get_time_sec() - start_time;
    stats[lcore_id].sent = total_sent;
    return 0;
}

int parse_app_args(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "f:s:p:")) != -1) {
        switch (opt) {
            case 'f': TARGET_FILE = strdup(optarg); break;
            case 's': TARGET_LEN = atoi(optarg); break;
            case 'p': TARGET_PORT = atoi(optarg); break;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_exit(EXIT_FAILURE, "EAL Init failed\n");
    argc -= ret; argv += ret;

    if (parse_app_args(argc, argv) < 0 || TARGET_FILE == NULL)
        rte_exit(EXIT_FAILURE, "Usage: -- -f file -s size\n");

    if (load_dataset(TARGET_FILE) < 0) rte_exit(EXIT_FAILURE, "Load failed\n");

    // Port Setup
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) rte_exit(EXIT_FAILURE, "No ports\n");
    uint16_t port_id = (TARGET_PORT >= 0) ? TARGET_PORT : rte_eth_find_next_owned_by(0, RTE_ETH_DEV_NO_OWNER);

    // Number of cores enabled via -l
    unsigned int n_lcores = rte_lcore_count();
    printf("[*] Enabled Cores: %u\n", n_lcores);

    mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL", NUM_MBUFS * n_lcores,
        MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!mbuf_pool) rte_exit(EXIT_FAILURE, "Pool creation failed\n");

    struct rte_eth_conf port_conf = { .rxmode = { } };
    
    // IMPORTANT: Configure Multiple TX Queues (1 per core)
    if (rte_eth_dev_configure(port_id, 1, n_lcores, &port_conf) < 0)
        rte_exit(EXIT_FAILURE, "Config failed\n");

    // Setup 1 TX Queue for each lcore index
    for (int i = 0; i < n_lcores; i++) {
        if (rte_eth_tx_queue_setup(port_id, i, TX_RING_SIZE, rte_eth_dev_socket_id(port_id), NULL) < 0)
            rte_exit(EXIT_FAILURE, "TXQ %d setup failed\n", i);
    }
    // We still need 1 RX queue even if unused
    if (rte_eth_rx_queue_setup(port_id, 0, 128, rte_eth_dev_socket_id(port_id), NULL, mbuf_pool) < 0)
        rte_exit(EXIT_FAILURE, "RXQ setup failed\n");

    if (rte_eth_dev_start(port_id) < 0) rte_exit(EXIT_FAILURE, "Start failed\n");
    rte_eth_promiscuous_enable(port_id);

    // LAUNCH WORKERS
    // 1. Launch on slave cores
    unsigned lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        rte_eal_remote_launch(worker_main, (void*)(uintptr_t)port_id, lcore_id);
    }
    // 2. Run on Master core too
    worker_main((void*)(uintptr_t)port_id);

    // 3. Wait for all
    rte_eal_mp_wait_lcore();

    // Stats
    rte_delay_us_block(200000);
    uint64_t total_sent = 0;
    double max_dur = 0;
    RTE_LCORE_FOREACH(lcore_id) {
        total_sent += stats[lcore_id].sent;
        if (stats[lcore_id].duration > max_dur) max_dur = stats[lcore_id].duration;
    }

    printf("\n=== Total ===\n");
    printf("Sent: %ld pkts\n", total_sent);
    printf("Rate: %.2f Mpps | %.2f Mbps\n", 
        (total_sent/max_dur)/1e6, (total_sent*TARGET_LEN*8.0/max_dur)/1e6);

    rte_eth_dev_stop(port_id);
    return 0;
}
