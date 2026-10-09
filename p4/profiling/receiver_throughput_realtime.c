#include <stdint.h>
#include <signal.h>
#include <inttypes.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <getopt.h>

#define RX_RING_SIZE 2048
#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 64

static volatile bool force_quit = false;

// Statistics per core
struct worker_stats {
    uint64_t packets;
    uint64_t bytes;
} stats[RTE_MAX_LCORE];

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM)
        force_quit = true;
}

// --- Worker Thread ---
static int rx_worker(void *arg) {
    uint16_t port_id = (uint16_t)(uintptr_t)arg;
    uint16_t queue_id = rte_lcore_index(rte_lcore_id());
    struct rte_mbuf *bufs[BURST_SIZE];

    printf("Core %u listening on Queue %u...\n", rte_lcore_id(), queue_id);

    while (!force_quit) {
        uint16_t nb_rx = rte_eth_rx_burst(port_id, queue_id, bufs, BURST_SIZE);

        if (nb_rx > 0) {
            stats[rte_lcore_id()].packets += nb_rx;
            for (int i = 0; i < nb_rx; i++) {
                stats[rte_lcore_id()].bytes += bufs[i]->pkt_len;
                rte_pktmbuf_free(bufs[i]); // Free immediately to keep pool available
            }
        }
    }
    return 0;
}

// --- Main ---
int main(int argc, char *argv[]) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_exit(EXIT_FAILURE, "EAL Init failed\n");
    argc -= ret; argv += ret;

    // Parse simple arg: -p PORT_ID
    int target_port = -1;
    if (argc > 1 && strcmp(argv[0], "-p") == 0) target_port = atoi(argv[1]);
    
    // Find Port
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) rte_exit(EXIT_FAILURE, "No ports found\n");
    uint16_t port_id = (target_port >= 0) ? target_port : rte_eth_find_next_owned_by(0, RTE_ETH_DEV_NO_OWNER);

    // Setup RSS (Distribute traffic to all enabled cores)
    unsigned int n_lcores = rte_lcore_count();
    struct rte_eth_conf port_conf = {
        .rxmode = { .mq_mode = RTE_ETH_MQ_RX_RSS },
        .rx_adv_conf = {
            .rss_conf = {
                .rss_key = NULL,
                .rss_hf = RTE_ETH_RSS_IP | RTE_ETH_RSS_UDP | RTE_ETH_RSS_TCP,
            },
        },
    };

    struct rte_mempool *mbuf_pool = 
        rte_pktmbuf_pool_create("MBUF_POOL",
                                NUM_MBUFS * n_lcores,
                                MBUF_CACHE_SIZE,
                                0,
                                RTE_MBUF_DEFAULT_BUF_SIZE,
                                rte_socket_id());
    
    if (!mbuf_pool) rte_exit(EXIT_FAILURE, "Pool creation failed\n");

    if (rte_eth_dev_configure(port_id, 1, n_lcores, &port_conf) < 0)
        rte_exit(EXIT_FAILURE, "Config failed (Check if NIC supports RSS)\n");

    // Setup RX Queues (1 per core)
    for (int i = 0; i < n_lcores; i++) {
        if (rte_eth_rx_queue_setup(port_id, i, RX_RING_SIZE, rte_eth_dev_socket_id(port_id), NULL, mbuf_pool) < 0)
            rte_exit(EXIT_FAILURE, "RXQ %d setup failed\n", i);
    }
    // Setup 1 Dummy TX Queue (Required by some drivers)
    if (rte_eth_tx_queue_setup(port_id, 0, 512, rte_eth_dev_socket_id(port_id), NULL) < 0)
        rte_exit(EXIT_FAILURE, "TXQ setup failed\n");

    if (rte_eth_dev_start(port_id) < 0) rte_exit(EXIT_FAILURE, "Start failed\n");
    rte_eth_promiscuous_enable(port_id);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Launch Workers
    unsigned lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        rte_eal_remote_launch(rx_worker, (void*)(uintptr_t)port_id, lcore_id);
    }

    printf("=== Receiver Running on Port %d with %d Cores ===\n", port_id, n_lcores);
    
    // Master Core: Print Stats Loop
    uint64_t last_pkts = 0;
    uint64_t last_bytes = 0;
    
    while (!force_quit) {
        rte_delay_us_block(1000000); // 1 Second
        
        uint64_t total_pkts = 0;
        uint64_t total_bytes = 0;
        
        // Sum up stats from all cores
        RTE_LCORE_FOREACH(lcore_id) {
            total_pkts += stats[lcore_id].packets;
            total_bytes += stats[lcore_id].bytes;
        }

        uint64_t pps = total_pkts - last_pkts;
        uint64_t bps = (total_bytes - last_bytes) * 8;
        
        printf("Throughput: %lu Mpps | %.2f Gbps\n", pps / 1000000, bps / 1e9);

        last_pkts = total_pkts;
        last_bytes = total_bytes;
    }

    rte_eal_mp_wait_lcore();
    rte_eth_dev_stop(port_id);
    return 0;
}
