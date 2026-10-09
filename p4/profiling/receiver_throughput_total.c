#include <stdint.h>
#include <signal.h>
#include <stdbool.h>
#include <inttypes.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <time.h>
#include <string.h>

#define RX_RING_SIZE 2048
#define TX_RING_SIZE 512
#define NUM_MBUFS 16383 
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 64
#define SILENCE_TIMEOUT_SEC 2.0 

static volatile bool force_quit = false;

struct worker_stats {
    uint64_t packets;
    uint64_t bytes;
} __rte_cache_aligned;

struct worker_stats stats[RTE_MAX_LCORE];

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM)
        force_quit = true;
}

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int rx_worker(void *arg) {
    uint16_t port_id = (uint16_t)(uintptr_t)arg;
    unsigned int lcore_id = rte_lcore_id();
    
    // We need to determine which RX Queue ID this core should handle.
    // Since we launch workers sequentially, we need to map lcore_id -> queue_id.
    // A simple way is to count how many workers have ID < current_ID.
    unsigned int queue_id = 0;
    unsigned int i;
    RTE_LCORE_FOREACH_WORKER(i) {
        if (i == lcore_id) break;
        queue_id++;
    }

    struct rte_mbuf *bufs[BURST_SIZE];
    stats[lcore_id].packets = 0;
    stats[lcore_id].bytes = 0;

    printf("Core %u listening on Queue %u...\n", lcore_id, queue_id);

    while (!force_quit) {
        uint16_t nb_rx = rte_eth_rx_burst(port_id, queue_id, bufs, BURST_SIZE);

        if (likely(nb_rx > 0)) {
            uint64_t b_count = 0;
            for (int k = 0; k < nb_rx; k++) {
                b_count += bufs[k]->pkt_len;
                rte_pktmbuf_free(bufs[k]);
            }
            stats[lcore_id].packets += nb_rx;
            stats[lcore_id].bytes += b_count;
        }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) rte_exit(EXIT_FAILURE, "EAL Init failed\n");
    argc -= ret; argv += ret;

    int target_port = -1;
    if (argc > 1 && strcmp(argv[0], "-p") == 0) target_port = atoi(argv[1]);
    
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) rte_exit(EXIT_FAILURE, "No ports found\n");
    uint16_t port_id = (target_port >= 0) ? target_port : rte_eth_find_next_owned_by(0, RTE_ETH_DEV_NO_OWNER);

    unsigned int n_lcores = rte_lcore_count();
    if (n_lcores < 2) {
        rte_exit(EXIT_FAILURE, "Need at least 2 cores (1 Master + 1 Worker). Use -l 0,1\n");
    }
    unsigned int n_workers = n_lcores - 1; 

    // RSS Configuration
    struct rte_eth_conf port_conf = {
        .rxmode = { .mq_mode = RTE_ETH_MQ_RX_RSS },
        .rx_adv_conf = {
            .rss_conf = {
                .rss_key = NULL,
                // Try standard IP RSS. If NIC fails, remove flags one by one.
                .rss_hf = RTE_ETH_RSS_IP | RTE_ETH_RSS_UDP | RTE_ETH_RSS_TCP,
            },
        },
    };

    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
        NUM_MBUFS * n_lcores, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    
    if (!mbuf_pool) rte_exit(EXIT_FAILURE, "Pool creation failed\n");

    // --- FIX IS HERE: Swapped n_workers (RX) and 1 (TX) ---
    // rte_eth_dev_configure(port, nb_rx_q, nb_tx_q, conf)
    if (rte_eth_dev_configure(port_id, n_workers, 1, &port_conf) < 0)
        rte_exit(EXIT_FAILURE, "Config failed (Check RSS support)\n");

    // Setup RX Queues (One for each Worker)
    for (int i = 0; i < n_workers; i++) {
        if (rte_eth_rx_queue_setup(port_id, i, RX_RING_SIZE, rte_eth_dev_socket_id(port_id), NULL, mbuf_pool) < 0)
            rte_exit(EXIT_FAILURE, "RXQ %d setup failed\n", i);
    }
    // Setup dummy TX
    if (rte_eth_tx_queue_setup(port_id, 0, TX_RING_SIZE, rte_eth_dev_socket_id(port_id), NULL) < 0)
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

    printf("=== Waiting for traffic on Port %d (%d Workers) ===\n", port_id, n_workers);

    bool active = false;
    double start_time = 0;
    double last_pkt_time = 0;
    uint64_t prev_total = 0;

    while (!force_quit) {
        rte_delay_us_block(1000); // 1ms poll

        uint64_t current_total = 0;
        RTE_LCORE_FOREACH_WORKER(lcore_id) {
            current_total += stats[lcore_id].packets;
        }

        double now = get_time_sec();

        if (!active) {
            if (current_total > 0) {
                printf("[*] Traffic detected! Starting timer...\n");
                active = true;
                start_time = now;
                last_pkt_time = now;
                prev_total = current_total;
            }
        } else {
            if (current_total > prev_total) {
                last_pkt_time = now;
                prev_total = current_total;
            } else {
                if ((now - last_pkt_time) > SILENCE_TIMEOUT_SEC) {
                    printf("[*] Silence detected. Stopping.\n");
                    force_quit = true;
                }
            }
        }
    }

    double duration = last_pkt_time - start_time;
    uint64_t final_total_pkts = 0;
    uint64_t final_total_bytes = 0;

    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        final_total_pkts += stats[lcore_id].packets;
        final_total_bytes += stats[lcore_id].bytes;
    }

    printf("\n=== RESULT ===\n");
    printf("Total Packets: %lu\n", final_total_pkts);
    printf("Total Bytes:   %lu\n", final_total_bytes);
    printf("Duration:      %.6f seconds\n", duration);
    
    if (duration > 0.000001) {
        double mpps = (final_total_pkts / duration) / 1e6;
        double gbps = (final_total_bytes * 8.0 / duration) / 1e9;
        printf("----------------------------------\n");
        printf("AVG THROUGHPUT: %.2f Mpps | %.2f Gbps\n", mpps, gbps);
        printf("----------------------------------\n");
    }

    rte_eal_mp_wait_lcore();
    rte_eth_dev_stop(port_id);
    return 0;
}
