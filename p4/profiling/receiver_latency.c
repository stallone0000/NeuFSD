#include <stdint.h>
#include <signal.h>
#include <stdbool.h>
#include <inttypes.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_byteorder.h>
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
    uint64_t total_latency_ns; 
} __rte_cache_aligned;

struct worker_stats stats[RTE_MAX_LCORE];

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM)
        force_quit = true;
}

static int rx_worker(void *arg) {
    uint16_t port_id = (uint16_t)(uintptr_t)arg;
    unsigned int lcore_id = rte_lcore_id();
    
    // Simple Queue Mapping
    unsigned int queue_id = 0;
    unsigned int i;
    RTE_LCORE_FOREACH_WORKER(i) {
        if (i == lcore_id) break;
        queue_id++;
    }

    struct rte_mbuf *bufs[BURST_SIZE];
    stats[lcore_id].packets = 0;
    stats[lcore_id].total_latency_ns = 0;

    printf("Core %u reading Raw IP Latency on Queue %u...\n", lcore_id, queue_id);

    while (!force_quit) {
        uint16_t nb_rx = rte_eth_rx_burst(port_id, queue_id, bufs, BURST_SIZE);

        if (likely(nb_rx > 0)) {
            uint64_t batch_latency = 0;
            uint64_t batch_pkts = 0;
            
            for (int k = 0; k < nb_rx; k++) {
                struct rte_mbuf *m = bufs[k];
                
                struct rte_ether_hdr *eth = rte_pktmbuf_mtod(m, struct rte_ether_hdr *);
                
                // 1. Check for IPv4
                if (eth->ether_type == rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
                    struct rte_ipv4_hdr *ip = (struct rte_ipv4_hdr *)(eth + 1);
                    
                    // Calculate IP Header Length (Usually 20 bytes, but check IHL to be safe)
                    // version_ihl contains Version (4 bits) + IHL (4 bits)
                    // IHL is number of 32-bit words. * 4 gives bytes.
                    uint8_t ihl = (ip->version_ihl & 0x0F) * 4;
                    
                    // 2. Check Packet Length
                    // Must contain Eth(14) + IP(ihl) + Timestamp(4)
                    if (rte_pktmbuf_data_len(m) >= (sizeof(struct rte_ether_hdr) + ihl + 4)) {
                        
                        // 3. Extract 32-bit Timestamp directly after IP header
                        // We do NOT check ip->next_proto_id. We assume the data follows IP.
                        void *payload_ptr = (void *)((uint8_t *)ip + ihl);
                        
                        uint32_t raw_be;
                        memcpy(&raw_be, payload_ptr, sizeof(uint32_t));
                        
                        // Convert Big Endian (Network) to CPU
                        uint32_t lat_ns = rte_be_to_cpu_32(raw_be);
                        
                        batch_latency += lat_ns;
                        batch_pkts++;
                    }
                }
                rte_pktmbuf_free(m);
            }
            
            stats[lcore_id].packets += batch_pkts; 
            stats[lcore_id].total_latency_ns += batch_latency;
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
    if (n_lcores < 2) rte_exit(EXIT_FAILURE, "Need at least 2 cores (-l 0,1)\n");
    unsigned int n_workers = n_lcores - 1; 

    // RSS Configuration to spread load based on IP
    struct rte_eth_conf port_conf = {
        .rxmode = { .mq_mode = RTE_ETH_MQ_RX_RSS },
        .rx_adv_conf = {
            .rss_conf = { .rss_key = NULL, .rss_hf = RTE_ETH_RSS_IP }
        },
    };

    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("MBUF_POOL",
        NUM_MBUFS * n_lcores, MBUF_CACHE_SIZE, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    
    if (!mbuf_pool) rte_exit(EXIT_FAILURE, "Pool failed\n");
    if (rte_eth_dev_configure(port_id, n_workers, 1, &port_conf) < 0) rte_exit(EXIT_FAILURE, "Config failed\n");

    for (int i = 0; i < n_workers; i++) 
        if (rte_eth_rx_queue_setup(port_id, i, RX_RING_SIZE, rte_eth_dev_socket_id(port_id), NULL, mbuf_pool) < 0)
            rte_exit(EXIT_FAILURE, "RXQ failed\n");
            
    if (rte_eth_tx_queue_setup(port_id, 0, TX_RING_SIZE, rte_eth_dev_socket_id(port_id), NULL) < 0)
        rte_exit(EXIT_FAILURE, "TXQ failed\n");

    rte_eth_dev_start(port_id);
    rte_eth_promiscuous_enable(port_id);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    unsigned lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        rte_eal_remote_launch(rx_worker, (void*)(uintptr_t)port_id, lcore_id);
    }

    printf("=== Raw IP Latency Receiver Running on Port %d ===\n", port_id);

    bool active = false;
    uint64_t prev_total = 0;

    // Monitor Loop
    while (!force_quit) {
        rte_delay_us_block(100000); // 0.1s poll

        uint64_t current_total = 0;
        RTE_LCORE_FOREACH_WORKER(lcore_id) current_total += stats[lcore_id].packets;

        if (!active && current_total > 0) {
            printf("[*] Traffic detected!\n");
            active = true;
            prev_total = current_total;
        } else if (active) {
            // Optional: Auto-stop logic if needed
            prev_total = current_total;
        }
    }

    // Results
    uint64_t final_pkts = 0;
    uint64_t final_lat_sum = 0;

    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        final_pkts += stats[lcore_id].packets;
        final_lat_sum += stats[lcore_id].total_latency_ns;
    }

    printf("\n=== RESULTS ===\n");
    printf("Total Packets: %lu\n", final_pkts);
    
    if (final_pkts > 0) {
        double avg_latency = (double)final_lat_sum / (double)final_pkts;
        printf("----------------------------------\n");
        printf("Average Latency: %.2f (switch time units)\n", avg_latency);
        printf("----------------------------------\n");
    } else {
        printf("No valid latency packets received.\n");
    }

    rte_eal_mp_wait_lcore();
    rte_eth_dev_stop(port_id);
    return 0;
}
