#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arpa/inet.h> // For htons, etc.

// --- Structures (Packed to prevent compiler padding) ---
#pragma pack(push, 1)

struct pcap_global_header {
    uint32_t magic; 
    uint16_t major; 
    uint16_t minor;
    uint32_t zone; 
    uint32_t sigfigs; 
    uint32_t snaplen; 
    uint32_t network;
};

struct pcap_packet_header {
    uint32_t ts_sec; 
    uint32_t ts_usec; 
    uint32_t caplen; 
    uint32_t len;
};

struct eth_header {
    uint8_t dest[6]; 
    uint8_t src[6]; 
    uint16_t type;
};

struct ip_header {
    uint8_t ver_ihl; 
    uint8_t tos; 
    uint16_t len; 
    uint16_t id; 
    uint16_t frag;
    uint8_t ttl; 
    uint8_t proto; 
    uint16_t check; 
    uint32_t src; 
    uint32_t dst;
};

struct udp_header {
    uint16_t sport; 
    uint16_t dport; 
    uint16_t len; 
    uint16_t check;
};

struct tcp_header {
    uint16_t sport; 
    uint16_t dport;
    uint32_t seq;
    uint32_t ack;
    uint8_t  offset_res; // Data offset (4 bits) + Reserved (4 bits)
    uint8_t  flags;
    uint16_t window;
    uint16_t check;
    uint16_t urg_ptr;
};

#pragma pack(pop)

// --- Constants ---
#define ETH_HEADER_LEN 14
#define IP_HEADER_LEN  20
#define UDP_HEADER_LEN 8
#define TCP_HEADER_LEN 20

// Minimum size required to hold Ethernet + IP + TCP (no payload)
// 14 + 20 + 20 = 54 bytes
#define MIN_PKT_SIZE 54 

int main(int argc, char *argv[]) {
    if (argc != 3) { 
        printf("Usage: %s <dataset_path> <packet_size_bytes>\n", argv[0]);
        printf("Example: %s dataset.dat 64\n", argv[0]);
        return 1; 
    }

    char *input_path = argv[1];
    int target_size = atoi(argv[2]);

    if (target_size < MIN_PKT_SIZE) {
        fprintf(stderr, "Error: Packet size must be at least %d bytes to fit TCP headers.\n", MIN_PKT_SIZE);
        return 1;
    }

    FILE *fin = fopen(input_path, "rb");
    if (!fin) { perror("Error opening input file"); return 1; }

    char out_path[256];
    snprintf(out_path, sizeof(out_path), "%s.pcap", input_path);
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { perror("Error opening output file"); fclose(fin); return 1; }

    // 1. Write PCAP Global Header
    // Magic: 0xa1b2c3d4 (Microseconds), Snaplen: 65535, LinkType: 1 (Ethernet)
    struct pcap_global_header gh = {0xa1b2c3d4, 2, 4, 0, 0, 65535, 1};
    fwrite(&gh, sizeof(gh), 1, fout);

    // 2. Prepare Buffers and Static Data
    uint8_t tuple[13];
    uint8_t packet_buffer[9000]; // Max Jumbo Frame support
    memset(packet_buffer, 0, sizeof(packet_buffer)); // Zero out padding

    // Pointers into the buffer
    struct eth_header *eth = (struct eth_header *)packet_buffer;
    struct ip_header *ip   = (struct ip_header *)(packet_buffer + ETH_HEADER_LEN);
    // L4 pointers will be cast dynamically based on protocol

    // Static Ethernet Headers (Modify MACs here if needed)
    uint8_t src_mac[6] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x01};
    uint8_t dst_mac[6] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x02};
    memcpy(eth->src, src_mac, 6);
    memcpy(eth->dest, dst_mac, 6);
    eth->type = htons(0x0800); // IPv4

    // Static IP Headers
    ip->ver_ihl = 0x45; // Ver 4, Header Len 5 words (20 bytes)
    ip->tos = 0;
    ip->id = 0;
    ip->frag = 0;
    ip->ttl = 64;
    ip->check = 0; // Kernel/NIC will handle, or valid as 0 for experiments
    // IP Total Length depends on packet size, ignoring Ethernet header
    ip->len = htons(target_size - ETH_HEADER_LEN); 

    struct pcap_packet_header pcap_ph = {0, 0, target_size, target_size};
    
    long count = 0;
    long dropped = 0;

    // 3. Processing Loop
    while (fread(tuple, 1, 13, fin) == 13) {
        // tuple structure: srcIP(4) dstIP(4) sport(2) dport(2) proto(1)
        uint8_t proto = tuple[12];

        if (proto != 6 && proto != 17) {
            dropped++;
            continue; // Skip non-TCP/UDP
        }

        // Fill IP Addresses
        memcpy(&ip->src, tuple, 4);
        memcpy(&ip->dst, tuple + 4, 4);
        ip->proto = proto;

        // Construct L4
        if (proto == 6) { // TCP
            struct tcp_header *tcp = (struct tcp_header *)(packet_buffer + ETH_HEADER_LEN + IP_HEADER_LEN);
            memcpy(&tcp->sport, tuple + 8, 2);
            memcpy(&tcp->dport, tuple + 10, 2);
            
            tcp->seq = htonl(1);
            tcp->ack = 0;
            // Data Offset: 5 words (20 bytes). 0x50 = 0101 0000
            tcp->offset_res = 0x50; 
            tcp->flags = 0x02; // SYN flag
            tcp->window = htons(8192);
            tcp->check = 0;
            tcp->urg_ptr = 0;
            
            // Payload is implicitly the rest of the zeros in packet_buffer
            // TCP Header (20) + Payload = (Target - 14 - 20)
        } 
        else if (proto == 17) { // UDP
            struct udp_header *udp = (struct udp_header *)(packet_buffer + ETH_HEADER_LEN + IP_HEADER_LEN);
            memcpy(&udp->sport, tuple + 8, 2);
            memcpy(&udp->dport, tuple + 10, 2);
            
            // UDP Length = Header(8) + Payload
            // Total IP Len = 20 + UDP Len. 
            // UDP Len = IP Len - 20 = (Target - 14) - 20 = Target - 34
            udp->len = htons(target_size - ETH_HEADER_LEN - IP_HEADER_LEN);
            udp->check = 0;
            
            // Payload is implicitly the rest of the zeros in packet_buffer
            // Padding added automatically because we memset buffer to 0 and write `target_size`
        }

        // Write Packet to PCAP
        // We write the header + the full buffer up to target_size
        fwrite(&pcap_ph, sizeof(pcap_ph), 1, fout);
        fwrite(packet_buffer, 1, target_size, fout);

        count++;
        if (count % 500000 == 0) {
            printf(" Processed %ld packets...\r", count);
            fflush(stdout);
        }
    }

    fclose(fin);
    fclose(fout);

    printf("\nConversion Complete.\n");
    printf(" - Written: %ld packets\n", count);
    printf(" - Dropped: %ld packets (Not TCP/UDP)\n", dropped);
    printf(" - Size:    %d bytes per packet\n", target_size);
    printf(" - Output:  %s\n", out_path);

    return 0;
}
