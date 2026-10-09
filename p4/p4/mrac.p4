#include <core.p4>
#include <tna.p4>

#define MRAC_INDEX_LEN 17
#define MRAC_BUCKET_SIZE (1 << MRAC_INDEX_LEN)

//#define ENABLE_TCP

#include "common/headers.p4"
#include "common/utils.p4"
#include "common/params.p4"

struct header_t {
    ethernet_h ethernet;
    ipv4_h     ipv4;
#ifdef ENABLE_TCP
    tcp_h      tcp;
#endif
}

struct metadata_t {}

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    TofinoIngressParser() tofino_parser;

    state start {
        tofino_parser.apply(pkt, ig_intr_md);
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select (hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4 : parse_ipv4;
            default : reject;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
#ifdef ENABLE_TCP
        transition select(hdr.ipv4.protocol) {
            IP_PROTOCOLS_TCP : parse_tcp;
            default : reject;
        }
#else
        transition accept;
#endif
    }

#ifdef ENABLE_TCP
    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }
#endif
}


control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {
    apply {
        pkt.emit(hdr);
    }
}


control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {
    Counter<bit<32>, bit<MRAC_INDEX_LEN>>(MRAC_BUCKET_SIZE, CounterType_t.PACKETS) counter;
    Hash<bit<MRAC_INDEX_LEN>>(HashAlgorithm_t.CRC32) hasher;

    BypassEgress() bypass_egress;

#ifdef ENABLE_TCP
    action count(bit<32> src_addr, bit<32> dst_addr,
                 bit<16> src_port, bit<16> dst_port, bit<8> proto) {
        bit<MRAC_INDEX_LEN> idx = hasher.get({
            src_addr, dst_addr,
            src_port, dst_port, proto
        });
        counter.count(idx);
    }
#else
    action count(bit<32> src_addr) {
        bit<MRAC_INDEX_LEN> idx = hasher.get(src_addr);
        counter.count(idx);
    }
#endif

    action forward(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port; 
    }

    apply {
        if (hdr.ipv4.isValid()) {
#ifdef ENABLE_TCP
            if (hdr.tcp.isValid()) {
                count(hdr.ipv4.src_addr, hdr.ipv4.dst_addr,
                      hdr.tcp.src_port, hdr.tcp.dst_port, hdr.ipv4.protocol);
            }
#else
            count(hdr.ipv4.src_addr);
#endif
            forward(DST_PORT);
        }
        bypass_egress.apply(ig_intr_tm_md);
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
