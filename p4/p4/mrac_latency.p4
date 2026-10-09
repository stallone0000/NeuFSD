#include <core.p4>
#include <tna.p4>

#define MRAC_INDEX_LEN 17
#define MRAC_BUCKET_SIZE (1 << MRAC_INDEX_LEN)

//#define ENABLE_TCP

#include "common/headers.p4"
#include "common/utils.p4"
#include "common/params.p4"

header bridge_md_h {
    bit<32> ingress_time;
}

header latency_h {
    bit<32> latency;
}

struct header_t {
    bridge_md_h bridge_md;
    ethernet_h ethernet;
    ipv4_h     ipv4;
#ifdef ENABLE_TCP
    tcp_h      tcp;
#endif
    latency_h  latency;
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
            hdr.bridge_md.setValid();
            hdr.bridge_md.ingress_time = (bit<32>)ig_intr_prsr_md.global_tstamp;
            
            forward(DST_PORT);
        }
        //bypass_egress.apply(ig_intr_tm_md);
    }
}

parser SwitchEgressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t eg_md,
        out egress_intrinsic_metadata_t eg_intr_md) {
    TofinoEgressParser() tofino_eg_parser;

    state start {
        tofino_eg_parser.apply(pkt, eg_intr_md);
        transition parse_bridge_md;
    }

    state parse_bridge_md {
        pkt.extract(hdr.bridge_md);
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
        transition parse_latency;
    }

    state parse_latency {
        pkt.extract(hdr.latency);
        transition accept;
    }
}

control SwitchEgress(
        inout header_t hdr,
        inout metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    apply {
        if (hdr.ipv4.isValid()) {
            hdr.latency.setValid();
            bit<32> ingress_time = hdr.bridge_md.ingress_time;
            bit<32> egress_time = (bit<32>)eg_intr_md_from_prsr.global_tstamp;
            bit<32> latency = egress_time - ingress_time;
            hdr.latency.latency = latency;
            hdr.bridge_md.setInvalid();
        }
    }
}

control SwitchEgressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t eg_md,
        in egress_intrinsic_metadata_for_deparser_t eg_intr_dprsr_md) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.latency);
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;
