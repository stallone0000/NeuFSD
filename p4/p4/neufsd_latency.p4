#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/utils.p4"
#include "common/params.p4"

#define NEUFSD_BUCKET_NUM 8192
#define NEUFSD_ARRAY_SIZE 131072
#define NEUFSD_ARRAY_INDEX_LEN 17
#define NEUFSD_LAMBDA 32
#define NEUFSD_LAMBDA_WIDTH 5

typedef bit<32> flow_id_t;

struct metadata_t {
    flow_id_t flow_id;      // current flow id
    bit<32>   vote_tot_div; // vote_total / LAMBDA
}

header bridge_md_h {
    bit<32> ingress_time;
}

header latency_h {
    bit<32> latency;
}

struct header_t {
    bridge_md_h bridge_md;
    ethernet_h  ethernet;
    ipv4_h      ipv4;
    latency_h   latency;
}

struct bucket_p1_t {
    bit<32> vote_tot;
}

struct bucket_p2_t {
    flow_id_t flow_id;
    bit<32>   vote_pos;
}

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {
    TofinoIngressParser() tofino_ig_parser;

    state start {
        tofino_ig_parser.apply(pkt, ig_intr_md);
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
        ig_md.flow_id = hdr.ipv4.src_addr;
        transition accept;
    }
}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md) {
    apply {
        pkt.emit(hdr.bridge_md);
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
    }
}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {
    // Hot Filter (Elastic-Sketch-like)
    Register<bucket_p1_t, _>(NEUFSD_BUCKET_NUM, {0}) bucket_p1;
    Register<bucket_p2_t, _>(NEUFSD_BUCKET_NUM, {0, 0}) bucket_p2;
    RegisterAction<_, _, bit<32>>(bucket_p1) bucket_p1_update = {
        void apply(inout bucket_p1_t bp1, out bit<32> vote_tot) {
            bp1.vote_tot = bp1.vote_tot + 1;
            vote_tot = bp1.vote_tot;
        }
    };
    RegisterAction<_, _, flow_id_t>(bucket_p2) bucket_p2_update = {
        void apply(inout bucket_p2_t bp2, out flow_id_t evicted_flow_id) {
            evicted_flow_id = 0;
            if (bp2.flow_id == ig_md.flow_id) {
                bp2.vote_pos = bp2.vote_pos + 1;
            } else if (ig_md.vote_tot_div >= bp2.vote_pos) {
                evicted_flow_id = bp2.flow_id; // TODO: how to output vote_pos?
                bp2.flow_id = ig_md.flow_id;
                bp2.vote_pos = bp2.vote_pos + 1;
            }
        }
    };

    action forward(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port; 
    }
    
    // Array (MRAC-like)
    Hash<bit<NEUFSD_ARRAY_INDEX_LEN>>(HashAlgorithm_t.CRC32) counter_hasher;
    Counter<bit<32>, bit<NEUFSD_ARRAY_INDEX_LEN>>(NEUFSD_ARRAY_SIZE, CounterType_t.PACKETS) counter;

    BypassEgress() bypass_egress;

    apply {
        if (hdr.ipv4.isValid()) {
            // Hot Filter (Elastic-Sketch-like)
            bit<32> vote_tot = bucket_p1_update.execute(ig_md.flow_id); // TODO: hash before indexing buckets
            ig_md.vote_tot_div = vote_tot >> NEUFSD_LAMBDA_WIDTH;
            flow_id_t evicted_flow_id = bucket_p2_update.execute(ig_md.flow_id);
            
            // Array (MRAC-like)
            if (evicted_flow_id != 0 && evicted_flow_id != ig_md.flow_id) {
                bit<NEUFSD_ARRAY_INDEX_LEN> idx = counter_hasher.get(evicted_flow_id);
                counter.count(idx);
            }

            hdr.bridge_md.setValid();
            hdr.bridge_md.ingress_time = (bit<32>)ig_intr_prsr_md.global_tstamp;
            
            forward(DST_PORT);
            //bypass_egress.apply(ig_intr_tm_md);
        }
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
