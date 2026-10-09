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

struct header_t {
    ethernet_h  ethernet;
    ipv4_h      ipv4;
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

            forward(DST_PORT);
            bypass_egress.apply(ig_intr_tm_md);
        }
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
