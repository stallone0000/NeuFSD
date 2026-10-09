#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/utils.p4"

#define ELASTIC_BUCKET_NUM 8192
#define ELASTIC_LAMBDA 32
#define ELASTIC_LAMBDA_WIDTH 5

typedef bit<32> flow_id_t;

struct metadata_t {
    flow_id_t flow_id;      // current flow id
    bit<32>   vote_tot_div; // vote_total / LAMBDA
    bool      running;      // whether to continue the algorithm
}

// Same as metadata, used to pass metadata between ingress and egress
header bridge_md_h {
    flow_id_t flow_id;
    bit<32>   vote_tot_div;
    bit<8>    running; // byte aligned
}

struct header_t {
    bridge_md_h bridge_md;
    ethernet_h  ethernet;
    ipv4_h      ipv4;
}

struct heavy_bucket_p1_t {
    bit<32>   vote_tot;
}

struct heavy_bucket_p2_t {
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
        transition select(hdr.ipv4.protocol) {
            IP_PROTOCOLS_TCP : accept;
            default : reject;
        }
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

control ElasticSketchHeavyPart(inout metadata_t md) {
    Hash<flow_id_t>(HashAlgorithm_t.IDENTITY) flow_id_hash;
    
    Register<heavy_bucket_p1_t, _>(ELASTIC_BUCKET_NUM, {0}) heavy_bucket_p1;
    Register<heavy_bucket_p2_t, _>(ELASTIC_BUCKET_NUM, {0, 0}) heavy_bucket_p2;
    RegisterAction<_, _, bit<32>>(heavy_bucket_p1) heavy_bucket_p1_update = {
        void apply(inout heavy_bucket_p1_t hbp1, out bit<32> vote_tot) {
            hbp1.vote_tot = hbp1.vote_tot + 1;
            vote_tot = hbp1.vote_tot;
        }
    };
    RegisterAction<_, _, bit<32>>(heavy_bucket_p2) heavy_bucket_p2_update = {
        void apply(inout heavy_bucket_p2_t hbp2, out flow_id_t evicted_flow_id) {
            evicted_flow_id = 0;
            if (hbp2.flow_id == md.flow_id) {
                hbp2.vote_pos = hbp2.vote_pos + 1; // TODO: should add previous vote_pos?
            } else if (md.vote_tot_div >= hbp2.vote_pos) {
                evicted_flow_id = hbp2.flow_id;
                hbp2.flow_id = md.flow_id;
                hbp2.vote_pos = hbp2.vote_pos + 1; // TODO: should add previous vote_pos?
            }
        }
    };

    apply {
        // TODO: hash flow_id
        bit<32> vote_tot = heavy_bucket_p1_update.execute(md.flow_id);
        md.vote_tot_div = vote_tot >> ELASTIC_LAMBDA_WIDTH;
        flow_id_t evicted_flow_id = heavy_bucket_p2_update.execute(md.flow_id);
        if (evicted_flow_id == 0 || evicted_flow_id == md.flow_id) {
            md.running = false;
        } else {
            md.flow_id = evicted_flow_id;
        }
    }
}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t ig_md,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_intr_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_intr_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_intr_tm_md) {
    ElasticSketchHeavyPart() heavy_part_s1;
    ElasticSketchHeavyPart() heavy_part_s2;
    ElasticSketchHeavyPart() heavy_part_s3;

    apply {
        if (hdr.ipv4.isValid()) {
            // Initialize metadata
            ig_md.flow_id = hdr.ipv4.src_addr;
            ig_md.running = true;

            // Heavy parts (1-3)
            heavy_part_s1.apply(ig_md);
            if (ig_md.running)
                heavy_part_s2.apply(ig_md);
            //if (ig_md.running)
            //    heavy_part_s3.apply(ig_md);

            // Backup metadata
            hdr.bridge_md.setValid();
            hdr.bridge_md.flow_id = ig_md.flow_id;
            hdr.bridge_md.vote_tot_div = ig_md.vote_tot_div;
            hdr.bridge_md.running = (bit<8>)(bit<1>) ig_md.running;
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
        transition select(hdr.ipv4.protocol) {
            IP_PROTOCOLS_TCP : accept;
            default : reject;
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
    }
}

control SwitchEgress(
        inout header_t hdr,
        inout metadata_t eg_md,
        in egress_intrinsic_metadata_t eg_intr_md,
        in egress_intrinsic_metadata_from_parser_t eg_intr_md_from_prsr,
        inout egress_intrinsic_metadata_for_deparser_t eg_intr_dprs_md,
        inout egress_intrinsic_metadata_for_output_port_t eg_intr_oport_md) {
    ElasticSketchHeavyPart() heavy_part_s4;
    
    apply {
        /*
        if (hdr.ipv4.isValid()) {
            // Restore metadata
            eg_md.flow_id = hdr.bridge_md.flow_id;
            eg_md.vote_tot_div = hdr.bridge_md.vote_tot_div;
            if (hdr.bridge_md.running != 0) {
                eg_md.running = true;
            } else {
                eg_md.running = false;
            }
            // eg_md.running = hdr.bridge_md.running == 0 ? false : true;
            hdr.bridge_md.setInvalid();
            
            // Heavy parts (4)
            if (eg_md.running)
                heavy_part_s4.apply(eg_md);

            // TODO: light part
        }
        */
    }
}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;
