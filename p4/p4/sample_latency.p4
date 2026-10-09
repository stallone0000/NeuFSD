#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/utils.p4"
#include "common/params.p4"

#define SAMPLE_FREQ 1000

#define ELASTIC_HEAVY_BUCKET_NUM 8192
#define ELASTIC_HEAVY_BUCKET_INDEX_WIDTH 13
#define ELASTIC_LIGHT_BUCKET_NUM 131072
#define ELASTIC_LIGHT_BUCKET_INDEX_WIDTH 17
#define ELASTIC_LAMBDA 32
#define ELASTIC_LAMBDA_WIDTH 5

typedef bit<32> flow_id_t;

struct metadata_t {
    bit<32>   tot_votes;
    bit<32>   tot_votes_div; // vote_total / LAMBDA
    flow_id_t flow_id;
    bit<32>   flow_freq;
    flow_id_t register_id;
    bit<1>    heavy_flag;
    bit<1>    light_flag;
}

// Used to pass metadata between ingress and egress
header bridge_md_h {
    flow_id_t flow_id;
    bit<32>   flow_freq;
    bit<32>   ingress_time;
    bit<1>    heavy_flag;
    bit<1>    light_flag;
    bit<6>    _padding;
}

header latency_h {
    bit<32>   latency;
}

struct header_t {
    bridge_md_h bridge_md;
    ethernet_h  ethernet;
    ipv4_h      ipv4;
    latency_h   latency;
}

struct heavy_bucket_p1_t {
    bit<32>   tot_votes;
}

struct heavy_bucket_p2_t {
    flow_id_t flow_id;
    bit<32>   flow_freq;
}

control ElasticSketchHeavyPart(inout metadata_t md) {
    Hash<bit<ELASTIC_HEAVY_BUCKET_INDEX_WIDTH>>(HashAlgorithm_t.IDENTITY) flow_id_hash;

    Register<heavy_bucket_p1_t, _>(ELASTIC_HEAVY_BUCKET_NUM, {0}) heavy_bucket_p1;
    Register<heavy_bucket_p2_t, _>(ELASTIC_HEAVY_BUCKET_NUM, {0, 0}) heavy_bucket_p2;
    RegisterAction<_, _, bit<32>>(heavy_bucket_p1) heavy_bucket_p1_update = {
        void apply(inout heavy_bucket_p1_t hbp1, out bit<32> tot_votes) {
            hbp1.tot_votes = hbp1.tot_votes + 1;
            tot_votes = hbp1.tot_votes;
        }
    };
    RegisterAction<_, _, bit<32>>(heavy_bucket_p2) heavy_bucket_p2_update = {
        void apply(inout heavy_bucket_p2_t hbp2, out flow_id_t evicted_flow_id) {
            evicted_flow_id = 0;
            if (md.tot_votes_div >= hbp2.flow_freq || md.flow_id == hbp2.flow_id) {
                evicted_flow_id = hbp2.flow_id;
                hbp2.flow_freq = hbp2.flow_freq + 1;
                hbp2.flow_id = md.flow_id;
            }
        }
    };

    apply {
        bit<ELASTIC_HEAVY_BUCKET_INDEX_WIDTH> idx = flow_id_hash.get(md.flow_id);
        md.tot_votes = heavy_bucket_p1_update.execute(idx);
        md.register_id = 0;
        
        md.tot_votes_div = md.tot_votes >> ELASTIC_LAMBDA_WIDTH;
        
        md.register_id = heavy_bucket_p2_update.execute(idx);
        
        if (md.register_id != 0) {
            if (md.register_id == md.flow_id) {
                md.heavy_flag = 0;
                md.light_flag = 0;    
            }
            md.flow_id = md.register_id;
            md.flow_freq = md.tot_votes_div;
        }
    }
}

control ElasticSketchLightPart(inout metadata_t md) {
    Hash<bit<ELASTIC_LIGHT_BUCKET_INDEX_WIDTH>>(HashAlgorithm_t.IDENTITY) flow_id_hash;
    Register<bit<32>, _>(ELASTIC_LIGHT_BUCKET_NUM, 0) counter;
    RegisterAction<_, _, bit<32>>(counter) counter_update = {
        void apply(inout bit<32> val) {
            val = val + md.flow_freq;
        }
    };
    
    apply {
        bit<ELASTIC_LIGHT_BUCKET_INDEX_WIDTH> idx = flow_id_hash.get(md.flow_id);
        counter_update.execute(idx);
    }
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
        ig_md.flow_freq = 1;
        transition accept;
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

    Register<bit<32>, _>(1, 0) counter;
    RegisterAction<_, _, bool>(counter) counter_update = {
        void apply(inout bit<32> val, out bool sampled) {
            sampled = false;
            if (val == SAMPLE_FREQ - 1) {
                sampled = true;
                val = 0;
            } else {
                val = val + 1;
            }
        }
    };

    action forward(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port; 
    }

    apply {
        if (hdr.ipv4.isValid()) {
            if (counter_update.execute(0)) {
                // Initialize metadata
                ig_md.heavy_flag = 1;
                ig_md.light_flag = 1;

                // Heavy parts (1-2)
                heavy_part_s1.apply(ig_md);
                if (ig_md.heavy_flag == 1)
                    heavy_part_s2.apply(ig_md);

                // Backup metadata
                hdr.bridge_md.setValid();
                hdr.bridge_md.flow_id = ig_md.flow_id;
                hdr.bridge_md.flow_freq = ig_md.flow_freq;
                hdr.bridge_md.ingress_time = (bit<32>)ig_intr_prsr_md.global_tstamp;
                hdr.bridge_md.heavy_flag = ig_md.heavy_flag;
                hdr.bridge_md.light_flag = ig_md.light_flag;
            }
            forward(DST_PORT);
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
        eg_md.flow_id = hdr.bridge_md.flow_id;
        eg_md.flow_freq = hdr.bridge_md.flow_freq;
        eg_md.heavy_flag = hdr.bridge_md.heavy_flag;
        eg_md.light_flag = hdr.bridge_md.light_flag;
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
    ElasticSketchHeavyPart() heavy_part_s3;
    ElasticSketchHeavyPart() heavy_part_s4;
    
    apply {
        if (hdr.ipv4.isValid()) {
            hdr.latency.setValid();
            bit<32> ingress_time = (bit<32>)hdr.bridge_md.ingress_time;
            bit<32> egress_time = (bit<32>)eg_intr_md_from_prsr.global_tstamp;
            bit<32> latency = egress_time - ingress_time;
            hdr.latency.latency = latency;
            hdr.bridge_md.setInvalid();
            
            // Heavy parts (3-4)
            if (eg_md.heavy_flag == 1)
                heavy_part_s3.apply(eg_md);
            if (eg_md.heavy_flag == 1)
                heavy_part_s4.apply(eg_md);
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
