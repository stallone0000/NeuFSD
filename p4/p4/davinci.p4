#include <core.p4>
#include <tna.p4>

#include "common/headers.p4"
#include "common/utils.p4"
#include "common/params.p4"

// Data-plane structures: a hot-flow part, three Tower counter layers,
// and three Fermat overflow arrays.

#define DAVINCI_HOT_BUCKET_NUM 8192
#define DAVINCI_HOT_BUCKET_INDEX_WIDTH 13
#define DAVINCI_TOWER2_BUCKET_NUM 65536
#define DAVINCI_TOWER2_INDEX_WIDTH 16
#define DAVINCI_TOWER4_BUCKET_NUM 32768
#define DAVINCI_TOWER4_INDEX_WIDTH 15
#define DAVINCI_TOWER8_BUCKET_NUM 16384
#define DAVINCI_TOWER8_INDEX_WIDTH 14
#define DAVINCI_FERMAT_BUCKET_NUM 16384
#define DAVINCI_FERMAT_INDEX_WIDTH 14
#define DAVINCI_LAMBDA_WIDTH 5

typedef bit<32> flow_id_t;

struct metadata_t {
    bit<32>   tot_votes;
    bit<32>   tot_votes_div;
    flow_id_t flow_id;
    bit<32>   flow_freq;
    flow_id_t register_id;
    bit<1>    hot_flag;
    bit<1>    cold_flag;
}

header bridge_md_h {
    flow_id_t flow_id;
    bit<32>   flow_freq;
    bit<1>    hot_flag;
    bit<1>    cold_flag;
    bit<6>    _padding;
}

struct header_t {
    bridge_md_h bridge_md;
    ethernet_h  ethernet;
    ipv4_h      ipv4;
}

struct hot_bucket_p1_t {
    bit<32> tot_votes;
}

struct hot_bucket_p2_t {
    flow_id_t flow_id;
    bit<32>   flow_freq;
}

struct fermat_id_t {
    flow_id_t xor_id;
}

struct fermat_count_t {
    bit<16> count;
}

control DaVinciHotPart(inout metadata_t md) {
    Hash<bit<DAVINCI_HOT_BUCKET_INDEX_WIDTH>>(HashAlgorithm_t.IDENTITY) flow_id_hash;

    Register<hot_bucket_p1_t, _>(DAVINCI_HOT_BUCKET_NUM, {0}) hot_bucket_p1;
    Register<hot_bucket_p2_t, _>(DAVINCI_HOT_BUCKET_NUM, {0, 0}) hot_bucket_p2;
    RegisterAction<_, _, bit<32>>(hot_bucket_p1) hot_bucket_p1_update = {
        void apply(inout hot_bucket_p1_t hbp1, out bit<32> tot_votes) {
            hbp1.tot_votes = hbp1.tot_votes + 1;
            tot_votes = hbp1.tot_votes;
        }
    };
    RegisterAction<_, _, flow_id_t>(hot_bucket_p2) hot_bucket_p2_update = {
        void apply(inout hot_bucket_p2_t hbp2, out flow_id_t evicted_flow_id) {
            evicted_flow_id = 0;
            if (md.flow_id == hbp2.flow_id) {
                hbp2.flow_freq = hbp2.flow_freq + 1;
                md.hot_flag = 0;
                md.cold_flag = 0;
            } else if (md.tot_votes_div >= hbp2.flow_freq) {
                evicted_flow_id = hbp2.flow_id;
                md.flow_freq = hbp2.flow_freq;
                hbp2.flow_id = md.flow_id;
                hbp2.flow_freq = 1;
            }
        }
    };

    apply {
        bit<DAVINCI_HOT_BUCKET_INDEX_WIDTH> idx = flow_id_hash.get(md.flow_id);
        md.tot_votes = hot_bucket_p1_update.execute(idx);
        md.tot_votes_div = md.tot_votes >> DAVINCI_LAMBDA_WIDTH;
        md.register_id = hot_bucket_p2_update.execute(idx);
        if (md.register_id != 0) {
            md.flow_id = md.register_id;
            md.cold_flag = 1;
        }
    }
}

control Tower2Layer(inout metadata_t md) {
    Hash<bit<DAVINCI_TOWER2_INDEX_WIDTH>>(HashAlgorithm_t.CRC32) hasher;
    Register<bit<8>, _>(DAVINCI_TOWER2_BUCKET_NUM, 0) counter;
    RegisterAction<_, _, bit<8>>(counter) update = {
        void apply(inout bit<8> val) {
            bit<32> sum = (bit<32>)val + md.flow_freq;
            if (sum > 3) {
                val = 3;
            } else {
                val = (bit<8>)sum;
            }
        }
    };
    apply {
        bit<DAVINCI_TOWER2_INDEX_WIDTH> idx = hasher.get({md.flow_id, 8w2});
        update.execute(idx);
    }
}

control Tower4Layer(inout metadata_t md) {
    Hash<bit<DAVINCI_TOWER4_INDEX_WIDTH>>(HashAlgorithm_t.CRC32) hasher;
    Register<bit<8>, _>(DAVINCI_TOWER4_BUCKET_NUM, 0) counter;
    RegisterAction<_, _, bit<8>>(counter) update = {
        void apply(inout bit<8> val) {
            bit<32> sum = (bit<32>)val + md.flow_freq;
            if (sum > 15) {
                val = 15;
            } else {
                val = (bit<8>)sum;
            }
        }
    };
    apply {
        bit<DAVINCI_TOWER4_INDEX_WIDTH> idx = hasher.get({md.flow_id, 8w4});
        update.execute(idx);
    }
}

control Tower8Layer(inout metadata_t md) {
    Hash<bit<DAVINCI_TOWER8_INDEX_WIDTH>>(HashAlgorithm_t.CRC32) hasher;
    Register<bit<8>, _>(DAVINCI_TOWER8_BUCKET_NUM, 0) counter;
    RegisterAction<_, _, bit<8>>(counter) update = {
        void apply(inout bit<8> val) {
            bit<32> sum = (bit<32>)val + md.flow_freq;
            if (sum > 255) {
                val = 255;
            } else {
                val = (bit<8>)sum;
            }
        }
    };
    apply {
        bit<DAVINCI_TOWER8_INDEX_WIDTH> idx = hasher.get({md.flow_id, 8w8});
        update.execute(idx);
    }
}

control FermatArray(bit<8> salt, inout metadata_t md) {
    Hash<bit<DAVINCI_FERMAT_INDEX_WIDTH>>(HashAlgorithm_t.CRC32) hasher;
    Register<fermat_id_t, _>(DAVINCI_FERMAT_BUCKET_NUM, {0}) id_xor;
    Register<fermat_count_t, _>(DAVINCI_FERMAT_BUCKET_NUM, {0}) counter;
    RegisterAction<_, _, bit<32>>(id_xor) id_update = {
        void apply(inout fermat_id_t val) {
            val.xor_id = val.xor_id ^ md.flow_id;
        }
    };
    RegisterAction<_, _, bit<16>>(counter) count_update = {
        void apply(inout fermat_count_t val) {
            bit<32> sum = (bit<32>)val.count + md.flow_freq;
            if (sum > 65535) {
                val.count = 65535;
            } else {
                val.count = (bit<16>)sum;
            }
        }
    };
    apply {
        bit<DAVINCI_FERMAT_INDEX_WIDTH> idx = hasher.get({md.flow_id, salt});
        id_update.execute(idx);
        count_update.execute(idx);
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
    DaVinciHotPart() hot_part_s1;
    DaVinciHotPart() hot_part_s2;
    DaVinciHotPart() hot_part_s3;
    Tower2Layer() tower2;
    Tower4Layer() tower4;
    FermatArray(8w17) fermat1;

    action forward(PortId_t port) {
        ig_intr_tm_md.ucast_egress_port = port;
    }

    apply {
        if (hdr.ipv4.isValid()) {
            ig_md.hot_flag = 1;
            ig_md.cold_flag = 1;

            hot_part_s1.apply(ig_md);
            if (ig_md.hot_flag == 1) {
                hot_part_s2.apply(ig_md);
            }
            if (ig_md.hot_flag == 1) {
                hot_part_s3.apply(ig_md);
            }
            if (ig_md.cold_flag == 1) {
                tower2.apply(ig_md);
                tower4.apply(ig_md);
                fermat1.apply(ig_md);
            }

            hdr.bridge_md.setValid();
            hdr.bridge_md.flow_id = ig_md.flow_id;
            hdr.bridge_md.flow_freq = ig_md.flow_freq;
            hdr.bridge_md.hot_flag = ig_md.hot_flag;
            hdr.bridge_md.cold_flag = ig_md.cold_flag;

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
        eg_md.hot_flag = hdr.bridge_md.hot_flag;
        eg_md.cold_flag = hdr.bridge_md.cold_flag;
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
    DaVinciHotPart() hot_part_s4;
    Tower8Layer() tower8;
    FermatArray(8w29) fermat2;
    FermatArray(8w43) fermat3;

    apply {
        if (hdr.ipv4.isValid()) {
            hdr.bridge_md.setInvalid();
            if (eg_md.hot_flag == 1) {
                hot_part_s4.apply(eg_md);
            }
            if (eg_md.cold_flag == 1) {
                tower8.apply(eg_md);
                fermat2.apply(eg_md);
                fermat3.apply(eg_md);
            }
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

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         SwitchEgressParser(),
         SwitchEgress(),
         SwitchEgressDeparser()) pipe;

Switch(pipe) main;
