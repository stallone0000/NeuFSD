#include <cassert>
#include <iostream>

#include "neufsd_dataplane.hpp"
#include "sample_dataplane.hpp"

int main() {
    using namespace mrac_dataplane;

    HashFlowSampler hash_sampler(128, 1.0);
    ArrayFlowSampler array_sampler(128, 1.0);
    NeuFSDDataPlane neufsd(64, 4, 256, 8);

    for (std::uint64_t i = 0; i < 1000; ++i) {
        const std::uint64_t flow_id = i % 25;
        hash_sampler.insert(flow_id);
        array_sampler.insert(flow_id);
        neufsd.insert(flow_id);
    }

    const auto hash_fsd = hash_sampler.estimate_fsd();
    const auto array_fsd = array_sampler.estimate_fsd();
    const auto snap = neufsd.snapshot();

    assert(!hash_fsd.fsd.empty());
    assert(!array_fsd.fsd.empty());
    assert(snap.total_packets == 1000);
    assert(snap.hot_packet_mass <= snap.total_packets);

    std::cout << "dataplane smoke ok\n";
    return 0;
}
