#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mrac_dataplane {

struct FlowSampleResult {
    std::vector<double> fsd;
    std::uint64_t sampled_flows = 0;
    std::uint64_t dropped_sampled_flows = 0;
};

class HashFlowSampler {
public:
    HashFlowSampler(std::size_t capacity, double sample_rate, std::uint64_t seed = 0x9e3779b97f4a7c15ULL)
        : capacity_(capacity), threshold_(rate_to_threshold(sample_rate)), sample_rate_(sample_rate), seed_(seed) {
        table_.reserve(capacity);
    }

    void insert(std::uint64_t flow_id) {
        auto it = table_.find(flow_id);
        if (it != table_.end()) {
            it->second += 1;
            return;
        }
        if (!sampled(flow_id)) {
            return;
        }
        if (table_.size() >= capacity_) {
            dropped_sampled_flows_ += 1;
            return;
        }
        table_.emplace(flow_id, 1);
        sampled_flows_ += 1;
    }

    FlowSampleResult estimate_fsd() const {
        std::uint32_t max_freq = 0;
        for (const auto& kv : table_) {
            max_freq = std::max(max_freq, kv.second);
        }
        FlowSampleResult out;
        out.fsd.assign(static_cast<std::size_t>(max_freq) + 1, 0.0);
        for (const auto& kv : table_) {
            out.fsd[kv.second] += 1.0 / sample_rate_;
        }
        out.sampled_flows = sampled_flows_;
        out.dropped_sampled_flows = dropped_sampled_flows_;
        return out;
    }

    std::size_t size() const { return table_.size(); }
    std::size_t capacity() const { return capacity_; }

private:
    static std::uint64_t rate_to_threshold(double rate) {
        if (rate <= 0.0) {
            return 0;
        }
        if (rate >= 1.0) {
            return UINT64_MAX;
        }
        long double scaled = static_cast<long double>(rate) * static_cast<long double>(UINT64_MAX);
        return static_cast<std::uint64_t>(scaled);
    }

    static std::uint64_t mix(std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    bool sampled(std::uint64_t flow_id) const {
        return mix(flow_id ^ seed_) <= threshold_;
    }

    std::size_t capacity_;
    std::uint64_t threshold_;
    double sample_rate_;
    std::uint64_t seed_;
    std::unordered_map<std::uint64_t, std::uint32_t> table_;
    std::uint64_t sampled_flows_ = 0;
    std::uint64_t dropped_sampled_flows_ = 0;
};

class ArrayFlowSampler {
public:
    struct Entry {
        std::uint64_t flow_id = 0;
        std::uint32_t count = 0;
        bool used = false;
    };

    ArrayFlowSampler(std::size_t capacity, double sample_rate, std::uint64_t seed = 0x517cc1b727220a95ULL)
        : entries_(capacity), threshold_(rate_to_threshold(sample_rate)), sample_rate_(sample_rate), seed_(seed) {}

    void insert(std::uint64_t flow_id) {
        if (!sampled(flow_id)) {
            return;
        }

        const std::size_t n = entries_.size();
        if (n == 0) {
            dropped_sampled_flows_ += 1;
            return;
        }

        std::size_t idx = static_cast<std::size_t>(mix(flow_id ^ (seed_ + 0x9e3779b97f4a7c15ULL)) % n);
        for (std::size_t probe = 0; probe < n; ++probe) {
            Entry& entry = entries_[idx];
            if (entry.used && entry.flow_id == flow_id) {
                entry.count += 1;
                return;
            }
            if (!entry.used) {
                entry.used = true;
                entry.flow_id = flow_id;
                entry.count = 1;
                sampled_flows_ += 1;
                return;
            }
            idx = (idx + 1) % n;
        }
        dropped_sampled_flows_ += 1;
    }

    FlowSampleResult estimate_fsd() const {
        std::uint32_t max_freq = 0;
        for (const auto& entry : entries_) {
            if (entry.used) {
                max_freq = std::max(max_freq, entry.count);
            }
        }
        FlowSampleResult out;
        out.fsd.assign(static_cast<std::size_t>(max_freq) + 1, 0.0);
        for (const auto& entry : entries_) {
            if (entry.used) {
                out.fsd[entry.count] += 1.0 / sample_rate_;
            }
        }
        out.sampled_flows = sampled_flows_;
        out.dropped_sampled_flows = dropped_sampled_flows_;
        return out;
    }

    std::size_t capacity() const { return entries_.size(); }

    std::size_t size() const {
        std::size_t occupied = 0;
        for (const auto& entry : entries_) {
            occupied += entry.used ? 1 : 0;
        }
        return occupied;
    }

private:
    static std::uint64_t rate_to_threshold(double rate) {
        if (rate <= 0.0) return 0;
        if (rate >= 1.0) return UINT64_MAX;
        long double scaled = static_cast<long double>(rate) * static_cast<long double>(UINT64_MAX);
        return static_cast<std::uint64_t>(scaled);
    }

    static std::uint64_t mix(std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    bool sampled(std::uint64_t flow_id) const {
        return mix(flow_id ^ seed_) <= threshold_;
    }

    std::vector<Entry> entries_;
    std::uint64_t threshold_;
    double sample_rate_;
    std::uint64_t seed_;
    std::uint64_t sampled_flows_ = 0;
    std::uint64_t dropped_sampled_flows_ = 0;
};

}  // namespace mrac_dataplane
