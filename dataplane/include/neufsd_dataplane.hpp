#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace mrac_dataplane {

struct NeuFSDHotEntry {
    std::uint64_t flow_id = 0;
    std::uint32_t count = 0;
    bool used = false;
};

struct NeuFSDBucket {
    std::uint32_t vote = 0;
    std::vector<NeuFSDHotEntry> slots;
};

struct NeuFSDSnapshot {
    std::vector<NeuFSDBucket> hot_buckets;
    std::vector<std::uint32_t> cold_array;
    std::uint64_t total_packets = 0;
    std::uint64_t hot_packet_mass = 0;
};

// Data-plane part of NeuFSD: Hot Filter plus a compact counter array.
// The neural decoder and residual-mass calibration are control-plane steps.
class NeuFSDDataPlane {
public:
    NeuFSDDataPlane(std::size_t hot_bucket_count = 8192,
                    std::size_t slots_per_bucket = 4,
                    std::size_t cold_array_size = 4096,
                    std::uint32_t lambda = 8,
                    std::uint64_t seed = 0x6a09e667f3bcc909ULL)
        : hot_buckets_(hot_bucket_count),
          cold_array_(cold_array_size, 0),
          lambda_(std::max<std::uint32_t>(1, lambda)),
          seed_(seed) {
        for (auto& bucket : hot_buckets_) {
            bucket.slots.resize(slots_per_bucket);
        }
    }

    void insert(std::uint64_t flow_id) {
        total_packets_ += 1;
        if (hot_buckets_.empty()) {
            update_cold(flow_id, 1);
            return;
        }

        NeuFSDBucket& bucket = hot_buckets_[hot_index(flow_id)];
        for (auto& slot : bucket.slots) {
            if (slot.used && slot.flow_id == flow_id) {
                slot.count += 1;
                hot_packet_mass_ += 1;
                return;
            }
        }

        for (auto& slot : bucket.slots) {
            if (!slot.used) {
                slot.used = true;
                slot.flow_id = flow_id;
                slot.count = 1;
                hot_packet_mass_ += 1;
                return;
            }
        }

        auto min_it = std::min_element(
            bucket.slots.begin(), bucket.slots.end(),
            [](const NeuFSDHotEntry& a, const NeuFSDHotEntry& b) { return a.count < b.count; });
        if (min_it == bucket.slots.end()) {
            update_cold(flow_id, 1);
            return;
        }

        bucket.vote += 1;
        const std::uint64_t threshold = static_cast<std::uint64_t>(lambda_) * std::max<std::uint32_t>(1, min_it->count);
        if (bucket.vote >= threshold) {
            update_cold(min_it->flow_id, min_it->count);
            hot_packet_mass_ -= min_it->count;
            min_it->flow_id = flow_id;
            min_it->count = 1;
            min_it->used = true;
            bucket.vote = 0;
            hot_packet_mass_ += 1;
        } else {
            update_cold(flow_id, 1);
        }
    }

    NeuFSDSnapshot snapshot() const {
        NeuFSDSnapshot out;
        out.hot_buckets = hot_buckets_;
        out.cold_array = cold_array_;
        out.total_packets = total_packets_;
        out.hot_packet_mass = hot_packet_mass_;
        return out;
    }

    std::vector<double> hot_fsd(std::uint32_t phi) const {
        std::uint32_t max_count = 0;
        for (const auto& bucket : hot_buckets_) {
            for (const auto& slot : bucket.slots) {
                if (slot.used && slot.count >= phi) {
                    max_count = std::max(max_count, slot.count);
                }
            }
        }
        std::vector<double> fsd(static_cast<std::size_t>(max_count) + 1, 0.0);
        for (const auto& bucket : hot_buckets_) {
            for (const auto& slot : bucket.slots) {
                if (slot.used && slot.count >= phi) {
                    fsd[slot.count] += 1.0;
                }
            }
        }
        return fsd;
    }

    const std::vector<std::uint32_t>& cold_array() const { return cold_array_; }
    std::uint64_t total_packets() const { return total_packets_; }
    std::uint64_t hot_packet_mass() const { return hot_packet_mass_; }
    std::uint64_t cold_packet_mass() const { return total_packets_ - hot_packet_mass_; }

private:
    static std::uint64_t mix(std::uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return x;
    }

    std::size_t hot_index(std::uint64_t flow_id) const {
        return static_cast<std::size_t>(mix(flow_id ^ seed_) % hot_buckets_.size());
    }

    std::size_t cold_index(std::uint64_t flow_id) const {
        if (cold_array_.empty()) {
            return 0;
        }
        return static_cast<std::size_t>(mix(flow_id ^ (seed_ + 0x9e3779b97f4a7c15ULL)) % cold_array_.size());
    }

    void update_cold(std::uint64_t flow_id, std::uint32_t delta) {
        if (cold_array_.empty()) {
            return;
        }
        std::uint32_t& counter = cold_array_[cold_index(flow_id)];
        const std::uint32_t room = std::numeric_limits<std::uint32_t>::max() - counter;
        counter += std::min(delta, room);
    }

    std::vector<NeuFSDBucket> hot_buckets_;
    std::vector<std::uint32_t> cold_array_;
    std::uint32_t lambda_;
    std::uint64_t seed_;
    std::uint64_t total_packets_ = 0;
    std::uint64_t hot_packet_mass_ = 0;
};

}  // namespace mrac_dataplane
