// Engine: parse body bytes + IVF kNN → fraud_count in [0..5].
// Adaptive two-pass: cheap nprobe first, only re-runs full nprobe when
// the count is on the wrong side of the approval threshold.
#pragma once

#include "ivf_index.hpp"
#include "parser.hpp"
#include <atomic>
#include <cstdint>

namespace rinha {

struct Engine {
  ivf::IvfIndex idx;
  int nprobe      = 70;
  int fast_nprobe = 5;

  mutable std::atomic<uint64_t> total{0};
  mutable std::atomic<uint64_t> escalated{0};
  mutable std::atomic<uint64_t> fast_hist[6]{};

  bool load(const char *path) { return ivf::load_index(path, idx); }
  void unload() { ivf::unload_index(idx); }

  // returns 0..5
  inline int fraud_count(const char *body, size_t len) const {
    float qf[ivf::DIM];
    parse_payload(body, len, qf);
    alignas(32) int16_t q16[ivf::STRIDE];
    ivf::quantize_query(qf, idx.scale, q16);

    total.fetch_add(1, std::memory_order_relaxed);

    if (fast_nprobe >= nprobe) {
      int fc = ivf::ivf_fraud_count(idx, q16, nprobe);
      escalated.fetch_add(1, std::memory_order_relaxed);
      fast_hist[fc < 0 ? 0 : (fc > 5 ? 5 : fc)].fetch_add(1, std::memory_order_relaxed);
      return fc;
    }

    int fc = ivf::ivf_fraud_count(idx, q16, fast_nprobe);
    fast_hist[fc < 0 ? 0 : (fc > 5 ? 5 : fc)].fetch_add(1, std::memory_order_relaxed);
    // Escalate any non-consensus count. Unanimous (0 or TOP_K) is
    // trusted; only those are escalated when the IVF clustering is
    // suspect — see notes in docs/results.md.
    if (fc > 0 && fc < ivf::TOP_K) {
      fc = ivf::ivf_fraud_count(idx, q16, nprobe);
      escalated.fetch_add(1, std::memory_order_relaxed);
    }
    return fc;
  }
};

} // namespace rinha
