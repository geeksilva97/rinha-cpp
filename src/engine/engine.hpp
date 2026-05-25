// Engine: parse body bytes + IVF kNN → fraud_count in [0..5].
// Adaptive two-pass: cheap nprobe first, only re-runs full nprobe when
// the count is on the wrong side of the approval threshold.
#pragma once

#include "ivf_index.hpp"
#include "parser.hpp"
#include <cstdint>

namespace rinha {

struct Engine {
  ivf::IvfIndex idx;
  int nprobe      = 70;
  int fast_nprobe = 5;

  bool load(const char *path) { return ivf::load_index(path, idx); }
  void unload() { ivf::unload_index(idx); }

  // returns 0..5
  inline int fraud_count(const char *body, size_t len) const {
    float qf[ivf::DIM];
    parse_payload(body, len, qf);
    alignas(32) int16_t q16[ivf::STRIDE];
    ivf::quantize_query(qf, idx.scale, q16);

    if (fast_nprobe >= nprobe)
      return ivf::ivf_fraud_count(idx, q16, nprobe);

    int fc = ivf::ivf_fraud_count(idx, q16, fast_nprobe);
    // Escalate any non-consensus count (not 0 and not 5) — wider band
    // than the Ruby version's {2,3} to eliminate borderline misses.
    if (fc > 0 && fc < ivf::TOP_K)
      fc = ivf::ivf_fraud_count(idx, q16, nprobe);
    return fc;
  }
};

} // namespace rinha
