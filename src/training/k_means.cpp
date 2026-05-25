#include "../engine/ivf_index.hpp"  // ivf::dist_sq (SIMD when available)
#include "json.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

static_assert(std::endian::native == std::endian::little,
              "this code assumes little-endian");

using namespace std;
using json = nlohmann::json;
using ivf::dist_sq;

std::mt19937 rng(42); // seed

// K is overridable via CLI for retraining experiments (e.g. K=4096).
// Default 1700 matches the originally shipped ivf.bin.
uint32_t K = 1700;
// T = 200 is a hard cap — well above any plausible convergence horizon.
// We rely on EPS_MOVE for the actual stopping condition. The previous
// training (T=20, EPS_MOVE=1e-4) had no logs, so we have no record of
// whether it actually converged or just timed out at 20 epochs.
constexpr uint32_t T = 200;
constexpr uint32_t DIM = 14;
// Tighter convergence: per-centroid avg movement well below the int16
// quantization step (1/16383 ≈ 6e-5) means quantizing more won't change
// the resulting ivf.bin meaningfully.
constexpr float    EPS_MOVE = 1e-6f;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "uso: " << argv[0] << " <references.json> [K]\n";
    return 1;
  }

  std::string path = argv[1];
  if (argc >= 3) K = static_cast<uint32_t>(std::atoi(argv[2]));
  std::ifstream f(path);
  json data = json::parse(f);
  std::vector<std::vector<float>> points;
  std::vector<uint8_t> labels;
  points.reserve(data.size());
  labels.reserve(data.size());
  for (auto &item : data) {
    points.push_back(item["vector"]);
    labels.push_back(item["label"] == "fraud" ? 1 : 0);
  }

  std::vector<size_t> assignments(data.size());
  std::vector<std::vector<float>> centroids(K);
  std::sample(points.begin(), points.end(), centroids.begin(), K, rng);
  std::vector<std::vector<float>> prev_centroids(K, std::vector<float>(DIM, 0.0f));
  std::vector<std::vector<float>> sum(K, std::vector<float>(DIM, 0.0f));
  std::vector<int> count(K, 0);

  using clock = std::chrono::steady_clock;
  auto t_start = clock::now();

  cout << "training k-means: K=" << K << " T=" << T << " DIM=" << DIM
       << " N=" << data.size() << "\n";

  for (uint32_t t = 0; t < T; ++t) {
    auto iter_start = clock::now();

    // snapshot centroids before this iteration's update (for convergence check)
    for (uint32_t j = 0; j < K; ++j) prev_centroids[j] = centroids[j];

    // ASSIGN (hot loop: N × K × D ops; dist_sq uses SIMD when available)
    for (size_t i = 0; i < data.size(); ++i) {
      float best_dist = std::numeric_limits<float>::max();
      int best_j = 0;
      const float *xi = points[i].data();

      for (uint32_t j = 0; j < K; ++j) {
        float dist = dist_sq(xi, centroids[j].data());
        if (dist < best_dist) {
          best_dist = dist;
          best_j = j;
        }
      }

      assignments[i] = best_j;
    }

    // UPDATE
    std::fill(count.begin(), count.end(), 0);
    for (auto &s : sum)
      std::fill(s.begin(), s.end(), 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
      int c = assignments[i];
      for (uint32_t d = 0; d < DIM; ++d)
        sum[c][d] += points[i][d];
      count[c]++;
    }

    for (uint32_t j = 0; j < K; ++j) {
      if (count[j] == 0) continue; // orphan centroid
      for (uint32_t d = 0; d < DIM; ++d) {
        centroids[j][d] = sum[j][d] / count[j];
      }
    }

    // total L2 movement of centroids vs previous iteration
    float move = 0.0f;
    for (uint32_t j = 0; j < K; ++j) {
      move += std::sqrt(dist_sq(centroids[j].data(), prev_centroids[j].data()));
    }

    auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now() - iter_start).count();
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(
        clock::now() - t_start).count();
    cout << "  iter " << (t + 1) << "/" << T
         << "  took=" << iter_ms << "ms"
         << "  total=" << total_s << "s"
         << "  move=" << move << "\n" << std::flush;

    if (move < EPS_MOVE) {
      cout << "converged at iter " << (t + 1) << " (move=" << move
           << " < eps=" << EPS_MOVE << ")\n";
      break;
    }
  }

  // ── int16 quantization ─────────────────────────────────────────────
  // Vectors are normalized (most dims in [-1,1] or [0,1]; one in [0,0.05]).
  // Find a single global scale that maps the largest-magnitude dim to the
  // full int16 range. dist² ordering is preserved (linear transform).
  float global_abs_max = 0.0f;
  for (const auto &p : points)
    for (uint32_t d = 0; d < DIM; ++d)
      global_abs_max = std::max(global_abs_max, std::abs(p[d]));
  for (const auto &c : centroids)
    for (uint32_t d = 0; d < DIM; ++d)
      global_abs_max = std::max(global_abs_max, std::abs(c[d]));

  // Use 16383 (= INT16_MAX/2) so that max_diff = 2*16383 = 32766 stays
  // within int16. Otherwise `_mm256_sub_epi16` wraps and `_mm256_madd_epi16`
  // computes garbage on (q_a - q_b) for extreme pairs.
  // Also keeps sum-of-2-squares per madd lane within int32: 2 * 32766² ≈ 2.147e9 < INT32_MAX.
  const float scale = (global_abs_max > 0.0f) ? (16383.0f / global_abs_max) : 1.0f;
  cout << "quant: global_abs_max=" << global_abs_max << " scale=" << scale
       << " (capped at 16383 to keep diff within int16)" << endl;

  auto qz = [scale](float v) -> int16_t {
    float q = std::round(v * scale);
    if (q >  16383.0f) q =  16383.0f;
    if (q < -16383.0f) q = -16383.0f;
    return static_cast<int16_t>(q);
  };

  // Storage stride: pad DIM=14 → 16 so each vector is one 32-byte AVX2 reg.
  // Last two int16 slots are zeroed; they contribute 0 to (a-b)² and
  // disappear from the SIMD dist_sq automatically.
  constexpr uint32_t STRIDE = 16;

  // save file
  ofstream out("ivf.bin", ios::binary);
  if (!out) { cerr << "Error while opening the file for writing"; return 1; }

  uint32_t N = static_cast<uint32_t>(data.size());

  // HEADER (16 bytes: K, D, N, scale)
  out.write(reinterpret_cast<const char *>(&K),      sizeof(K));
  out.write(reinterpret_cast<const char *>(&DIM),    sizeof(DIM));
  out.write(reinterpret_cast<const char *>(&N),      sizeof(N));
  out.write(reinterpret_cast<const char *>(&scale),  sizeof(scale));   // NEW
  // (STRIDE=16 is implicit from DIM=14 → next power of 2 with ≥DIM lanes
  // for AVX2 alignment. Reader hardcodes it.)

  // CENTROIDS (K × STRIDE × int16)
  {
    std::vector<int16_t> buf(STRIDE, 0);
    for (const auto &centroid : centroids) {
      for (uint32_t d = 0; d < DIM; ++d) buf[d] = qz(centroid[d]);
      // STRIDE-DIM tail stays 0
      out.write(reinterpret_cast<const char *>(buf.data()),
                STRIDE * sizeof(int16_t));
    }
  }

  // OFFSETS (prefix-sum dos counts)
  std::vector<uint32_t> offsets(K + 1);
  offsets[0] = 0;
  for (uint32_t j = 0; j < K; ++j) offsets[j + 1] = offsets[j] + count[j];
  out.write(reinterpret_cast<const char *>(offsets.data()),
            offsets.size() * sizeof(uint32_t));

  // VECTORS sorted by cluster (using cursor array), STRIDE int16 each
  auto cursor = offsets;
  std::vector<int16_t> ordered_vectors(N * STRIDE, 0); // zero-init for tail
  std::vector<uint8_t> ordered_labels(N);

  for (uint32_t i = 0; i < N; ++i) {
    uint32_t cluster_index = assignments[i];
    uint32_t pos = cursor[cluster_index]++;
    for (uint32_t d = 0; d < DIM; ++d)
      ordered_vectors[pos * STRIDE + d] = qz(points[i][d]);
    // tail (d = DIM..STRIDE-1) already 0
    ordered_labels[pos] = labels[i];
  }

  out.write(reinterpret_cast<const char *>(ordered_vectors.data()),
            ordered_vectors.size() * sizeof(int16_t));

  // LABELS
  out.write(reinterpret_cast<const char *>(ordered_labels.data()),
            ordered_labels.size() * sizeof(uint8_t));

  out.close();
  cout << "wrote ivf.bin: K=" << K << " D=" << DIM << " STRIDE=" << STRIDE
       << " N=" << N << " scale=" << scale << endl;

  return 0;
}
