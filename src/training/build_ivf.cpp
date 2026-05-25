// build_ivf — turn a centroids checkpoint into a deployable ivf.bin.
//
// Reads:
//   references.json     (3M points + labels, ~285 MB)
//   centroids.f32.bin   (K * DIM float32 centroids; produced by k_means)
// Writes:
//   ivf.bin             (header + quantized centroids + offsets + vectors + labels)
//
// One full ASSIGN pass with the current centroids partitions the
// vectors, then we apply the same int16 quantization the runtime
// expects (scale chosen from global |max| × 16383).

#include "../engine/ivf_index.hpp"
#include "json.hpp"

#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

static_assert(std::endian::native == std::endian::little,
              "this code assumes little-endian");

using std::cerr;
using std::cout;
using json = nlohmann::json;
using ivf::dist_sq;

constexpr uint32_t DIM = 14;
constexpr uint32_t STRIDE = 16;
constexpr const char *CHECKPOINT_PATH = "centroids.f32.bin";

static bool load_checkpoint(std::vector<std::vector<float>> &centroids, uint32_t &K) {
  std::ifstream in(CHECKPOINT_PATH, std::ios::binary);
  if (!in) { cerr << "cannot open " << CHECKPOINT_PATH << "\n"; return false; }
  uint32_t k, d, ci, rsv;
  in.read(reinterpret_cast<char *>(&k),  4);
  in.read(reinterpret_cast<char *>(&d),  4);
  in.read(reinterpret_cast<char *>(&ci), 4);
  in.read(reinterpret_cast<char *>(&rsv), 4);
  if (d != DIM) { cerr << "checkpoint DIM=" << d << " != " << DIM << "\n"; return false; }
  K = k;
  centroids.assign(k, std::vector<float>(DIM, 0.0f));
  for (uint32_t i = 0; i < k; ++i)
    in.read(reinterpret_cast<char *>(centroids[i].data()), DIM * sizeof(float));
  cerr << "loaded centroids checkpoint: K=" << k << " completed_iter=" << ci << "\n";
  return true;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0] << " <references.json> [out=ivf.bin]\n";
    return 1;
  }
  const char *out_path = (argc >= 3) ? argv[2] : "ivf.bin";

  std::vector<std::vector<float>> centroids;
  uint32_t K = 0;
  if (!load_checkpoint(centroids, K)) return 1;

  cerr << "reading " << argv[1] << "...\n";
  std::ifstream f(argv[1]);
  json data = json::parse(f);
  std::vector<std::vector<float>> points;
  std::vector<uint8_t> labels;
  points.reserve(data.size());
  labels.reserve(data.size());
  for (auto &item : data) {
    points.push_back(item["vector"]);
    labels.push_back(item["label"] == "fraud" ? 1 : 0);
  }
  uint32_t N = static_cast<uint32_t>(points.size());
  cerr << "N=" << N << "\n";

  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  // ASSIGN pass
  std::vector<uint32_t> assignments(N, 0);
  std::vector<int> count(K, 0);
  for (uint32_t i = 0; i < N; ++i) {
    float best = std::numeric_limits<float>::max();
    uint32_t best_j = 0;
    const float *xi = points[i].data();
    for (uint32_t j = 0; j < K; ++j) {
      float d = dist_sq(xi, centroids[j].data());
      if (d < best) { best = d; best_j = j; }
    }
    assignments[i] = best_j;
    count[best_j]++;
  }
  auto t_assign = std::chrono::duration_cast<std::chrono::seconds>(clock::now() - t0).count();
  cerr << "ASSIGN done in " << t_assign << "s\n";

  // Global scale for quantization.
  float global_abs_max = 0.0f;
  for (auto &p : points)
    for (uint32_t d = 0; d < DIM; ++d)
      global_abs_max = std::max(global_abs_max, std::abs(p[d]));
  for (auto &c : centroids)
    for (uint32_t d = 0; d < DIM; ++d)
      global_abs_max = std::max(global_abs_max, std::abs(c[d]));
  const float scale = (global_abs_max > 0.0f) ? (16383.0f / global_abs_max) : 1.0f;
  cerr << "scale=" << scale << " (abs_max=" << global_abs_max << ")\n";

  auto qz = [scale](float v) -> int16_t {
    float q = std::round(v * scale);
    if (q >  16383.0f) q =  16383.0f;
    if (q < -16383.0f) q = -16383.0f;
    return static_cast<int16_t>(q);
  };

  std::ofstream out(out_path, std::ios::binary);
  if (!out) { cerr << "cannot open " << out_path << "\n"; return 1; }

  out.write(reinterpret_cast<const char *>(&K),     sizeof(K));
  uint32_t DIM_u = DIM;
  out.write(reinterpret_cast<const char *>(&DIM_u), sizeof(DIM_u));
  out.write(reinterpret_cast<const char *>(&N),     sizeof(N));
  out.write(reinterpret_cast<const char *>(&scale), sizeof(scale));

  // CENTROIDS (K × STRIDE × int16)
  {
    std::vector<int16_t> buf(STRIDE, 0);
    for (auto &c : centroids) {
      for (uint32_t d = 0; d < DIM; ++d) buf[d] = qz(c[d]);
      out.write(reinterpret_cast<const char *>(buf.data()),
                STRIDE * sizeof(int16_t));
    }
  }

  // OFFSETS
  std::vector<uint32_t> offsets(K + 1, 0);
  for (uint32_t j = 0; j < K; ++j) offsets[j + 1] = offsets[j] + count[j];
  out.write(reinterpret_cast<const char *>(offsets.data()),
            offsets.size() * sizeof(uint32_t));

  // VECTORS partitioned by cluster
  auto cursor = offsets;
  std::vector<int16_t> ordered(static_cast<size_t>(N) * STRIDE, 0);
  std::vector<uint8_t> ord_labels(N);
  for (uint32_t i = 0; i < N; ++i) {
    uint32_t c = assignments[i];
    uint32_t pos = cursor[c]++;
    for (uint32_t d = 0; d < DIM; ++d)
      ordered[static_cast<size_t>(pos) * STRIDE + d] = qz(points[i][d]);
    ord_labels[pos] = labels[i];
  }
  out.write(reinterpret_cast<const char *>(ordered.data()),
            ordered.size() * sizeof(int16_t));
  out.write(reinterpret_cast<const char *>(ord_labels.data()),
            ord_labels.size());
  out.close();

  cerr << "wrote " << out_path << ": K=" << K << " D=" << DIM
       << " N=" << N << " scale=" << scale << "\n";
  return 0;
}
