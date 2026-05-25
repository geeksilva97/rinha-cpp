// k-means trainer with pause/resume.
//
// After every iter the float-precision centroids are atomically
// written to `centroids.f32.bin`. On startup we look for that file;
// if present, training resumes from those centroids instead of doing
// a random k-init. SIGINT / SIGTERM finish the current iter and exit
// cleanly so we always leave a consistent checkpoint behind.
//
// This binary does NOT write the deployable `ivf.bin` — that step
// (quantize + assign + serialize the whole partitioned dataset) lives
// in `build_ivf`, which reads the same checkpoint and produces the
// final blob in ~30 seconds. Keep training tight; defer the heavy
// I/O to the moment you actually want a snapshot.

#include "../engine/ivf_index.hpp"
#include "json.hpp"

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <csignal>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/stat.h>

static_assert(std::endian::native == std::endian::little,
              "this code assumes little-endian");

using std::cerr;
using std::cout;
using json = nlohmann::json;
using ivf::dist_sq;

namespace {

std::mt19937 rng(42);

uint32_t K = 1700;
constexpr uint32_t T = 200;
constexpr uint32_t DIM = 14;
// Tighter than the int16 quantization step (1/16383 ≈ 6e-5). Below
// this further iters are just shuffling borderline points.
constexpr float    EPS_MOVE = 1e-6f;

constexpr const char *CHECKPOINT_PATH = "centroids.f32.bin";

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

bool file_exists(const char *p) {
  struct stat st;
  return stat(p, &st) == 0;
}

// Checkpoint layout: 16-byte header (K, DIM, completed_iter, reserved)
// followed by K * DIM float32 centroids. Atomic via rename.
bool write_checkpoint(const std::vector<std::vector<float>> &centroids,
                      uint32_t completed_iter) {
  std::string tmp = std::string(CHECKPOINT_PATH) + ".tmp";
  std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
  if (!out) { cerr << "checkpoint: open failed\n"; return false; }
  uint32_t k = static_cast<uint32_t>(centroids.size());
  uint32_t d = DIM;
  uint32_t reserved = 0;
  out.write(reinterpret_cast<const char *>(&k), 4);
  out.write(reinterpret_cast<const char *>(&d), 4);
  out.write(reinterpret_cast<const char *>(&completed_iter), 4);
  out.write(reinterpret_cast<const char *>(&reserved), 4);
  for (auto &c : centroids)
    out.write(reinterpret_cast<const char *>(c.data()), DIM * sizeof(float));
  out.close();
  if (std::rename(tmp.c_str(), CHECKPOINT_PATH) != 0) {
    cerr << "checkpoint: rename failed\n"; return false;
  }
  return true;
}

bool load_checkpoint(std::vector<std::vector<float>> &centroids,
                     uint32_t &completed_iter) {
  std::ifstream in(CHECKPOINT_PATH, std::ios::binary);
  if (!in) return false;
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
  completed_iter = ci;
  return true;
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "usage: " << argv[0] << " <references.json> [K]\n";
    return 1;
  }
  if (argc >= 3) K = static_cast<uint32_t>(std::atoi(argv[2]));

  std::signal(SIGINT,  handle_signal);
  std::signal(SIGTERM, handle_signal);

  cerr << "reading " << argv[1] << "...\n";
  std::ifstream f(argv[1]);
  json data = json::parse(f);
  std::vector<std::vector<float>> points;
  points.reserve(data.size());
  for (auto &item : data) points.push_back(item["vector"]);
  cerr << "N=" << points.size() << "\n";

  std::vector<size_t> assignments(points.size());
  std::vector<std::vector<float>> centroids;
  uint32_t start_iter = 0;

  if (file_exists(CHECKPOINT_PATH)) {
    if (load_checkpoint(centroids, start_iter)) {
      cerr << "resuming from " << CHECKPOINT_PATH
           << " (K=" << K << ", completed_iter=" << start_iter << ")\n";
    } else {
      cerr << "checkpoint exists but failed to load — aborting to avoid\n"
              "silently re-initializing. Move or delete it.\n";
      return 1;
    }
  } else {
    centroids.assign(K, std::vector<float>(DIM, 0.0f));
    std::sample(points.begin(), points.end(), centroids.begin(), K, rng);
    cerr << "fresh init via std::sample (K=" << K << ")\n";
  }

  std::vector<std::vector<float>> prev(K, std::vector<float>(DIM, 0.0f));
  std::vector<std::vector<float>> sum(K, std::vector<float>(DIM, 0.0f));
  std::vector<int> count(K, 0);

  using clock = std::chrono::steady_clock;
  auto t_start = clock::now();
  cout << "training k-means: K=" << K << " T=" << T << " DIM=" << DIM
       << " N=" << points.size() << " resume_from=" << start_iter
       << " eps=" << EPS_MOVE << "\n";

  for (uint32_t t = start_iter; t < T; ++t) {
    if (g_stop.load(std::memory_order_relaxed)) {
      cout << "stop signal received at start of iter " << (t+1) << "\n";
      break;
    }
    auto iter_start = clock::now();
    for (uint32_t j = 0; j < K; ++j) prev[j] = centroids[j];

    // ASSIGN
    for (size_t i = 0; i < points.size(); ++i) {
      float best_dist = std::numeric_limits<float>::max();
      int best_j = 0;
      const float *xi = points[i].data();
      for (uint32_t j = 0; j < K; ++j) {
        float d = dist_sq(xi, centroids[j].data());
        if (d < best_dist) { best_dist = d; best_j = j; }
      }
      assignments[i] = best_j;
    }

    // UPDATE
    std::fill(count.begin(), count.end(), 0);
    for (auto &s : sum) std::fill(s.begin(), s.end(), 0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
      int c = assignments[i];
      for (uint32_t d = 0; d < DIM; ++d) sum[c][d] += points[i][d];
      count[c]++;
    }
    for (uint32_t j = 0; j < K; ++j) {
      if (count[j] == 0) continue;
      for (uint32_t d = 0; d < DIM; ++d) centroids[j][d] = sum[j][d] / count[j];
    }

    // movement
    float move = 0.0f;
    for (uint32_t j = 0; j < K; ++j)
      move += std::sqrt(dist_sq(centroids[j].data(), prev[j].data()));

    auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        clock::now() - iter_start).count();
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(
        clock::now() - t_start).count();
    cout << "  iter " << (t + 1) << "/" << T
         << "  took=" << iter_ms << "ms"
         << "  total=" << total_s << "s"
         << "  move=" << move << "\n" << std::flush;

    if (!write_checkpoint(centroids, t + 1))
      cerr << "  (warning: checkpoint write failed)\n";

    if (move < EPS_MOVE) {
      cout << "converged at iter " << (t + 1)
           << " (move=" << move << " < eps=" << EPS_MOVE << ")\n";
      break;
    }
  }

  cout << "done. checkpoint at " << CHECKPOINT_PATH
       << ". run `build_ivf <references.json>` to produce ivf.bin.\n";
  return 0;
}
