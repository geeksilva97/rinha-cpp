// IVF index, int16-quantized. mmap'd read-only — page cache shared
// across worker processes that open the same file.
//
// File layout (see docs/IVF-LAYOUT.md):
//   uint32  K, D, N
//   float   scale
//   int16   centroids[K * STRIDE]
//   uint32  offsets[K + 1]
//   int16   vectors[N * STRIDE]
//   uint8   labels[N]
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__AVX2__)
#  include <immintrin.h>
#  define IVF_HAS_AVX2 1
#else
#  define IVF_HAS_AVX2 0
#endif

namespace ivf {

constexpr uint32_t DIM    = 14;
constexpr uint32_t STRIDE = 16;   // 14 padded to 16 int16 = one AVX2 reg
constexpr int      TOP_K  = 5;

struct IvfIndex {
  void *base = nullptr;
  size_t size = 0;
  int fd = -1;
  uint32_t K = 0, D = 0, N = 0;
  float scale = 1.0f;
  const int16_t  *centroids = nullptr;
  const uint32_t *offsets   = nullptr;
  const int16_t  *vectors   = nullptr;
  const uint8_t  *labels    = nullptr;
};

static inline int64_t dist_sq(const int16_t *a, const int16_t *b) {
#if IVF_HAS_AVX2
  __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
  __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b));
  __m256i diff = _mm256_sub_epi16(va, vb);
  __m256i sq   = _mm256_madd_epi16(diff, diff);
  __m256i sq_lo64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(sq));
  __m256i sq_hi64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(sq, 1));
  __m256i sum64   = _mm256_add_epi64(sq_lo64, sq_hi64);
  __m128i lo = _mm256_castsi256_si128(sum64);
  __m128i hi = _mm256_extracti128_si256(sum64, 1);
  __m128i s2 = _mm_add_epi64(lo, hi);
  return _mm_extract_epi64(s2, 0) + _mm_extract_epi64(s2, 1);
#else
  int64_t d = 0;
  for (uint32_t i = 0; i < DIM; ++i) {
    int32_t x = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
    d += static_cast<int64_t>(x) * static_cast<int64_t>(x);
  }
  return d;
#endif
}

static inline void quantize_query(const float *q, float scale, int16_t *out) {
  for (uint32_t d = 0; d < DIM; ++d) {
    float v = std::round(q[d] * scale);
    if (v >  16383.0f) v =  16383.0f;
    if (v < -16383.0f) v = -16383.0f;
    out[d] = static_cast<int16_t>(v);
  }
  for (uint32_t d = DIM; d < STRIDE; ++d) out[d] = 0;
}

inline bool load_index(const char *path, IvfIndex &idx) {
  idx.fd = open(path, O_RDONLY);
  if (idx.fd == -1) return false;
  struct stat st;
  if (fstat(idx.fd, &st) == -1) { close(idx.fd); return false; }
  idx.size = st.st_size;
  idx.base = mmap(nullptr, idx.size, PROT_READ, MAP_PRIVATE, idx.fd, 0);
  if (idx.base == MAP_FAILED) { close(idx.fd); return false; }
  madvise(idx.base, idx.size, MADV_RANDOM);
  // Prefault the entire index into the page cache to eliminate
  // cold-page tail latency during the first thousand-or-so requests.
  // ~99 MB sequential read is fast on any disk and we have the RAM.
  {
    volatile const uint8_t *p = static_cast<const uint8_t *>(idx.base);
    uint64_t sink = 0;
    for (size_t off = 0; off < idx.size; off += 4096) sink += p[off];
    asm volatile("" :: "r"(sink));
  }

  const char *p = static_cast<const char *>(idx.base);
  const uint32_t *u = reinterpret_cast<const uint32_t *>(p);
  idx.K = u[0]; idx.D = u[1]; idx.N = u[2];
  idx.scale = *reinterpret_cast<const float *>(p + 12);
  if (idx.D != DIM) { munmap(idx.base, idx.size); close(idx.fd); return false; }

  const size_t cent_bytes = static_cast<size_t>(idx.K) * STRIDE * sizeof(int16_t);
  const size_t off_bytes  = (static_cast<size_t>(idx.K) + 1) * sizeof(uint32_t);
  const size_t vec_bytes  = static_cast<size_t>(idx.N) * STRIDE * sizeof(int16_t);
  idx.centroids = reinterpret_cast<const int16_t *>(p + 16);
  idx.offsets   = reinterpret_cast<const uint32_t *>(p + 16 + cent_bytes);
  idx.vectors   = reinterpret_cast<const int16_t *>(p + 16 + cent_bytes + off_bytes);
  idx.labels    = reinterpret_cast<const uint8_t *>(p + 16 + cent_bytes + off_bytes + vec_bytes);

  std::cerr << "ivf.mmap K=" << idx.K << " N=" << idx.N
            << " scale=" << idx.scale << " size=" << idx.size << "\n";
  return true;
}

inline void unload_index(IvfIndex &idx) {
  if (idx.base && idx.base != MAP_FAILED) munmap(idx.base, idx.size);
  if (idx.fd >= 0) close(idx.fd);
  idx = {};
}

inline uint32_t nearest_centroid(const IvfIndex &idx, const int16_t *q) {
  int64_t best = std::numeric_limits<int64_t>::max();
  uint32_t best_c = 0;
  for (uint32_t i = 0; i < idx.K; ++i) {
    int64_t d = dist_sq(q, idx.centroids + i * STRIDE);
    if (d < best) { best = d; best_c = i; }
  }
  return best_c;
}

inline void top_n_centroids(const IvfIndex &idx, const int16_t *q,
                            int nprobe, uint32_t *out) {
  int64_t *top_d = static_cast<int64_t *>(alloca(nprobe * sizeof(int64_t)));
  for (int k = 0; k < nprobe; ++k) {
    top_d[k] = std::numeric_limits<int64_t>::max();
    out[k] = 0;
  }
  for (uint32_t i = 0; i < idx.K; ++i) {
    int64_t d = dist_sq(q, idx.centroids + i * STRIDE);
    if (d >= top_d[nprobe - 1]) continue;
    int p = nprobe - 1;
    while (p > 0 && top_d[p - 1] > d) {
      top_d[p] = top_d[p - 1]; out[p] = out[p - 1]; --p;
    }
    top_d[p] = d; out[p] = i;
  }
}

inline void merge_top5_from_cluster(const IvfIndex &idx, const int16_t *q,
                                    uint32_t cluster,
                                    int64_t top_d[TOP_K],
                                    uint32_t top_i[TOP_K]) {
  uint32_t begin = idx.offsets[cluster];
  uint32_t end   = idx.offsets[cluster + 1];
  for (uint32_t i = begin; i < end; ++i) {
    int64_t d = dist_sq(q, idx.vectors + i * STRIDE);
    if (d >= top_d[TOP_K - 1]) continue;
    int p = TOP_K - 1;
    while (p > 0 && top_d[p - 1] > d) {
      top_d[p] = top_d[p - 1]; top_i[p] = top_i[p - 1]; --p;
    }
    top_d[p] = d; top_i[p] = i;
  }
}

inline int ivf_fraud_count(const IvfIndex &idx, const int16_t *q, int nprobe) {
  int64_t  top_d[TOP_K];
  uint32_t top_i[TOP_K];
  for (int k = 0; k < TOP_K; ++k) {
    top_d[k] = std::numeric_limits<int64_t>::max();
    top_i[k] = 0;
  }
  if (nprobe <= 1) {
    uint32_t c = nearest_centroid(idx, q);
    merge_top5_from_cluster(idx, q, c, top_d, top_i);
  } else {
    uint32_t *clusters = static_cast<uint32_t *>(alloca(nprobe * sizeof(uint32_t)));
    top_n_centroids(idx, q, nprobe, clusters);
    for (int n = 0; n < nprobe; ++n)
      merge_top5_from_cluster(idx, q, clusters[n], top_d, top_i);
  }
  int frauds = 0;
  for (int k = 0; k < TOP_K; ++k) frauds += idx.labels[top_i[k]];
  return frauds;
}

} // namespace ivf
