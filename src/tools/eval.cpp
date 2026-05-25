// Offline eval: runs every entry from k6's test-data.json through the
// engine and prints each disagreement with expected_approved alongside
// the fast-pass and full-pass kNN diagnostics.
//
// Usage:
//   eval <ivf.bin> <test-data.json>
//
// We parse the k6 dataset minimally: each entry has shape
//   { "request": {...}, "expected_approved": bool, ... }
// and we re-serialize the "request" object as a JSON string to feed
// into rinha::parse_payload, exactly as the live server would.

#include "../engine/engine.hpp"
#include "../engine/ivf_index.hpp"
#include "../engine/parser.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ivf;
using namespace rinha;

// Minimal JSON walker: find the next top-level `{...}` block starting
// at or after `pos` in `blob`. Returns {-1, -1} if none.
static std::pair<size_t,size_t> next_object(const std::string &blob, size_t pos) {
  int depth = 0;
  size_t start = 0;
  bool in_str = false;
  bool esc = false;
  for (size_t i = pos; i < blob.size(); ++i) {
    char ch = blob[i];
    if (in_str) {
      if (esc) { esc = false; continue; }
      if (ch == '\\') { esc = true; continue; }
      if (ch == '"')  in_str = false;
      continue;
    }
    if (ch == '"') { in_str = true; continue; }
    if (ch == '{') { if (depth == 0) start = i; depth++; }
    else if (ch == '}') {
      depth--;
      if (depth == 0) return {start, i + 1};
    }
  }
  return {std::string::npos, std::string::npos};
}

// Find the byte-range of value for key `name` inside the object at
// [obj_start, obj_end). Caller scopes already to the immediate parent.
// Returns {-1,-1} if not found.
static std::pair<size_t,size_t> find_value(const std::string &blob,
                                           size_t obj_start, size_t obj_end,
                                           const char *name) {
  size_t nlen = std::strlen(name);
  // search for "name":
  std::string needle = "\"";
  needle += name;
  needle += "\"";
  size_t i = obj_start;
  while (i < obj_end) {
    size_t f = blob.find(needle, i);
    if (f == std::string::npos || f >= obj_end) return {std::string::npos, std::string::npos};
    // ensure the key is at top level of this object — naive: assume
    // unique key names within entries (true for the test dataset).
    // skip past closing quote + optional whitespace + ':'
    size_t k = f + needle.size();
    while (k < obj_end && (blob[k] == ' ' || blob[k] == '\t')) k++;
    if (k >= obj_end || blob[k] != ':') { i = f + 1; continue; }
    k++;
    while (k < obj_end && (blob[k] == ' ' || blob[k] == '\t' ||
                            blob[k] == '\n' || blob[k] == '\r')) k++;
    size_t vstart = k;
    // value: object, string, number, true/false/null
    if (k >= obj_end) return {std::string::npos, std::string::npos};
    if (blob[k] == '{' || blob[k] == '[') {
      char open = blob[k], close = (open == '{' ? '}' : ']');
      int dep = 1; k++;
      while (k < obj_end && dep > 0) {
        if (blob[k] == '"') {
          k++;
          while (k < obj_end && blob[k] != '"') {
            if (blob[k] == '\\') k++;
            k++;
          }
        } else if (blob[k] == open) dep++;
        else if (blob[k] == close) dep--;
        k++;
      }
      return {vstart, k};
    }
    if (blob[k] == '"') {
      k++;
      while (k < obj_end && blob[k] != '"') {
        if (blob[k] == '\\') k++;
        k++;
      }
      k++;
      return {vstart, k};
    }
    // primitive: scan until , } ]
    while (k < obj_end && blob[k] != ',' && blob[k] != '}' && blob[k] != ']') k++;
    return {vstart, k};
  }
  return {std::string::npos, std::string::npos};
}

struct Entry {
  size_t req_start, req_end;
  bool expected_approved;
};

static std::vector<Entry> load_dataset(const std::string &blob) {
  // Top level: {"entries": [...], "stats": {...}}.
  // Find the "entries" array and iterate its objects.
  auto entries_kv = find_value(blob, 0, blob.size(), "entries");
  if (entries_kv.first == std::string::npos) {
    std::cerr << "no `entries` key\n";
    return {};
  }
  std::vector<Entry> out;
  size_t pos = entries_kv.first + 1; // past '['
  while (pos < entries_kv.second) {
    auto obj = next_object(blob, pos);
    if (obj.first == std::string::npos || obj.first >= entries_kv.second) break;
    auto req = find_value(blob, obj.first, obj.second, "request");
    auto ea  = find_value(blob, obj.first, obj.second, "expected_approved");
    if (req.first == std::string::npos || ea.first == std::string::npos) {
      pos = obj.second; continue;
    }
    Entry e;
    e.req_start = req.first;
    e.req_end   = req.second;
    e.expected_approved = (blob[ea.first] == 't');
    out.push_back(e);
    pos = obj.second;
  }
  return out;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: " << argv[0] << " <ivf.bin> <test-data.json>\n";
    return 2;
  }
  Engine eng;
  if (!eng.load(argv[1])) { std::cerr << "load ivf failed\n"; return 1; }
  eng.nprobe = 70;
  eng.fast_nprobe = 5;
  if (const char *e = std::getenv("NPROBE"))      eng.nprobe = std::atoi(e);
  if (const char *e = std::getenv("FAST_NPROBE")) eng.fast_nprobe = std::atoi(e);

  std::ifstream f(argv[2]);
  if (!f) { std::cerr << "cannot open " << argv[2] << "\n"; return 1; }
  std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::cerr << "loaded " << blob.size() << " bytes\n";

  auto entries = load_dataset(blob);
  std::cerr << "entries=" << entries.size() << "\n";

  // For each entry: run fast=N and full=70 separately to diagnose
  // where the disagreement comes from.
  int tp = 0, tn = 0, fp = 0, fn = 0;
  int n_disagree = 0;
  // matrix[fc_fast][fc_full][expected_approved]
  int matrix[6][6][2] = {{{0}}};
  for (auto &e : entries) {
    const char *body = blob.data() + e.req_start;
    size_t len = e.req_end - e.req_start;

    float qf[DIM];
    parse_payload(body, len, qf);
    alignas(32) int16_t q16[STRIDE];
    quantize_query(qf, eng.idx.scale, q16);

    int fc_fast = ivf_fraud_count(eng.idx, q16, eng.fast_nprobe);
    int fc_full = ivf_fraud_count(eng.idx, q16, eng.nprobe);

    // Engine's actual decision (adaptive).
    int fc_adaptive = fc_fast;
    if (eng.fast_nprobe < eng.nprobe && fc_fast > 0 && fc_fast < TOP_K)
      fc_adaptive = fc_full;

    bool approved = fc_adaptive < 3;

    int ff = fc_fast < 0 ? 0 : (fc_fast > 5 ? 5 : fc_fast);
    int fl = fc_full < 0 ? 0 : (fc_full > 5 ? 5 : fc_full);
    matrix[ff][fl][e.expected_approved ? 1 : 0]++;

    bool expected = e.expected_approved;
    if (approved == expected) {
      if (approved) tn++; else tp++;
    } else {
      n_disagree++;
      if (approved) fn++; else fp++;
      // Print first 50 disagreements with nprobe scan + brute force.
      if (n_disagree <= 50) {
        std::printf("DISAGREE expected=%s got=%s  fc_fast=%d fc_full=%d\n  payload: %.*s\n  scan:",
                    expected ? "approved" : "denied",
                    approved ? "approved" : "denied",
                    fc_fast, fc_full, (int)len, body);
        for (int np : {5, 8, 10, 15, 20, 30, 40, 70, 200, 500, (int)eng.idx.K}) {
          int x = ivf_fraud_count(eng.idx, q16, np);
          std::printf(" np=%d→%d", np, x);
        }
        // Brute-force top-5 across ALL N vectors (no clustering).
        int64_t bd[TOP_K];
        uint32_t bi[TOP_K];
        for (int k = 0; k < TOP_K; ++k) {
          bd[k] = std::numeric_limits<int64_t>::max(); bi[k] = 0;
        }
        for (uint32_t i = 0; i < eng.idx.N; ++i) {
          int64_t d = dist_sq(q16, eng.idx.vectors + i * STRIDE);
          if (d >= bd[TOP_K - 1]) continue;
          int p = TOP_K - 1;
          while (p > 0 && bd[p-1] > d) {
            bd[p] = bd[p-1]; bi[p] = bi[p-1]; --p;
          }
          bd[p] = d; bi[p] = i;
        }
        int brute_fc = 0;
        for (int k = 0; k < TOP_K; ++k) brute_fc += eng.idx.labels[bi[k]];
        std::printf("  BRUTE fc=%d  dists=[", brute_fc);
        for (int k = 0; k < TOP_K; ++k)
          std::printf("%s%lld:l%u", k ? "," : "",
                      (long long)bd[k], eng.idx.labels[bi[k]]);
        std::printf("]\n");
      }
    }
  }

  std::cerr << "tp=" << tp << " tn=" << tn << " fp=" << fp << " fn=" << fn
            << "  (disagreements " << n_disagree << ")\n";

  // Confusion matrix: how each combination of (fc_fast, fc_full,
  // expected) is distributed in the dataset. Reveals exactly which
  // escalation rules would help and which would create new errors.
  std::printf("\n# (fc_fast, fc_full)  legit  fraud\n");
  for (int a = 0; a < 6; ++a) {
    for (int b = 0; b < 6; ++b) {
      int legit = matrix[a][b][1];
      int fraud = matrix[a][b][0];
      if (legit + fraud == 0) continue;
      std::printf("  fast=%d full=%d         %5d  %5d\n", a, b, legit, fraud);
    }
  }

  // Simulate alternative escalation policies.
  auto sim = [&](const char *name, auto policy) {
    int sfp = 0, sfn = 0, esc = 0;
    for (int a = 0; a < 6; ++a)
      for (int b = 0; b < 6; ++b)
        for (int x = 0; x < 2; ++x) {
          int cnt = matrix[a][b][x];
          if (!cnt) continue;
          bool escalate = policy(a);
          int fc = escalate ? b : a;
          if (escalate) esc += cnt;
          bool approved = fc < 3;
          bool expected = (x == 1);
          if (approved != expected) {
            if (approved) sfn += cnt; else sfp += cnt;
          }
        }
    std::printf("policy=%-24s  fp=%d fn=%d  escalated=%d (%.1f%%)\n",
                name, sfp, sfn, esc, 100.0 * esc / 54100);
  };
  sim("current {1,2,3,4}",      [](int a){ return a >= 1 && a <= 4; });
  sim("widen >=1",              [](int a){ return a >= 1; });
  sim("widen <=4",              [](int a){ return a <= 4; });
  sim("always escalate",        [](int a){ (void)a; return true; });

  return 0;
}
