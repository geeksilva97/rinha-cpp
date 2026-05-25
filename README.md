# rinha-cpp

Full-C++ submission for **rinha-de-backend-2026** (fraud detection).
LB, HTTP server, JSON parser, and IVF kNN index — all C++. No nginx, no
Ruby, no third-party HTTP framework.

## Design

```
  client ──TCP──> lb ──SCM_RIGHTS─> api1 ──direct TCP──> client
                     \──SCM_RIGHTS─> api2 ──direct TCP──> client
```

### Load balancer (`src/lb/lb.cpp`)
Listens TCP `:9999`. Holds one persistent UNIX-stream connection to each
API's control socket. On `accept()`, picks the next upstream
round-robin, calls `sendmsg()` with `SCM_RIGHTS` to hand the *accepted
client fd* to that worker, then `close()`s its own copy. After that the
LB is out of the data path entirely — the worker reads and writes the
TCP socket directly.

Per request the LB does **3 syscalls**: `accept4`, `sendmsg`, `close`.
No HTTP parsing (rinha-rule compliant by construction — it cannot
inspect what it never reads).

### API server (`src/http/server.cpp`)
Single thread, edge-triggered epoll. Receives client fds from the LB
via `recvmsg(SCM_RIGHTS)` and treats them like normal client
connections. Hand-rolled HTTP/1.1 parser; six pre-built JSON responses
indexed by `fraud_count ∈ {0..5}` (no JSON serialization on the hot
path). Keep-alive supported.

### Engine (`src/engine/`)
- `parser.hpp` — alloc-free byte-walker specialized for the
  `/fraud-score` payload schema (no DOM, no string copies).
- `ivf_index.hpp` — int16-quantized IVF (Inverted File) kNN with AVX2
  `_mm256_madd_epi16` distance kernel and an mmap'd read-only index.
- `engine.hpp` — adaptive nprobe: cheap first pass, full pass only when
  the cheap result is non-consensus (would otherwise flip the
  decision).

## Build & run

```sh
make                                 # local build
docker compose up --build            # full stack on :9999
curl -s http://localhost:9999/ready  # → ok
```

The IVF index (`resources/ivf.bin.gz` ≈ 33 MB) is decompressed inside
the image at build time; the runtime image carries the raw `ivf.bin`
mmap'd by both API processes.

## Resource budget (1 CPU / 350 MB total)

| service | cpus | memory |
|---------|-----:|-------:|
| lb      | 0.10 |  20 MB |
| api1    | 0.45 | 165 MB |
| api2    | 0.45 | 165 MB |
