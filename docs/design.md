# Design notes

## Why a single-binary C++ stack

This is the **rinha-cpp** rewrite of an earlier mixed Ruby/C++ submission.
That earlier version learned, through profiling, that:

1. The IVF query itself was already ~30–100 µs on Haswell.
2. Ruby request handling (Puma → Rack → ext call → response build) was
   adding ~100 µs of overhead on top of that.
3. nginx → unix-socket Puma added another ~30 µs.
4. The dominant remaining tail-latency contributors were Ractor IPC,
   Ruby GC, and proxy buffer copies in nginx.

In a 1-CPU/350-MB budget those overheads are a sizeable fraction of
total p99. Replacing the whole stack with hand-rolled C++ should
collapse them.

## The SCM_RIGHTS handoff

The previous design had a fan-in/fan-out chain:

```
client → nginx → unix → ruby/puma → ext.c → ruby → unix → nginx → client
```

Every arrow is two pipe copies and at least one syscall. Five arrows.

The current design uses `SCM_RIGHTS` to short-circuit the data path:

```
                                ┌─────► api1 ◄─────┐
client ─── TCP ─► lb ─── SCM ───┤                   │── direct TCP ─► client
                                └─────► api2 ◄─────┘
```

After the LB's `sendmsg`, the LB is **completely out of the data path**.
The worker reads the TCP socket directly; the LB cannot inspect what it
never sees.

Per request, the LB performs three syscalls:
1. `accept4()` — get the client fd
2. `sendmsg()` with `SCM_RIGHTS` — hand it off
3. `close()` — drop our copy

That's the entire LB hot path. Adding more upstreams or routing logic
is essentially free.

## What's specialized

Generic frameworks pay generality costs we don't want. Each layer is
specialized to this exact endpoint:

| layer | generic version would do | we do |
|---|---|---|
| LB | parse HTTP, route, buffer | pass fd; never read |
| HTTP parser | tokenize headers, build maps | scan for `method`, `path`, `content-length` |
| body parser | JSON DOM + nested object lookup | one-pass byte walker keyed by 1st byte + key length |
| response builder | serialize JSON | one of 6 pre-built string literals |
| kNN | full nprobe every time | adaptive: fast first, escalate only when borderline |
| index format | float32 + dynamic | int16-quantized, padded for AVX2, mmap'd RO |

## Resource split

| service | cpus | memory | why |
|---|---:|---:|---|
| lb   | 0.10 |  20 MB | almost idle CPU; just fd-passing |
| api1 | 0.45 | 165 MB | mmap of 99 MB ivf.bin + epoll buffers |
| api2 | 0.45 | 165 MB | same |

Both api containers ship the same image and read the same on-disk
`ivf.bin`. The kernel page cache is shared via inode identity, so we
do not pay for the index twice.
