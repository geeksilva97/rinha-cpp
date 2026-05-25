// Edge-triggered epoll HTTP/1.1 server, single thread per process.
//
// Inbound model: the LB hands us already-accepted client TCP fds via
// SCM_RIGHTS over our control socket. We never call accept() on a
// listening socket — the kernel has done that work for us.
//
// Hot path:
//   read() into per-conn buffer → parse req → engine.fraud_count() →
//   write() one of 6 pre-built responses. HTTP keep-alive supported.

#include "../engine/engine.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

// Sentinels stored in epoll_event.data.u64. Real Conn* pointers are
// always above the first 4 GiB, so any value < 2^32 is a sentinel.
constexpr uint64_t CTRL_TAG   = 1;
constexpr uint64_t LISTEN_TAG = 2;
constexpr int MAX_EVENTS = 256;
constexpr int CONN_BUF   = 8192;

// Pre-built responses indexed by fraud_count (0..5).
static const char *RESPONSES[6];
static int RESPONSES_LEN[6];
static char RESP_STORAGE[6][192];

static const char READY_RESP[] =
  "HTTP/1.1 200 OK\r\n"
  "content-type: text/plain\r\n"
  "content-length: 2\r\n"
  "\r\n"
  "ok";
static const char NOT_FOUND_RESP[] =
  "HTTP/1.1 404 Not Found\r\n"
  "content-length: 0\r\n"
  "\r\n";
static const char BAD_REQ_RESP[] =
  "HTTP/1.1 400 Bad Request\r\n"
  "content-length: 0\r\n"
  "connection: close\r\n"
  "\r\n";
constexpr int READY_LEN     = sizeof(READY_RESP) - 1;
constexpr int NOT_FOUND_LEN = sizeof(NOT_FOUND_RESP) - 1;
constexpr int BAD_REQ_LEN   = sizeof(BAD_REQ_RESP) - 1;

static void build_responses() {
  for (int i = 0; i <= 5; ++i) {
    const char *appr = (i < 3) ? "true" : "false"; // threshold 0.6 = 3/5
    char body[64];
    int bn = std::snprintf(body, sizeof(body),
                           "{\"approved\":%s,\"fraud_score\":%.1f}",
                           appr, i * 0.2f);
    int n = std::snprintf(RESP_STORAGE[i], sizeof(RESP_STORAGE[i]),
      "HTTP/1.1 200 OK\r\n"
      "content-type: application/json\r\n"
      "content-length: %d\r\n"
      "\r\n"
      "%s",
      bn, body);
    RESPONSES[i] = RESP_STORAGE[i];
    RESPONSES_LEN[i] = n;
  }
}

struct Conn {
  int  fd = -1;
  char buf[CONN_BUF];
  int  len = 0;
  // pending write
  const char *wbuf = nullptr;
  int  wlen = 0;
  int  wpos = 0;
};

static rinha::Engine g_engine;

static int set_nonblock(int fd) {
  int fl = fcntl(fd, F_GETFL, 0);
  return fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

static int write_all(Conn *c, int epfd) {
  while (c->wpos < c->wlen) {
    ssize_t n = write(c->fd, c->wbuf + c->wpos, c->wlen - c->wpos);
    if (n > 0) { c->wpos += n; continue; }
    if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      epoll_event ev{};
      ev.events = EPOLLIN | EPOLLOUT | EPOLLET | EPOLLRDHUP;
      ev.data.ptr = c;
      epoll_ctl(epfd, EPOLL_CTL_MOD, c->fd, &ev);
      return 0;
    }
    return -1;
  }
  c->wbuf = nullptr; c->wlen = 0; c->wpos = 0;
  return 1;
}

static int parse_request_headers(const char *p, int len, int &content_length,
                                 const char *&method, int &method_len,
                                 const char *&path, int &path_len) {
  const char *eol = static_cast<const char *>(memchr(p, '\n', len));
  if (!eol) return -1;
  method = p;
  const char *sp1 = static_cast<const char *>(memchr(p, ' ', eol - p));
  if (!sp1) return -2;
  method_len = sp1 - p;
  path = sp1 + 1;
  const char *sp2 = static_cast<const char *>(memchr(path, ' ', eol - path));
  if (!sp2) return -2;
  path_len = sp2 - path;

  const char *hdr_end = nullptr;
  for (int i = 0; i + 3 < len; ++i) {
    if (p[i] == '\r' && p[i+1] == '\n' && p[i+2] == '\r' && p[i+3] == '\n') {
      hdr_end = p + i + 4; break;
    }
  }
  if (!hdr_end) return -1;

  content_length = 0;
  const char *cur = eol + 1;
  while (cur < hdr_end - 2) {
    const char *line_end = static_cast<const char *>(memchr(cur, '\n', hdr_end - cur));
    if (!line_end) break;
    int hlen = line_end - cur;
    if (hlen >= 16) {
      static const char CL[] = "content-length:";
      bool match = true;
      for (int j = 0; j < 15; ++j) {
        char ch = cur[j];
        if (ch >= 'A' && ch <= 'Z') ch += 32;
        if (ch != CL[j]) { match = false; break; }
      }
      if (match) {
        const char *v = cur + 15;
        while (v < line_end && (*v == ' ' || *v == '\t')) v++;
        int n = 0;
        while (v < line_end && *v >= '0' && *v <= '9') { n = n*10 + (*v - '0'); v++; }
        content_length = n;
      }
    }
    cur = line_end + 1;
  }
  return hdr_end - p;
}

static void close_conn(int epfd, Conn *c) {
  epoll_ctl(epfd, EPOLL_CTL_DEL, c->fd, nullptr);
  close(c->fd);
  delete c;
}

static int send_response(Conn *c, const char *resp, int rlen, int epfd) {
  c->wbuf = resp; c->wlen = rlen; c->wpos = 0;
  return write_all(c, epfd);
}

static int handle_readable(Conn *c, int epfd) {
  for (;;) {
    if (c->len >= CONN_BUF) return -1;
    ssize_t n = read(c->fd, c->buf + c->len, CONN_BUF - c->len);
    if (n > 0) c->len += n;
    else if (n == 0) return -1;
    else if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    else return -1;
  }

  while (c->len > 0) {
    int cl = 0;
    const char *method = nullptr, *path = nullptr;
    int mlen = 0, plen = 0;
    int hdr_bytes = parse_request_headers(c->buf, c->len, cl, method, mlen, path, plen);
    if (hdr_bytes == -1) return 0;
    if (hdr_bytes == -2) {
      send_response(c, BAD_REQ_RESP, BAD_REQ_LEN, epfd);
      return -1;
    }
    int total = hdr_bytes + cl;
    if (c->len < total) return 0;

    int rc = 0;
    if (mlen == 3 && method[0] == 'G' && plen == 6 &&
        std::memcmp(path, "/ready", 6) == 0) {
      rc = send_response(c, READY_RESP, READY_LEN, epfd);
    } else if (mlen == 4 && method[0] == 'P' && plen == 12 &&
               std::memcmp(path, "/fraud-score", 12) == 0) {
      int fc = g_engine.fraud_count(c->buf + hdr_bytes, cl);
      if (fc < 0) fc = 0; else if (fc > 5) fc = 5;
      rc = send_response(c, RESPONSES[fc], RESPONSES_LEN[fc], epfd);
    } else {
      rc = send_response(c, NOT_FOUND_RESP, NOT_FOUND_LEN, epfd);
    }
    if (rc < 0) return -1;

    int remaining = c->len - total;
    if (remaining > 0) std::memmove(c->buf, c->buf + total, remaining);
    c->len = remaining;

    if (c->wbuf != nullptr) return 0; // backpressure
  }
  return 0;
}

static int handle_writable(Conn *c, int epfd) {
  if (c->wbuf) {
    int rc = write_all(c, epfd);
    if (rc < 0) return -1;
    if (c->wbuf != nullptr) return 0;
  }
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET | EPOLLRDHUP;
  ev.data.ptr = c;
  epoll_ctl(epfd, EPOLL_CTL_MOD, c->fd, &ev);
  return 0;
}

// Receive one fd from the LB via SCM_RIGHTS. Returns -1 on EAGAIN/empty.
static int recv_fd(int ctrl) {
  char dummy;
  iovec iov{};
  iov.iov_base = &dummy; iov.iov_len = 1;

  union {
    char buf[CMSG_SPACE(sizeof(int))];
    cmsghdr align;
  } u;
  std::memset(&u, 0, sizeof(u));

  msghdr msg{};
  msg.msg_iov = &iov; msg.msg_iovlen = 1;
  msg.msg_control = u.buf; msg.msg_controllen = sizeof(u.buf);

  ssize_t n = recvmsg(ctrl, &msg, MSG_DONTWAIT);
  if (n <= 0) return -1;

  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS)
    return -1;
  int fd = -1;
  std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  return fd;
}

static void register_client(int epfd, int cfd) {
  set_nonblock(cfd);
  int yes = 1;
  setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  Conn *c = new Conn();
  c->fd = cfd;
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET | EPOLLRDHUP;
  ev.data.ptr = c;
  epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);
}

} // namespace

int main() {
  signal(SIGPIPE, SIG_IGN);
  build_responses();

  const char *ivf_path = std::getenv("IVF_PATH");
  if (!ivf_path) ivf_path = "/data/ivf.bin";
  if (!g_engine.load(ivf_path)) {
    std::cerr << "engine: failed to load " << ivf_path << "\n";
    return 1;
  }
  if (const char *e = std::getenv("NPROBE"))      g_engine.nprobe      = std::atoi(e);
  if (const char *e = std::getenv("FAST_NPROBE")) g_engine.fast_nprobe = std::atoi(e);
  std::cerr << "engine: nprobe=" << g_engine.nprobe
            << " fast_nprobe=" << g_engine.fast_nprobe << "\n";

  // Set up the control socket — the LB will connect here and send fds.
  const char *ctrl_path = std::getenv("CTRL");
  if (!ctrl_path) ctrl_path = "/run/sock/api.ctrl";
  unlink(ctrl_path);
  int listen_ctrl = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, ctrl_path, sizeof(addr.sun_path) - 1);
  if (bind(listen_ctrl, (sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
  chmod(ctrl_path, 0666);
  if (listen(listen_ctrl, 8) < 0) { perror("listen"); return 1; }

  // Warmup with example payloads — page in mmap, prime branch predictors.
  if (const char *wpath = std::getenv("WARMUP_PATH")) {
    FILE *f = std::fopen(wpath, "rb");
    if (f) {
      std::fseek(f, 0, SEEK_END);
      long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      std::string blob(sz, 0);
      if (std::fread(blob.data(), 1, sz, f) == (size_t)sz) {
        int rounds = std::getenv("WARMUP_ROUNDS") ? std::atoi(std::getenv("WARMUP_ROUNDS")) : 5;
        std::vector<std::pair<size_t,size_t>> payloads;
        int depth = 0; size_t start = 0;
        for (size_t i = 0; i < blob.size(); ++i) {
          char ch = blob[i];
          if (ch == '{') { if (depth == 0) start = i; depth++; }
          else if (ch == '}') {
            depth--;
            if (depth == 0) payloads.push_back({start, i - start + 1});
          }
        }
        auto t0 = std::chrono::steady_clock::now();
        int total = 0;
        for (int r = 0; r < rounds; ++r) {
          for (auto &p : payloads) {
            g_engine.fraud_count(blob.data() + p.first, p.second);
            total++;
          }
        }
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cerr << "warmup: " << total << " queries in " << ms << "ms\n";
      }
      std::fclose(f);
    }
  }

  int epfd = epoll_create1(EPOLL_CLOEXEC);
  epoll_event lev{};
  lev.events = EPOLLIN | EPOLLET;
  lev.data.u64 = LISTEN_TAG;
  epoll_ctl(epfd, EPOLL_CTL_ADD, listen_ctrl, &lev);

  // Single LB control connection (we only register it after accept).
  int lb_fd = -1;

  std::cerr << "api: ctrl=" << ctrl_path << " ready, waiting for LB\n";

  epoll_event events[MAX_EVENTS];
  for (;;) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; ++i) {
      uint64_t raw = events[i].data.u64;
      // Sentinels are small constants; real Conn* pointers always have
      // address bits set far higher than 2^16.
      if (raw == LISTEN_TAG) {
        int afd = accept4(listen_ctrl, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (afd < 0) continue;
        if (lb_fd >= 0) close(lb_fd);
        lb_fd = afd;
        epoll_event cev{};
        cev.events = EPOLLIN | EPOLLET;
        cev.data.u64 = CTRL_TAG;
        epoll_ctl(epfd, EPOLL_CTL_ADD, lb_fd, &cev);
        std::cerr << "api: LB connected (fd=" << lb_fd << ")\n";
        continue;
      }
      if (raw == CTRL_TAG) {
        for (;;) {
          int newfd = recv_fd(lb_fd);
          if (newfd < 0) break;
          register_client(epfd, newfd);
        }
        continue;
      }
      Conn *c = static_cast<Conn *>(events[i].data.ptr);
      uint32_t evs = events[i].events;
      if (evs & (EPOLLHUP | EPOLLRDHUP | EPOLLERR)) {
        close_conn(epfd, c); continue;
      }
      if (evs & EPOLLIN) {
        if (handle_readable(c, epfd) < 0) { close_conn(epfd, c); continue; }
      }
      if (evs & EPOLLOUT) {
        if (handle_writable(c, epfd) < 0) { close_conn(epfd, c); continue; }
      }
    }
  }
}
