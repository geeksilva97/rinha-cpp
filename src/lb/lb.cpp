// SCM_RIGHTS load balancer.
//
// On accept(:9999) we hand the client TCP fd to the next API worker via
// SCM_RIGHTS (sendmsg ancillary data over a Unix datagram socket) and
// close our local copy. After that we are completely out of the data
// path — the worker reads/writes the TCP socket directly.
//
// We cannot inspect the payload because we never read it. The bytes
// never traverse this process. Rinha-compliant by construction.
//
// Per-request cost in the LB: accept4 + sendmsg + close. Nothing else.

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int MAX_EVENTS = 64;
constexpr int BACKLOG    = 4096;

static std::vector<std::string> g_upstreams;
static std::vector<int>          g_upstream_fds;
static uint32_t                  g_rr = 0;

static int dial_upstream(const std::string &path) {
  for (int attempt = 0; attempt < 60; ++attempt) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (connect(fd, (sockaddr *)&addr, sizeof(addr)) == 0) {
      // Bump send buffer so a burst of fd-passes doesn't block.
      int sndbuf = 1 << 20;
      setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));
      return fd;
    }
    close(fd);
    if (attempt == 0) std::cerr << "lb: waiting for " << path << "...\n";
    usleep(500 * 1000);
  }
  return -1;
}

static int send_fd(int ctrl, int fd) {
  // One-byte payload required; the cmsg carries the fd.
  char dummy = 'F';
  iovec iov{};
  iov.iov_base = &dummy;
  iov.iov_len  = 1;

  union {
    char buf[CMSG_SPACE(sizeof(int))];
    cmsghdr align;
  } u;
  std::memset(&u, 0, sizeof(u));

  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = u.buf;
  msg.msg_controllen = sizeof(u.buf);

  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type  = SCM_RIGHTS;
  cmsg->cmsg_len   = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));

  for (;;) {
    ssize_t n = sendmsg(ctrl, &msg, MSG_NOSIGNAL);
    if (n == 1) return 0;
    if (n < 0 && errno == EINTR) continue;
    return -1;
  }
}

} // namespace

int main() {
  signal(SIGPIPE, SIG_IGN);

  const char *up = std::getenv("UPSTREAMS");
  if (!up || !*up) { std::cerr << "UPSTREAMS env required\n"; return 1; }
  std::string s = up;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t c = s.find(',', pos);
    std::string item = s.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    if (!item.empty()) g_upstreams.push_back(item);
    if (c == std::string::npos) break;
    pos = c + 1;
  }

  for (auto &u : g_upstreams) {
    int fd = dial_upstream(u);
    if (fd < 0) { std::cerr << "lb: cannot dial " << u << "\n"; return 1; }
    std::cerr << "lb: connected to " << u << " (fd=" << fd << ")\n";
    g_upstream_fds.push_back(fd);
  }

  int port = 9999;
  if (const char *e = std::getenv("PORT")) port = std::atoi(e);

  int srv = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  int yes = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  int defer = 1;
  // Wait for first byte before waking us up — TCP_DEFER_ACCEPT can shave
  // one accept/epoll round-trip when client opens then immediately sends.
  setsockopt(srv, IPPROTO_TCP, TCP_DEFER_ACCEPT, &defer, sizeof(defer));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = INADDR_ANY;
  if (bind(srv, (sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
  if (listen(srv, BACKLOG) < 0) { perror("listen"); return 1; }

  int epfd = epoll_create1(EPOLL_CLOEXEC);
  epoll_event ev{};
  ev.events = EPOLLIN | EPOLLET;
  ev.data.fd = srv;
  epoll_ctl(epfd, EPOLL_CTL_ADD, srv, &ev);

  std::cerr << "lb: listening :" << port
            << " upstreams=" << g_upstream_fds.size() << "\n";

  epoll_event events[MAX_EVENTS];
  uint64_t accepted = 0, failed = 0;
  for (;;) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; ++i) {
      if (events[i].data.fd != srv) continue;
      for (;;) {
        int cfd = accept4(srv, nullptr, nullptr, SOCK_CLOEXEC);
        if (cfd < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) break;
          if (errno == EINTR) continue;
          perror("accept"); break;
        }
        int ctrl = g_upstream_fds[g_rr++ % g_upstream_fds.size()];
        if (send_fd(ctrl, cfd) < 0) {
          failed++;
          // best-effort: just close
        } else {
          accepted++;
        }
        close(cfd);
      }
    }
    // periodic log every 8K accepts
    if (accepted && (accepted % 8192) == 0) {
      std::cerr << "lb: accepted=" << accepted << " failed=" << failed << "\n";
    }
  }
}
