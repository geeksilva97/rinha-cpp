FROM --platform=linux/amd64 debian:bookworm-slim AS builder
ARG CXX_MARCH=haswell

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      g++ make ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Makefile ./
COPY src ./src
RUN make CXX_MARCH=${CXX_MARCH} -j$(nproc)

# The IVF index ships inside the image (committed as .gz to keep the
# repo on GitHub). Decompress at build time so the runtime image has the
# raw mmap-able file.
COPY resources/ivf.bin.gz /tmp/ivf.bin.gz
RUN gunzip /tmp/ivf.bin.gz && ls -lh /tmp/ivf.bin

# --- runtime image ---
FROM --platform=linux/amd64 debian:bookworm-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/api /usr/local/bin/api
COPY --from=builder /src/build/lb  /usr/local/bin/lb
COPY --from=builder /tmp/ivf.bin   /data/ivf.bin
COPY resources/example-payloads.json /data/example-payloads.json

CMD ["/usr/local/bin/api"]
