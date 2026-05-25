#!/usr/bin/env bash
# Run the rinha k6 stress on the local stack and save results.
# Usage: scripts/run-stress.sh <name> [env=val ...]
set -euo pipefail

cd "$(dirname "$0")/.."

name="${1:-default}"; shift || true

mkdir -p lab/results stress/test
chmod 777 stress/test
rm -f stress/test/results.json

echo ">>> [$name] tearing down stack + wiping caches"
docker compose down -v 2>&1 | tail -2 || true

echo ">>> [$name] starting stack ($*)"
env "$@" docker compose up -d --wait --wait-timeout 180 2>&1 | tail -3

# Warm page cache once.
docker compose exec -T api1 sh -c 'cat /data/ivf.bin > /dev/null 2>&1' || true

echo ">>> [$name] running k6 (rinha-style: --network host, 2 CPU, 8G)"
docker run --rm \
  --network=host \
  --cpus=2 \
  --memory=8g \
  -v "$PWD/stress:/test" \
  -w /test \
  -e K6_NO_USAGE_REPORT=true \
  grafana/k6:latest run --quiet /test/test.js 2>&1 | tail -5

out="lab/results/${name}.json"
if [ -f stress/test/results.json ]; then
  cp stress/test/results.json "$out"
  python3 -c "
import json
d = json.load(open('$out'))
s = d['scoring']
b = s['breakdown']
print(f\"  -> score={s['final_score']:.1f}  p99={d['p99']}  errors={b['http_errors']}  failure={s['failure_rate']}\")
print(f\"  -> tp={b['true_positive_detections']} tn={b['true_negative_detections']} fp={b['false_positive_detections']} fn={b['false_negative_detections']}\")
"
else
  echo "  -> no results.json produced"
fi
