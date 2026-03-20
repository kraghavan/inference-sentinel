#!/bin/bash
# Test script for Phase 3 Observability
# Run this after starting the stack to generate metrics

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"
NUM_REQUESTS="${NUM_REQUESTS:-10}"

echo "=== Inference Sentinel Observability Test ==="
echo "Base URL: $BASE_URL"
echo "Requests: $NUM_REQUESTS"
echo ""

# Check health
echo "1. Health Check..."
curl -s "$BASE_URL/health" | jq .
echo ""

# Check metrics endpoint
echo "2. Metrics Endpoint..."
curl -s "$BASE_URL/metrics" | head -20
echo "..."
echo ""

# Generate mixed traffic
echo "3. Generating mixed traffic..."

# Public requests (tier 0 → cloud)
echo "   - Sending PUBLIC tier requests..."
for i in $(seq 1 $((NUM_REQUESTS / 2))); do
    curl -s -X POST "$BASE_URL/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}]}' \
        | jq -r '"   Request '$i': tier=\(.sentinel.privacy_tier) route=\(.sentinel.route) latency=\(.sentinel.inference_latency_ms | floor)ms"'
done

# Sensitive requests (tier 3 → local)
echo "   - Sending RESTRICTED tier requests..."
for i in $(seq 1 $((NUM_REQUESTS / 2))); do
    curl -s -X POST "$BASE_URL/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "My SSN is 123-45-6789, please help me"}]}' \
        | jq -r '"   Request '$i': tier=\(.sentinel.privacy_tier) route=\(.sentinel.route) latency=\(.sentinel.inference_latency_ms | floor)ms"'
done

echo ""
echo "4. Checking Prometheus metrics..."
echo ""

# Show key metrics
echo "=== Request Counts ==="
curl -s "$BASE_URL/metrics" | grep -E "^sentinel_requests_total" | head -10

echo ""
echo "=== TTFT Histogram ==="
curl -s "$BASE_URL/metrics" | grep -E "^sentinel_ttft_seconds" | head -5

echo ""
echo "=== Classification Counts ==="
curl -s "$BASE_URL/metrics" | grep -E "^sentinel_classifications_total" | head -5

echo ""
echo "=== Cost Tracking ==="
curl -s "$BASE_URL/metrics" | grep -E "^sentinel_cost" | head -5

echo ""
echo "=== Backend Health ==="
curl -s "$BASE_URL/metrics" | grep -E "^sentinel_(local|cloud)_backend_healthy"

echo ""
echo "=== Test Complete ==="
echo ""
echo "View dashboards:"
echo "  Grafana:     http://localhost:3000 (admin/sentinel)"
echo "  Prometheus:  http://localhost:9091"
echo "  Tempo:       http://localhost:3200"
