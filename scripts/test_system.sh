#!/bin/bash
# test_system.sh - Comprehensive system test for inference-sentinel
#
# Tests all major functionality:
#   1. Health checks
#   2. Privacy classification (all tiers)
#   3. Local/Cloud routing
#   4. Round-robin selection (local and cloud)
#   5. Shadow mode
#   6. Controller status
#   7. Observability endpoints
#
# Prerequisites:
#   - docker-compose up -d sentinel prometheus grafana
#   - Native Ollama running with gemma3:4b and mistral
#   - ANTHROPIC_API_KEY and GOOGLE_API_KEY set
#
# Usage:
#   ./scripts/test_system.sh

set -e

SENTINEL_URL="${SENTINEL_URL:-http://localhost:8000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }
info() { echo -e "${BLUE}→ $1${NC}"; }
section() { echo -e "\n${YELLOW}═══════════════════════════════════════${NC}"; echo -e "${YELLOW}  $1${NC}"; echo -e "${YELLOW}═══════════════════════════════════════${NC}"; }

# ============================================================
section "1. Health Checks"
# ============================================================

info "Checking sentinel health..."
HEALTH=$(curl -s "${SENTINEL_URL}/health")
if echo "$HEALTH" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
    pass "Sentinel is healthy"
else
    fail "Sentinel health check failed: $HEALTH"
fi

info "Checking backend health..."
echo "  Response: $(echo "$HEALTH" | jq -c '.')"

# ============================================================
section "2. Privacy Classification"
# ============================================================

test_classification() {
    local desc="$1"
    local content="$2"
    local expected_tier="$3"
    local expected_route="$4"
    
    info "Testing: $desc"
    RESPONSE=$(curl -s -X POST "${SENTINEL_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d "{\"messages\": [{\"role\": \"user\", \"content\": \"$content\"}], \"max_tokens\": 10}")
    
    TIER=$(echo "$RESPONSE" | jq -r '.sentinel.privacy_tier')
    ROUTE=$(echo "$RESPONSE" | jq -r '.sentinel.route')
    
    if [ "$TIER" = "$expected_tier" ] && [ "$ROUTE" = "$expected_route" ]; then
        pass "Tier $TIER → $ROUTE"
    else
        fail "Expected tier=$expected_tier route=$expected_route, got tier=$TIER route=$ROUTE"
    fi
}

test_classification "Public content" "What is 2+2?" "0" "cloud"
test_classification "Email (CONFIDENTIAL)" "Contact me at john@example.com" "2" "local"
test_classification "SSN (RESTRICTED)" "My SSN is 123-45-6789" "3" "local"

# ============================================================
section "3. Cloud Round-Robin"
# ============================================================

info "Testing cloud backend alternation..."
CLOUD_MODELS=""
for i in {1..4}; do
    MODEL=$(curl -s -X POST "${SENTINEL_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 5}' \
        | jq -r '.sentinel.model')
    CLOUD_MODELS="$CLOUD_MODELS $MODEL"
    sleep 1
done

echo "  Cloud models used:$CLOUD_MODELS"
if echo "$CLOUD_MODELS" | grep -q "claude" && echo "$CLOUD_MODELS" | grep -q "gemini"; then
    pass "Cloud round-robin working (both Claude and Gemini used)"
else
    echo -e "${YELLOW}⚠ Only one cloud provider responding (check API keys)${NC}"
fi

# ============================================================
section "4. Local Round-Robin"
# ============================================================

info "Testing local backend alternation..."
LOCAL_MODELS=""
for i in {1..4}; do
    MODEL=$(curl -s -X POST "${SENTINEL_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "My SSN is 111-22-3333"}], "max_tokens": 5}' \
        | jq -r '.sentinel.model')
    LOCAL_MODELS="$LOCAL_MODELS $MODEL"
    sleep 1
done

echo "  Local models used:$LOCAL_MODELS"
if echo "$LOCAL_MODELS" | grep -q "gemma" && echo "$LOCAL_MODELS" | grep -q "mistral"; then
    pass "Local round-robin working (both gemma and mistral used)"
elif echo "$LOCAL_MODELS" | grep -qE "gemma|mistral"; then
    echo -e "${YELLOW}⚠ Only one local model responding (pull mistral if missing)${NC}"
else
    fail "No local models responding"
fi

# ============================================================
section "5. Shadow Mode"
# ============================================================

info "Checking shadow mode status..."
SHADOW=$(curl -s "${SENTINEL_URL}/admin/shadow/metrics")
ENABLED=$(echo "$SHADOW" | jq -r '.enabled')
TOTAL=$(echo "$SHADOW" | jq -r '.total_shadows')

if [ "$ENABLED" = "true" ]; then
    pass "Shadow mode enabled (total comparisons: $TOTAL)"
else
    echo -e "${YELLOW}⚠ Shadow mode disabled${NC}"
fi

if [ "$TOTAL" != "0" ] && [ "$TOTAL" != "null" ]; then
    info "Recent shadow results..."
    curl -s "${SENTINEL_URL}/admin/shadow/results?limit=2" | jq -r '.results[] | "  \(.cloud_model) vs \(.local_model): similarity=\(.similarity_score) latency_diff=\(.latency_diff_ms | floor)ms"'
fi

# ============================================================
section "6. Controller Status"
# ============================================================

info "Checking controller status..."
CONTROLLER=$(curl -s "${SENTINEL_URL}/admin/controller/status")
CTRL_ENABLED=$(echo "$CONTROLLER" | jq -r '.enabled')
MODE=$(echo "$CONTROLLER" | jq -r '.mode')

if [ "$CTRL_ENABLED" = "true" ]; then
    pass "Controller enabled (mode: $MODE)"
    echo "  Recommendations:"
    echo "$CONTROLLER" | jq -r '.recommendations[] | "    Tier \(.tier): \(.recommendation) (\(.confidence) confidence)"' 2>/dev/null || echo "    (no recommendations yet)"
else
    echo -e "${YELLOW}⚠ Controller disabled${NC}"
fi

# ============================================================
section "7. Observability"
# ============================================================

info "Checking Prometheus metrics..."
METRICS=$(curl -s "${SENTINEL_URL}/metrics")
METRIC_COUNT=$(echo "$METRICS" | grep -c "^sentinel_" || true)
echo "  Found $METRIC_COUNT sentinel_* metrics"

if [ "$METRIC_COUNT" -gt 10 ]; then
    pass "Prometheus metrics available"
else
    fail "Missing Prometheus metrics"
fi

info "Key metrics:"
echo "$METRICS" | grep -E "^sentinel_requests_total|^sentinel_classifications_total" | head -5

# ============================================================
section "8. Summary"
# ============================================================

echo ""
echo -e "${GREEN}All system tests completed!${NC}"
echo ""
echo "Endpoints:"
echo "  API:        ${SENTINEL_URL}"
echo "  Metrics:    ${SENTINEL_URL}/metrics"
echo "  Grafana:    http://localhost:3000 (admin/sentinel)"
echo "  Prometheus: http://localhost:9091"
echo ""
echo "Quick commands:"
echo "  docker-compose logs -f sentinel    # Watch logs"
echo "  ./scripts/test_shadow_mode.sh      # Detailed shadow test"
echo "  pytest tests/ -v                   # Unit tests"
