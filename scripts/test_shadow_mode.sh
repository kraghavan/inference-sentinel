#!/bin/bash
# test_shadow_mode.sh - Test shadow mode functionality
# 
# Prerequisites:
# - Sentinel running with shadow mode enabled
# - Both cloud (Anthropic) and local (Ollama) backends available
# 
# Usage:
#   ./scripts/test_shadow_mode.sh

set -e

SENTINEL_URL="${SENTINEL_URL:-http://localhost:8000}"

echo "=============================================="
echo "Shadow Mode Testing Script"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Check health
echo -e "${YELLOW}1. Checking service health...${NC}"
curl -s "${SENTINEL_URL}/health" | jq .
echo ""

# 2. Check shadow metrics (before)
echo -e "${YELLOW}2. Shadow metrics (before tests)...${NC}"
curl -s "${SENTINEL_URL}/admin/shadow/metrics" | jq .
echo ""

# 3. Send tier 0 requests (should trigger shadow mode)
echo -e "${YELLOW}3. Sending tier 0 requests (public content)...${NC}"
echo "   These should route to cloud AND trigger shadow mode"
echo ""

for i in {1..3}; do
    echo "   Request $i..."
    curl -s "${SENTINEL_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
            "max_tokens": 50
        }' | jq '{route: .sentinel.route, backend: .sentinel.backend, tier: .sentinel.privacy_tier}'
    sleep 1
done
echo ""

# 4. Send tier 2 request (should NOT trigger shadow mode)
echo -e "${YELLOW}4. Sending tier 2 request (should route local, no shadow)...${NC}"
curl -s "${SENTINEL_URL}/v1/inference" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [{"role": "user", "content": "My email is john.doe@example.com. What can you help me with?"}],
        "max_tokens": 50
    }' | jq '{route: .sentinel.route, backend: .sentinel.backend, tier: .sentinel.privacy_tier}'
echo ""

# 5. Wait for shadow tasks to complete
echo -e "${YELLOW}5. Waiting for shadow tasks to complete...${NC}"
sleep 5
echo ""

# 6. Check shadow metrics (after)
echo -e "${YELLOW}6. Shadow metrics (after tests)...${NC}"
curl -s "${SENTINEL_URL}/admin/shadow/metrics" | jq .
echo ""

# 7. Get recent shadow results
echo -e "${YELLOW}7. Recent shadow comparison results...${NC}"
curl -s "${SENTINEL_URL}/admin/shadow/results?limit=5" | jq .
echo ""

# 8. Summary
echo -e "${GREEN}=============================================="
echo "Shadow Mode Test Complete!"
echo "=============================================="
echo ""
echo "Check the metrics above to verify:"
echo "  - total_shadows increased (should be 3)"
echo "  - successful_shadows matches total"
echo "  - quality_match_rate indicates local quality"
echo "  - latency_diff_ms shows speed comparison"
echo ""
echo "Grafana dashboard: http://localhost:3000"
echo "Look for shadow_* metrics in Prometheus"
