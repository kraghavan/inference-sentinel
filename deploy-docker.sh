#!/bin/bash
# deploy-docker.sh - Deploy inference-sentinel with Docker Compose
# Location: ops/docker/deploy.sh (or repo root)
set -e

echo "🚀 inference-sentinel Docker Compose Deployment"
echo "================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Handle both ops/docker/ location and repo root location
if [[ -f "${SCRIPT_DIR}/docker-compose.yml" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../docker-compose.yml" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    REPO_ROOT="${SCRIPT_DIR}"
fi

cd "${REPO_ROOT}"
echo "Working directory: ${REPO_ROOT}"

# Parse arguments
CLEAN=false
REBUILD=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --rebuild)
            REBUILD=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean    Remove all volumes and rebuild from scratch"
            echo "  --rebuild  Rebuild images without using cache"
            echo "  --quick    Skip health checks, just start"
            echo "  --help     Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate required env vars
if [[ -z "${ANTHROPIC_API_KEY}" ]] || [[ -z "${GOOGLE_API_KEY}" ]]; then
    echo -e "${RED}❌ ERROR: ANTHROPIC_API_KEY and GOOGLE_API_KEY must be set${NC}"
    echo ""
    echo "Export them first:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    echo "  export GOOGLE_API_KEY='AIza...'"
    echo ""
    echo "Or run inline:"
    echo "  ANTHROPIC_API_KEY='...' GOOGLE_API_KEY='...' ./deploy-docker.sh"
    exit 1
fi

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ docker not found${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Start Docker Desktop first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found${NC}"
    exit 1
fi

# Use docker compose (v2) or docker-compose (v1)
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Check if docker-compose.yml exists
if [[ ! -f "docker-compose.yml" ]]; then
    echo -e "${RED}❌ docker-compose.yml not found in ${REPO_ROOT}${NC}"
    exit 1
fi

# Clean up if requested
if [[ "$CLEAN" == true ]]; then
    echo -e "\n${YELLOW}Cleaning up (removing containers and volumes)...${NC}"
    $COMPOSE_CMD down -v --remove-orphans 2>/dev/null || true
    
    # Remove specific volumes if they exist
    docker volume rm inference-sentinel_prometheus_data 2>/dev/null || true
    docker volume rm inference-sentinel_grafana_data 2>/dev/null || true
    docker volume rm inference-sentinel_loki_data 2>/dev/null || true
    docker volume rm inference-sentinel_tempo_data 2>/dev/null || true
    
    # Clear pycache
    echo -e "${YELLOW}Clearing Python cache...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
fi

# Check Ollama is running
echo -e "\n${YELLOW}Checking Ollama...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    OLLAMA_MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null | head -3 | tr '\n' ', ' | sed 's/,$//')
    echo -e "${GREEN}✅ Ollama running with models: ${OLLAMA_MODELS}${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama not detected at localhost:11434${NC}"
    echo "   Local inference will fail. Start Ollama with: ollama serve"
fi

# Build
echo -e "\n${YELLOW}Building containers...${NC}"
if [[ "$REBUILD" == true ]]; then
    $COMPOSE_CMD build --no-cache sentinel
else
    $COMPOSE_CMD build sentinel
fi

# Start services
echo -e "\n${YELLOW}Starting services...${NC}"
$COMPOSE_CMD up -d

if [[ "$QUICK" == true ]]; then
    echo -e "\n${GREEN}✅ Services started (quick mode, skipping health checks)${NC}"
else
    # Wait for services to be ready
    echo -e "\n${YELLOW}Waiting for services to be ready...${NC}"
    
    # Wait for sentinel
    echo -n "  Sentinel: "
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅${NC}"
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo -e "${RED}❌ timeout${NC}"
        fi
        sleep 1
    done
    
    # Wait for Grafana
    echo -n "  Grafana:  "
    for i in {1..30}; do
        if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅${NC}"
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo -e "${RED}❌ timeout${NC}"
        fi
        sleep 1
    done
    
    # Wait for Prometheus
    echo -n "  Prometheus: "
    for i in {1..20}; do
        if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
            echo -e "${GREEN}✅${NC}"
            break
        fi
        if [[ $i -eq 20 ]]; then
            echo -e "${YELLOW}⚠️ slow${NC}"
        fi
        sleep 1
    done
fi

# Show status
echo -e "\n${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "============================================"
echo "ACCESS"
echo "============================================"
echo ""
echo "  🌐 Sentinel API:  http://localhost:8000"
echo "  📊 Grafana:       http://localhost:3000 (admin/admin)"
echo "  📈 Prometheus:    http://localhost:9090"
echo ""
echo "Test:"
echo "  curl http://localhost:8000/health"
echo ""
echo "Quick inference test:"
echo '  curl -s -X POST http://localhost:8000/v1/inference \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '\''{"model":"auto","messages":[{"role":"user","content":"Hello"}]}'\'' | jq .'
echo ""
echo "============================================"
echo "BENCHMARKS"
echo "============================================"
echo ""
echo "Generate test data and run:"
echo "  python -m benchmarks.harness --generate --count 200 --experiment all --ner"
echo ""
echo "Generate report:"
echo "  python -m benchmarks.report"
echo ""
echo "============================================"

# Show container status
echo -e "\n${YELLOW}Container status:${NC}"
$COMPOSE_CMD ps