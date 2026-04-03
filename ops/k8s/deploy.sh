#!/bin/bash
# deploy.sh - Deploy inference-sentinel to minikube
# Location: ops/k8s/deploy.sh
set -e

echo "🚀 inference-sentinel Kubernetes Deployment"
echo "============================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory (ops/k8s/) and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DASHBOARDS_DIR="${REPO_ROOT}/observability/grafana/dashboards"

echo "Script directory: ${SCRIPT_DIR}"
echo "Repo root: ${REPO_ROOT}"
echo "Dashboards directory: ${DASHBOARDS_DIR}"

# Validate required env vars upfront
if [[ -z "${ANTHROPIC_API_KEY}" ]] || [[ -z "${GOOGLE_API_KEY}" ]]; then
    echo -e "${RED}❌ ERROR: ANTHROPIC_API_KEY and GOOGLE_API_KEY must be set${NC}"
    echo ""
    echo "Export them first:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    echo "  export GOOGLE_API_KEY='AIza...'"
    echo ""
    echo "Or run inline:"
    echo "  ANTHROPIC_API_KEY='...' GOOGLE_API_KEY='...' ./ops/k8s/deploy.sh"
    exit 1
fi

# Cleanup function for failed/orphaned minikube resources
cleanup_minikube() {
    echo -e "${YELLOW}Cleaning up orphaned minikube resources...${NC}"
    docker rm -f minikube-preload-sidecar 2>/dev/null || true
    docker rm -f $(docker ps -aq --filter "label=name.minikube.sigs.k8s.io=minikube" 2>/dev/null) 2>/dev/null || true
    docker rm -f $(docker ps -aq --filter "name=minikube" 2>/dev/null) 2>/dev/null || true
}

# Trap to cleanup on script failure
trap 'echo -e "${RED}Script failed. Run: minikube delete --purge${NC}"' ERR

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command -v minikube &> /dev/null; then
    echo -e "${RED}❌ minikube not found. Install from: https://minikube.sigs.k8s.io/docs/start/${NC}"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Install from: https://kubernetes.io/docs/tasks/tools/${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Start Docker Desktop first.${NC}"
    exit 1
fi

# Clean up any orphaned containers BEFORE checking status
cleanup_minikube

# Check minikube status
echo -e "\n${YELLOW}Checking minikube status...${NC}"
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "None")
echo "Current minikube status: $MINIKUBE_STATUS"

if [ "$MINIKUBE_STATUS" != "Running" ]; then
    echo -e "${YELLOW}Starting minikube (current status: $MINIKUBE_STATUS)...${NC}"
    
    # If in a corrupted/unknown state, delete first
    if [ "$MINIKUBE_STATUS" != "None" ] && [ "$MINIKUBE_STATUS" != "Stopped" ]; then
        echo -e "${YELLOW}Detected corrupted state, performing full cleanup...${NC}"
        minikube delete --purge 2>/dev/null || true
        cleanup_minikube
        docker volume rm minikube 2>/dev/null || true
        sleep 2
    fi
    
    # Start fresh
    minikube start --cpus=4 --memory=8192 --driver=docker --preload=false
fi

# Wait for minikube to be fully ready
echo -e "\n${YELLOW}Waiting for minikube cluster to be ready...${NC}"
if ! kubectl cluster-info --request-timeout=60s &> /dev/null; then
    echo -e "${RED}❌ Cluster not responding. Try: minikube delete --purge && ./deploy.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Cluster is ready${NC}"

# Enable addons
echo -e "\n${YELLOW}Enabling minikube addons...${NC}"
minikube addons enable ingress
minikube addons enable metrics-server

# Build image
echo -e "\n${YELLOW}Building Docker image in minikube...${NC}"
eval $(minikube docker-env)

if [ -f "${REPO_ROOT}/Dockerfile" ]; then
    docker build -t inference-sentinel:latest "${REPO_ROOT}"
else
    echo -e "${RED}❌ Dockerfile not found at ${REPO_ROOT}/Dockerfile${NC}"
    exit 1
fi

# Create namespace first
echo -e "\n${YELLOW}Creating namespace...${NC}"
kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"

# Create Grafana dashboards ConfigMap from observability directory
echo -e "\n${YELLOW}Creating Grafana dashboards ConfigMap...${NC}"
if [ -f "${DASHBOARDS_DIR}/overview.json" ] && [ -f "${DASHBOARDS_DIR}/controller.json" ]; then
    kubectl create configmap grafana-dashboards \
        --from-file="${DASHBOARDS_DIR}/overview.json" \
        --from-file="${DASHBOARDS_DIR}/controller.json" \
        -n inference-sentinel \
        --dry-run=client -o yaml | kubectl apply -f -
    echo -e "${GREEN}✅ Dashboards ConfigMap created${NC}"
else
    echo -e "${RED}❌ Dashboard JSON files not found in ${DASHBOARDS_DIR}${NC}"
    echo "   Expected: overview.json and controller.json"
    exit 1
fi

# Deploy with kustomize (this applies secrets.yaml with empty values)
echo -e "\n${YELLOW}Deploying all resources with kustomize...${NC}"
kubectl apply -k "${SCRIPT_DIR}"

# Override secrets with actual values from env vars (AFTER kustomize)
echo -e "\n${YELLOW}Updating secrets from environment variables...${NC}"
kubectl create secret generic sentinel-secrets \
    --from-literal=ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    --from-literal=GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
    -n inference-sentinel \
    --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✅ Secrets updated${NC}"

# Restart sentinel to pick up new secrets
kubectl rollout restart deployment sentinel -n inference-sentinel

# Wait for pods
echo -e "\n${YELLOW}Waiting for pods to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app=sentinel -n inference-sentinel --timeout=120s || true
kubectl wait --for=condition=ready pod -l app=prometheus -n inference-sentinel --timeout=60s || true
kubectl wait --for=condition=ready pod -l app=grafana -n inference-sentinel --timeout=60s || true
kubectl wait --for=condition=ready pod -l app=loki -n inference-sentinel --timeout=60s || true
kubectl wait --for=condition=ready pod -l app=tempo -n inference-sentinel --timeout=60s || true

# Get minikube IP
MINIKUBE_IP=$(minikube ip)

echo -e "\n${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "============================================"
echo "ACCESS"
echo "============================================"
echo ""
echo "Port-forward (recommended):"
echo "  kubectl port-forward -n inference-sentinel svc/sentinel 8000:8000 &"
echo "  kubectl port-forward -n inference-sentinel svc/grafana 3000:3000 &"
echo ""
echo "Then access:"
echo "  🌐 Sentinel API:  http://localhost:8000"
echo "  📊 Grafana:       http://localhost:3000 (admin/sentinel)"
echo ""
echo "Test:"
echo "  curl http://localhost:8000/health"
echo ""
echo "============================================"
echo "ALTERNATIVE: Ingress (requires tunnel)"
echo "============================================"
echo "Add to /etc/hosts:"
echo "  ${MINIKUBE_IP} sentinel.local grafana.local prometheus.local"
echo ""
echo "Run: minikube tunnel"
echo ""
echo "============================================"

# Show pod status
echo -e "\n${YELLOW}Pod status:${NC}"
kubectl get pods -n inference-sentinel

# Clear the trap on success
trap - ERR
