# inference-sentinel Kubernetes Deployment

Deploy inference-sentinel to minikube with full observability stack.

> For general project documentation, see the [main README](../../README.md).

## Prerequisites

- [minikube](https://minikube.sigs.k8s.io/docs/start/)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Docker Desktop
- Ollama running on your Mac (port 11434)
- API keys: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`

## Quick Start

```bash
# From repo root
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."

./ops/k8s/deploy.sh
```

**What it does:**
1. Starts minikube (4 CPUs, 8GB RAM)
2. Builds Docker image inside minikube
3. Creates namespace `inference-sentinel`
4. Deploys all resources via Kustomize
5. Injects secrets from environment variables
6. Provisions Grafana dashboards
7. Waits for all pods to be ready

## Access Services

### Port-Forward (Recommended)

```bash
kubectl port-forward -n inference-sentinel svc/sentinel 8000:8000 &
kubectl port-forward -n inference-sentinel svc/grafana 3000:3000 &
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Sentinel API | http://localhost:8000 | - |
| Grafana | http://localhost:3000 | admin / sentinel |
| Prometheus | http://localhost:9090 | - |

### Ingress (Alternative)

```bash
# Add to /etc/hosts
echo "$(minikube ip) sentinel.local grafana.local prometheus.local" | sudo tee -a /etc/hosts

# Start tunnel (keep running)
minikube tunnel
```

Then access via `http://sentinel.local`, etc.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         minikube cluster                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Ingress   │────▶│  sentinel   │────▶│ Prometheus  │       │
│   │   (nginx)   │     │   :8000     │     │   :9090     │       │
│   └─────────────┘     └──────┬──────┘     └─────────────┘       │
│                              │                                    │
│         ┌────────────────────┼────────────────────┐              │
│         │                    │                    │              │
│         ▼                    ▼                    ▼              │
│   ┌───────────┐       ┌───────────┐       ┌───────────┐         │
│   │   Tempo   │       │   Loki    │       │  Grafana  │         │
│   │   :4317   │       │   :3100   │       │   :3000   │         │
│   └───────────┘       └───────────┘       └───────────┘         │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                │ host.minikube.internal:11434
                                ▼
                    ┌───────────────────────┐
                    │   Mac Mini M4 Ollama  │
                    │   gemma3:4b + mistral │
                    └───────────────────────┘
```

## Manifest Files

| File | Description |
|------|-------------|
| `namespace.yaml` | `inference-sentinel` namespace |
| `configmap.yaml` | All `SENTINEL_*` environment variables |
| `secrets.yaml` | Placeholder (overwritten by deploy.sh) |
| `sentinel.yaml` | Gateway deployment + service |
| `prometheus.yaml` | Metrics + RBAC for pod discovery |
| `grafana.yaml` | Dashboards + datasource provisioning |
| `grafana-dashboards-configmap.yaml` | Dashboard provider config |
| `loki.yaml` | Log aggregation |
| `tempo.yaml` | Distributed tracing (OTLP :4317) |
| `pvcs.yaml` | Persistent volumes (3.5GB total) |
| `ingress.yaml` | nginx ingress rules |
| `kustomization.yaml` | Kustomize resources list |

## Resource Usage

| Component | Memory Request | Memory Limit | CPU Request | CPU Limit |
|-----------|----------------|--------------|-------------|-----------|
| sentinel | 512Mi | 2Gi | 250m | 1000m |
| prometheus | 256Mi | 512Mi | 100m | 500m |
| grafana | 128Mi | 256Mi | 50m | 250m |
| loki | 128Mi | 256Mi | 50m | 250m |
| tempo | 128Mi | 256Mi | 50m | 250m |

**Total:** ~1.2GB requested, ~3.3GB limit

## Storage (PVCs)

| Volume | Size |
|--------|------|
| prometheus-data | 1Gi |
| grafana-data | 500Mi |
| loki-data | 1Gi |
| tempo-data | 1Gi |

**Total:** 3.5GB

## Monitoring with k9s

```bash
k9s -n inference-sentinel

# Useful commands:
# :pods      - List pods
# :svc       - List services
# :deploy    - List deployments
# :events    - View events (debug errors)
# l          - View logs (on selected pod)
# d          - Describe resource
# Ctrl+k     - Kill pod
```

## Troubleshooting

### Pods not starting

```bash
kubectl describe pod -n inference-sentinel <pod-name>
kubectl logs -n inference-sentinel <pod-name>
```

### Can't reach Ollama

```bash
# Test from inside cluster
kubectl exec -it -n inference-sentinel deploy/sentinel -- \
  curl -s http://host.minikube.internal:11434/api/tags
```

### Secrets are empty

```bash
# Check secret values
kubectl get secret sentinel-secrets -n inference-sentinel \
  -o jsonpath='{.data.ANTHROPIC_API_KEY}' | base64 -d | wc -c

# If 0 bytes, delete and redeploy
kubectl delete secret sentinel-secrets -n inference-sentinel
ANTHROPIC_API_KEY="..." GOOGLE_API_KEY="..." ./ops/k8s/deploy.sh
```

### Grafana dashboards not showing

```bash
# Check ConfigMap has actual dashboard data
kubectl get configmap grafana-dashboards -n inference-sentinel -o jsonpath='{.data}' | jq 'keys'
# Should show: ["controller.json", "overview.json"]

# If it shows "_placeholder", recreate:
kubectl delete configmap grafana-dashboards -n inference-sentinel
kubectl create configmap grafana-dashboards \
  --from-file=../../observability/grafana/dashboards/overview.json \
  --from-file=../../observability/grafana/dashboards/controller.json \
  -n inference-sentinel
kubectl rollout restart deployment grafana -n inference-sentinel
```

### Reset everything

```bash
# Delete namespace (removes everything)
kubectl delete namespace inference-sentinel

# Or delete and purge minikube entirely
minikube delete --purge
```

## Configuration

### Update ConfigMap values

```bash
kubectl edit configmap sentinel-config -n inference-sentinel
kubectl rollout restart deployment sentinel -n inference-sentinel
```

### Key settings in configmap.yaml

```yaml
SENTINEL_NER_ENABLED: "true"
SENTINEL_SHADOW_ENABLED: "true"
SENTINEL_CONTROLLER_ENABLED: "true"
SENTINEL_CONTROLLER_MODE: "observe"
SENTINEL_LOCAL_ENDPOINTS: "http://host.minikube.internal:11434/api/generate|gemma3:4b,http://host.minikube.internal:11434/api/generate|mistral"
```

## Benchmarking Against K8s

```bash
# Ensure port-forward is running
kubectl port-forward -n inference-sentinel svc/sentinel 8000:8000 &

# Run benchmarks
cd ~/path/to/inference-sentinel
python -m benchmarks.harness --generate --count 200 --experiment all --ner
python -m benchmarks.report
```
