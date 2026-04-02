# inference-sentinel Kubernetes Deployment

Deploy inference-sentinel to minikube with full observability stack.

## Prerequisites

- [minikube](https://minikube.sigs.k8s.io/docs/start/) installed
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- Docker installed
- Ollama running natively on your Mac Mini M4 (port 11434)

## Quick Start

### 1. Start minikube

```bash
# Start minikube with enough resources
minikube start --cpus=4 --memory=8192 --driver=docker

# Enable required addons
minikube addons enable ingress
minikube addons enable metrics-server
```

### 2. Build the Docker image

```bash
# Point docker to minikube's daemon
eval $(minikube docker-env)

# Build the image (from your inference-sentinel repo root)
docker build -t inference-sentinel:latest .

# Verify
docker images | grep inference-sentinel
```

### 3. Configure secrets

```bash
# Option A: Create secret from command line (recommended)
kubectl create namespace inference-sentinel

kubectl create secret generic sentinel-secrets \
  --from-literal=ANTHROPIC_API_KEY='sk-ant-your-key-here' \
  --from-literal=GOOGLE_API_KEY='your-google-key-here' \
  -n inference-sentinel

# Option B: Edit secrets.yaml and base64 encode your keys
# echo -n 'sk-ant-xxx' | base64
# Then: kubectl apply -f secrets.yaml
```

### 4. Deploy everything

```bash
# Deploy with kustomize
kubectl apply -k .

# Or apply individually
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvcs.yaml
kubectl apply -f prometheus.yaml
kubectl apply -f loki.yaml
kubectl apply -f tempo.yaml
kubectl apply -f grafana.yaml
kubectl apply -f sentinel.yaml
kubectl apply -f ingress.yaml
```

### 5. Configure /etc/hosts

```bash
# Get minikube IP
minikube ip

# Add to /etc/hosts (replace with your minikube IP)
echo "$(minikube ip) sentinel.local grafana.local prometheus.local" | sudo tee -a /etc/hosts
```

### 6. Verify deployment

```bash
# Check all pods are running
kubectl get pods -n inference-sentinel

# Expected output:
# NAME                          READY   STATUS    RESTARTS   AGE
# sentinel-xxx                  1/1     Running   0          1m
# prometheus-xxx                1/1     Running   0          1m
# grafana-xxx                   1/1     Running   0          1m
# loki-xxx                      1/1     Running   0          1m
# tempo-xxx                     1/1     Running   0          1m

# Check services
kubectl get svc -n inference-sentinel

# Check ingress
kubectl get ingress -n inference-sentinel
```

### 7. Test the gateway

```bash
# Health check
curl http://sentinel.local/health

# Test inference (PUBLIC tier - goes to cloud)
curl -X POST http://sentinel.local/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'

# Test inference (RESTRICTED tier - goes to local Ollama)
curl -X POST http://sentinel.local/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "My SSN is 123-45-6789, what should I do?"}'
```

## Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Sentinel API** | http://sentinel.local | - |
| **Grafana** | http://grafana.local | admin / sentinel |
| **Prometheus** | http://prometheus.local | - |

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

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl describe pod -n inference-sentinel <pod-name>

# Check logs
kubectl logs -n inference-sentinel <pod-name>
```

### Can't reach Ollama from sentinel pod

```bash
# Test connectivity from inside the cluster
kubectl exec -it -n inference-sentinel deploy/sentinel -- curl http://host.minikube.internal:11434/api/tags

# If that fails, try with host network (temporary debug)
# Or use minikube tunnel in another terminal
minikube tunnel
```

### Ingress not working

```bash
# Check ingress controller
kubectl get pods -n ingress-nginx

# Check ingress status
kubectl describe ingress -n inference-sentinel sentinel-ingress

# Alternative: use port-forward
kubectl port-forward -n inference-sentinel svc/sentinel 8000:8000
```

### Reset everything

```bash
# Delete all resources
kubectl delete -k .

# Or delete namespace (removes everything)
kubectl delete namespace inference-sentinel

# Delete PVCs if you want fresh data
kubectl delete pvc --all -n inference-sentinel
```

## Resource Usage

| Component | Memory Request | Memory Limit | CPU Request | CPU Limit |
|-----------|----------------|--------------|-------------|-----------|
| sentinel | 512Mi | 2Gi | 250m | 1000m |
| prometheus | 256Mi | 512Mi | 100m | 500m |
| grafana | 128Mi | 256Mi | 50m | 250m |
| loki | 128Mi | 256Mi | 50m | 250m |
| tempo | 128Mi | 256Mi | 50m | 250m |

**Total:** ~1.2GB memory requested, ~3.3GB limit

## Storage (PVCs)

| Component | Size |
|-----------|------|
| prometheus-data | 1Gi |
| grafana-data | 500Mi |
| loki-data | 1Gi |
| tempo-data | 1Gi |

**Total:** 3.5GB (under 5GB as specified)

## Notes

- The sentinel image must be built inside minikube's Docker environment (`eval $(minikube docker-env)`)
- Ollama runs natively on your Mac and is accessed via `host.minikube.internal:11434`
- Cloud API keys are stored in a Kubernetes Secret
- All observability data persists across pod restarts via PVCs
- Default Grafana credentials: `admin` / `sentinel`
