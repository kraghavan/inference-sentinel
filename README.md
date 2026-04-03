# inference-sentinel

**Privacy-aware LLM routing gateway** that classifies prompts by sensitivity and routes to local (Ollama) or cloud (Claude/Gemini) backends.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

inference-sentinel is a smart routing layer that sits between your application and LLM backends. It:

1. **Classifies** incoming prompts into 4 privacy tiers (PUBLIC → RESTRICTED)
2. **Routes** sensitive requests to local Ollama, non-sensitive to cloud APIs
3. **Tracks** cost savings from local inference
4. **Learns** via shadow mode comparison and closed-loop optimization

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client    │────▶│ inference-sentinel│────▶│  Local (Ollama) │
│ Application │     │    Classification │     │  gemma3:4b      │
└─────────────┘     │    + Routing      │     │  mistral        │
                    └────────┬──────────┘     └─────────────────┘
                             │
                             │ Tier 0-1 only
                             ▼
                    ┌─────────────────┐
                    │  Cloud APIs     │
                    │  Claude/Gemini  │
                    └─────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **4-Tier Privacy Classification** | PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED |
| **Hybrid Detection** | Regex fast-path + BERT NER for ambiguous cases |
| **Session Stickiness** | PII-triggered session locking with context handoff |
| **Shadow Mode** | A/B comparison of local vs cloud quality |
| **Closed-Loop Controller** | Auto-adjusts routing based on quality metrics |
| **Full Observability** | Prometheus, Grafana, Loki, Tempo |

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- [Ollama](https://ollama.ai/) running locally with models:
  ```bash
  ollama pull gemma3:4b
  ollama pull mistral
  ```
- API keys for cloud backends:
  - `ANTHROPIC_API_KEY` (Claude)
  - `GOOGLE_API_KEY` (Gemini)

---

## Deployment Options

### Option 1: Docker Compose (Recommended for Development)

```bash
# Clone the repo
git clone https://github.com/kraghavan/inference-sentinel.git
cd inference-sentinel

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."

# Deploy
./deploy-docker.sh

# Or with clean rebuild
./deploy-docker.sh --clean --rebuild
```

**Flags:**
| Flag | Description |
|------|-------------|
| `--clean` | Remove volumes, clear Prometheus data |
| `--rebuild` | Force no-cache Docker build |
| `--quick` | Skip health checks |

**Access:**
| Service | URL | Credentials |
|---------|-----|-------------|
| Sentinel API | http://localhost:8000 | - |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |

---

### Option 2: Kubernetes (minikube)

Best for testing production-like deployments on your local machine.

```bash
cd inference-sentinel

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."

# Deploy to minikube (starts cluster if needed)
./ops/k8s/deploy.sh
```

**What it does:**
1. Starts minikube (4 CPUs, 8GB RAM)
2. Builds Docker image inside minikube
3. Deploys sentinel + full observability stack
4. Configures secrets from environment variables
5. Provisions Grafana dashboards

**Access via port-forward:**
```bash
kubectl port-forward -n inference-sentinel svc/sentinel 8000:8000 &
kubectl port-forward -n inference-sentinel svc/grafana 3000:3000 &
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Sentinel API | http://localhost:8000 | - |
| Grafana | http://localhost:3000 | admin / sentinel |
| Prometheus | http://localhost:9090 | - |

**Monitoring with k9s:**
```bash
k9s -n inference-sentinel
```

See [ops/k8s/README.md](ops/k8s/README.md) for detailed K8s documentation.

---

## Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Inference (Auto-Routing)

```bash
# PUBLIC tier → routes to cloud
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is the capital of France?"}]}'

# RESTRICTED tier → routes to local Ollama
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"My SSN is 123-45-6789"}]}'
```

### Classification Only

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Send payment to account 1234567890"}'
```

### Response Format

```json
{
  "id": "sentinel-abc123",
  "choices": [{
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "sentinel": {
    "tier": 3,
    "tier_label": "RESTRICTED",
    "endpoint": "local",
    "backend": "gemma",
    "entities_detected": ["ssn"],
    "classification_latency_ms": 0.12,
    "session_state": "LOCAL_LOCKED"
  }
}
```

## Privacy Tiers

| Tier | Label | Examples | Routing |
|------|-------|----------|---------|
| 0 | PUBLIC | General questions, public info | Cloud |
| 1 | INTERNAL | Project codes, internal URLs | Cloud |
| 2 | CONFIDENTIAL | Email, phone, address | Local (default) |
| 3 | RESTRICTED | SSN, credit card, health records | Local (forced) |

## Architecture

```
src/sentinel/
├── main.py                 # FastAPI application
├── classification/
│   ├── hybrid.py           # Regex + NER pipeline
│   ├── patterns.py         # Regex patterns
│   └── ner_classifier.py   # BERT NER model
├── backends/
│   ├── manager.py          # Backend orchestration
│   ├── ollama.py           # Local inference
│   ├── anthropic.py        # Claude API
│   └── google.py           # Gemini API
├── session/
│   └── manager.py          # Session stickiness
├── shadow/
│   └── runner.py           # A/B comparison
├── controller/
│   └── __init__.py         # Closed-loop optimization
└── telemetry/
    └── __init__.py         # Logging & tracing
```

## Observability

### Grafana Dashboards

| Dashboard | Description |
|-----------|-------------|
| **Overview** | Request rates, latency, routing distribution, cost savings |
| **Controller** | Shadow comparison, similarity scores, recommendations |

### Key Metrics

| Metric | Description |
|--------|-------------|
| `sentinel_requests_total` | Total requests by tier and endpoint |
| `sentinel_classification_seconds` | Classification latency histogram |
| `sentinel_inference_seconds` | End-to-end latency histogram |
| `sentinel_cost_dollars` | Estimated API costs |
| `sentinel_shadow_similarity` | Local vs cloud quality score |

## Benchmarking

```bash
# Generate test data and run all experiments
python -m benchmarks.harness --generate --count 200 --experiment all --ner

# Generate markdown report with charts
python -m benchmarks.report

# Individual experiments:
python -m benchmarks.harness --experiment classification --ner
python -m benchmarks.harness --experiment routing
python -m benchmarks.harness --experiment cost
python -m benchmarks.harness --experiment controller
python -m benchmarks.harness --experiment session
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTINEL_NER_ENABLED` | `true` | Enable BERT NER classifier |
| `SENTINEL_NER_MODEL` | `fast` | NER model: fast, accurate, multilingual |
| `SENTINEL_SHADOW_ENABLED` | `true` | Enable shadow mode A/B comparison |
| `SENTINEL_CONTROLLER_ENABLED` | `true` | Enable closed-loop controller |
| `SENTINEL_CONTROLLER_MODE` | `observe` | Controller mode: observe, recommend, auto |
| `SENTINEL_SESSION__ENABLED` | `true` | Enable session stickiness |
| `SENTINEL_SESSION__TTL_SECONDS` | `900` | Session timeout (15 min) |
| `SENTINEL_LOCAL_ENDPOINTS` | `gemma,mistral` | Ollama models to use |
| `SENTINEL_CLOUD_PRIMARY` | `anthropic` | Primary cloud backend |
| `SENTINEL_CLOUD_FALLBACK` | `google` | Fallback cloud backend |

### Routing Config

Edit `config/routing.yaml`:

```yaml
tiers:
  0:  # PUBLIC
    allow_cloud: true
    prefer_local: false
  1:  # INTERNAL
    allow_cloud: true
    prefer_local: false
  2:  # CONFIDENTIAL
    allow_cloud: false
    prefer_local: true
  3:  # RESTRICTED
    allow_cloud: false
    prefer_local: true
    force_local: true
```

## Project Structure

```
inference-sentinel/
├── src/sentinel/           # Main application code
├── benchmarks/             # Benchmark harness & reports
├── config/                 # Routing configuration
├── observability/
│   └── grafana/
│       └── dashboards/     # Grafana dashboard JSON
├── ops/
│   └── k8s/                # Kubernetes manifests
│       ├── deploy.sh       # K8s deployment script
│       ├── kustomization.yaml
│       └── *.yaml          # K8s resources
├── deploy-docker.sh        # Docker Compose deployment
├── docker-compose.yml
└── Dockerfile
```

## Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev,ner]"

# Run tests
pytest

# Run locally (development mode)
uvicorn sentinel.main:app --reload
```

## Roadmap

- [x] Phase 1: Privacy classification (v0.1.0)
- [x] Phase 2: Cloud backends & routing (v0.2.0)
- [x] Phase 3: Shadow mode & NER (v0.3.0)
- [x] Phase 4: Closed-loop controller (v0.4.0)
- [x] Phase 5: Session stickiness (v0.5.0)
- [x] Phase 6: Kubernetes deployment (v0.6.0)
- [ ] Phase 7: Helm charts, production hardening

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Karthika Raghavan** - [LinkedIn](https://linkedin.com/in/karthikaraghavan) | [GitHub](https://github.com/kraghavan)
