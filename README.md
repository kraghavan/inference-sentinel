# inference-sentinel

**Privacy-aware LLM routing gateway with production-grade observability**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

inference-sentinel automatically routes LLM prompts to **local** or **cloud** inference based on privacy classification. Sensitive data stays on your hardware; safe queries go to faster cloud providers.

```
Your App ──▶ inference-sentinel ──▶ [Privacy Check] ──▶ Local (Ollama) or Cloud (Claude/Gemini)
                                           │
                                           └──▶ Full observability (Prometheus/Grafana)
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- A model pulled: `ollama pull gemma3:4b`

### Installation

```bash
# Clone the repository
git clone https://github.com/kraghavan/inference-sentinel.git
cd inference-sentinel

# Install dependencies
pip install -e ".[dev]"

# Run the application
make run
```

### Test it

```bash
# Health check
curl http://localhost:8000/health

# Run inference
curl -X POST http://localhost:8000/v1/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

### Docker (Full Stack with Observability)

```bash
# Start everything (sentinel + ollama + prometheus + grafana)
make docker-up

# View logs
make docker-logs-all

# Access Grafana at http://localhost:3000 (admin/sentinel)
```

## Configuration

Edit `config/settings.yaml` or use environment variables:

```yaml
local:
  endpoints:
    - name: mac-mini
      host: localhost
      port: 11434
      model: "gemma3:4b"
      priority: 1
```

See [.env.example](.env.example) for all options.

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 - Foundation | ✅ | FastAPI skeleton, Ollama backend, Docker Compose |
| 1 - Classification | 🔜 | Privacy taxonomy, regex classifier |
| 2 - Routing | 🔜 | Tier-based routing, cloud backends |
| 3 - Observability | 🔜 | Full telemetry, Grafana dashboards |
| 4 - Controller | 🔜 | Closed-loop threshold adaptation |
| 5 - Benchmarks | 🔜 | Reproducible experiments |

## License

MIT

---

See [DESIGN_SPEC.md](docs/DESIGN_SPEC.md) for full architecture documentation.
