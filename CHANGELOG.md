# Changelog

All notable changes to inference-sentinel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 6: Kubernetes manifests, security hardening, CI/CD (planned)

---

## [0.5.0] - 2026-03-19

### Added
- **Session Stickiness**: PII-triggered session locking with context handoff
  - One-way state machine: `CLOUD_ELIGIBLE` → `LOCAL_LOCKED`
  - Session ID via `SHA256(client_ip + daily_salt)` with 24h rotation
  - Rolling buffer (5 turns / 4000 chars) for context preservation
  - Automatic handoff prompt injection on state transition
  - TTL-based session expiration (15 min default)
  - Response headers: `X-Sentinel-Session`, `X-Sentinel-Route`, `X-Sentinel-Backend`, `X-Sentinel-Tier`
- **Benchmark Experiment 5**: Session stickiness verification
- New session configuration via environment variables:
  - `SENTINEL_SESSION__ENABLED`
  - `SENTINEL_SESSION__TTL_SECONDS`
  - `SENTINEL_SESSION__LOCK_THRESHOLD_TIER`
  - `SENTINEL_SESSION__BUFFER_SIZE`
  - `SENTINEL_SESSION__CAPABILITY_GUARDRAIL`

### Fixed
- `finish_reason` normalization: Anthropic's `max_tokens` → `length`, `end_turn` → `stop`
- Google backend model default: `gemini-1.5-flash` → `gemini-2.0-flash`
- Controller datetime comparison error in `metrics_reader.py`
- Benchmark session isolation to prevent cross-experiment state leakage

### Changed
- Benchmark experiments now generate unique `X-Forwarded-For` headers per request
- `SentinelMetadata` schema extended with `session_state` and `session_locked_by_pii` fields

---

## [0.4.0] - 2026-03-15

### Added
- **Closed-Loop Controller**: Observes shadow metrics, recommends routing changes
  - Three modes: `observe`, `recommend`, `auto`
  - Configurable evaluation intervals and thresholds
  - Recommendation history tracking
  - Admin endpoints: `/admin/controller/status`, `/admin/controller/history`, `/admin/controller/evaluate`
- **Hot Reload**: Update routing config without restart via `POST /admin/reload`
- **Round-Robin Cloud Selection**: Load balance across Anthropic and Google backends
  - Strategies: `round_robin`, `primary_fallback`
  - Automatic failover on backend errors
- Controller Grafana dashboard

### Changed
- Backend manager refactored to support multiple cloud backends
- Shadow mode now tracks per-tier quality metrics

---

## [0.3.0] - 2026-03-10

### Added
- **Shadow Mode**: A/B comparison of local vs cloud inference
  - Parallel execution for Tier 0-1 requests
  - Semantic similarity scoring via `sentence-transformers`
  - Cost savings calculation
  - Admin endpoints: `/admin/shadow/metrics`, `/admin/shadow/results`
- **NER Classifier**: Named entity recognition for ambiguous PII
  - Model: `dslim/bert-base-NER`
  - Hybrid pipeline: regex fast-pass → NER fallback
- Full observability stack:
  - Prometheus metrics with detailed histograms
  - Grafana dashboards (Overview, Privacy, Cost)
  - Loki log aggregation
  - Tempo distributed tracing

### Changed
- Classification pipeline now two-stage (regex → NER)
- Docker Compose includes full observability stack

---

## [0.2.0] - 2026-03-05

### Added
- **Cloud Backends**: Anthropic Claude and Google Gemini adapters
- **Tier-Based Routing**: Route by privacy classification
  - Tier 0-1: Cloud eligible
  - Tier 2: Local by default
  - Tier 3: Local forced (no override)
- Backend health checking with automatic failover
- Routing configuration via `config/routing.yaml`
- Cost tracking per request

### Changed
- API response includes `sentinel` metadata block with routing details

---

## [0.1.0] - 2026-03-01

### Added
- Initial release
- **Privacy Classification**: 4-tier taxonomy (PUBLIC → RESTRICTED)
- **Regex Classifier**: High-confidence pattern matching for:
  - SSN, credit cards, bank accounts
  - Email, phone, addresses
  - API keys, AWS credentials, private keys
- **Ollama Backend**: Local inference via Ollama API
- FastAPI application with OpenAI-compatible endpoints:
  - `POST /v1/chat/completions`
  - `POST /v1/classify`
  - `GET /health`
  - `GET /metrics`
- Docker Compose with Ollama integration
- Basic structured logging

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.5.0 | 2026-03-19 | Session stickiness, context handoff |
| 0.4.0 | 2026-03-15 | Closed-loop controller, hot reload, round-robin |
| 0.3.0 | 2026-03-10 | Shadow mode, NER classifier, observability |
| 0.2.0 | 2026-03-05 | Cloud backends, tier-based routing |
| 0.1.0 | 2026-03-01 | Initial release, classification, Ollama |

[0.5.0]: https://github.com/kraghavan/inference-sentinel/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/kraghavan/inference-sentinel/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kraghavan/inference-sentinel/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/kraghavan/inference-sentinel/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/kraghavan/inference-sentinel/releases/tag/v0.1.0