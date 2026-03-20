# inference-sentinel

## Design Specification v0.1

**A privacy-aware LLM routing gateway with production-grade observability**

---

## Table of Contents

1. [Vision & Positioning](#1-vision--positioning)
2. [Privacy Taxonomy](#2-privacy-taxonomy)
3. [Routing Engine](#3-routing-engine)
4. [Session Stickiness](#4-session-stickiness)
5. [Closed-Loop Controller](#5-closed-loop-controller)
6. [Telemetry Schema](#6-telemetry-schema)
7. [System Architecture](#7-system-architecture)
8. [Benchmark Methodology](#8-benchmark-methodology)
9. [Phased Roadmap](#9-phased-roadmap)
10. [Repository Structure](#10-repository-structure)

---

## 1. Vision & Positioning

### 1.1 Problem Statement

Organizations using LLMs face a tension:

- **Cloud LLMs** offer superior quality and speed but require sending data externally
- **Local LLMs** preserve privacy but sacrifice capability and require infrastructure expertise
- **Current solutions** force a binary choice or require manual intervention

No existing system provides **automatic, real-time, privacy-aware routing with cost/latency observability**.

### 1.2 What inference-sentinel Does

inference-sentinel acts as an intelligent proxy between applications and LLM backends. For every request, it:

1. **Classifies** the prompt for sensitive data (PII, secrets, proprietary content)
2. **Routes** to local or cloud inference based on privacy rules + performance thresholds
3. **Observes** the entire request lifecycle with production-grade telemetry
4. **Adapts** routing thresholds based on observed performance (closed-loop)

### 1.3 Differentiation

| Existing Tool | What It Does | What It Lacks |
|---------------|--------------|---------------|
| Presidio / Private AI | PII detection + redaction | No routing, no inference |
| LiteLLM / Portkey | Multi-provider routing | Privacy-blind, no local path |
| Ollama / vLLM | Local inference | No routing intelligence |
| LangSmith / OpenLLMetry | LLM observability | No hardware cost modeling |

**inference-sentinel combines all four**: classification → routing → inference → observability.

### 1.4 Target Audience

- **Primary**: Infrastructure/platform engineers evaluating LLM deployment strategies
- **Secondary**: Security-conscious teams needing auditable AI usage
- **Tertiary**: Researchers benchmarking local vs. cloud tradeoffs

### 1.5 Success Metrics (for the project itself)

- Reproducible benchmark results with documented methodology
- Grafana dashboard demonstrating cost savings quantification
- Clean abstraction allowing new backends in <100 LOC
- Sub-50ms routing decision overhead (p99)
- ITL collection with <1ms measurement overhead
- TTFT/ITL/TPOT comparison data for Llama vs Gemma on M4 vs M1

---

## 2. Privacy Taxonomy

### 2.1 Classification Tiers

Prompts are classified into **four privacy tiers**, each with default routing behavior:

| Tier | Label | Description | Default Route | Override Allowed |
|------|-------|-------------|---------------|------------------|
| 0 | `PUBLIC` | No sensitive content detected | Cloud | Yes |
| 1 | `INTERNAL` | Business context, non-regulated | Cloud (configurable) | Yes |
| 2 | `CONFIDENTIAL` | PII, credentials, proprietary data | Local | Config-dependent |
| 3 | `RESTRICTED` | Regulated data (PHI, financial PII) | Local (forced) | No |

### 2.2 Entity Types & Tier Mapping

```yaml
privacy_taxonomy:
  # Tier 3 - RESTRICTED (always local, no override)
  restricted:
    - ssn              # Social Security Number
    - credit_card      # Full card numbers
    - bank_account     # Account + routing numbers
    - health_record    # PHI under HIPAA
    - biometric        # Fingerprints, face geometry
    - government_id    # Passport, driver's license numbers

  # Tier 2 - CONFIDENTIAL (local by default, configurable)
  confidential:
    - email            # Email addresses
    - phone            # Phone numbers
    - address          # Physical addresses
    - dob              # Date of birth
    - api_key          # API keys, tokens
    - password         # Passwords, secrets
    - private_key      # Cryptographic keys
    - aws_credentials  # Cloud provider creds
    - ip_address       # Internal IP addresses

  # Tier 1 - INTERNAL (cloud allowed, logged)
  internal:
    - employee_name    # Internal personnel names
    - project_code     # Internal project identifiers
    - revenue_figure   # Unannounced financials
    - customer_name    # Customer/client names
    - internal_url     # Intranet URLs

  # Tier 0 - PUBLIC (cloud, minimal logging)
  public:
    - general_knowledge
    - public_documentation
    - open_source_code
```

### 2.3 Detection Strategy

**Hybrid approach** to balance speed and accuracy:

```
┌─────────────────────────────────────────────────────────────┐
│                    DETECTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input ──▶ [Regex Fast-Pass] ──▶ [Confidence Check] ──▶ Route
│                   │                      │                  │
│                   │ HIGH confidence      │ LOW/MEDIUM       │
│                   │ (SSN, CC patterns)   │ confidence       │
│                   │                      │                  │
│                   ▼                      ▼                  │
│            Direct Tier 3          [NER Model Pass]          │
│                                          │                  │
│                                          ▼                  │
│                                   Final Classification      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Stage 1: Regex Fast-Pass** (~1-5ms)

High-precision patterns for unambiguous entities:

```python
PATTERNS = {
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b",
    "api_key": r"\b(?:sk-[a-zA-Z0-9]{32,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36})\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "aws_secret_key": r"\b[A-Za-z0-9/+=]{40}\b",  # Context-dependent
}
```

**Stage 2: NER Model Pass** (~20-50ms, optional)

For ambiguous cases (names, addresses, context-dependent secrets):

- **Model**: `dslim/bert-base-NER` or `microsoft/presidio-analyzer`
- **Trigger**: When regex confidence < threshold OR entity type requires context
- **Optimization**: Only invoked for Tier 1-2 candidates, skipped for clear Tier 0/3

### 2.4 Classification Output Schema

```python
@dataclass
class ClassificationResult:
    tier: int                           # 0-3
    tier_label: str                     # PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
    entities_detected: list[Entity]     # List of found entities
    confidence: float                   # 0.0-1.0 overall confidence
    detection_method: str               # "regex", "ner", "hybrid"
    detection_latency_ms: float         # Time taken to classify
    
@dataclass
class Entity:
    entity_type: str                    # e.g., "ssn", "email"
    value_hash: str                     # SHA-256 of detected value (never log raw)
    start_pos: int                      # Character position in prompt
    end_pos: int
    confidence: float                   # Detection confidence
    tier: int                           # Entity's tier
```

---

## 3. Routing Engine

### 3.1 Decision Flow

```
┌───────────────────────────────────────────────────────────────────────┐
│                         ROUTING DECISION FLOW                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ClassificationResult                                                 │
│         │                                                             │
│         ▼                                                             │
│  ┌─────────────────┐                                                  │
│  │ Tier == 3?      │──YES──▶ ROUTE: Local (forced, no override)       │
│  └────────┬────────┘                                                  │
│           │ NO                                                        │
│           ▼                                                           │
│  ┌─────────────────┐                                                  │
│  │ Tier == 2?      │──YES──▶ Check: local_override_allowed?           │
│  └────────┬────────┘              │                                   │
│           │ NO                    ├─ NO ──▶ ROUTE: Local              │
│           │                       └─ YES ─▶ [SLO Check]               │
│           ▼                                                           │
│  ┌─────────────────┐                                                  │
│  │ Tier == 0 or 1  │──YES──▶ [SLO Check]                              │
│  └─────────────────┘                                                  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │                        SLO CHECK                              │    │
│  │                                                               │    │
│  │  Local backend healthy?                                       │    │
│  │    └─ NO ──▶ ROUTE: Cloud (with privacy_override_logged=true) │    │
│  │    └─ YES ─▶ Continue                                         │    │
│  │                                                               │    │
│  │  A/B shadow mode enabled?                                     │    │
│  │    └─ YES ─▶ ROUTE: Both (compare)                            │    │
│  │    └─ NO ──▶ Continue                                         │    │
│  │                                                               │    │
│  │  Estimated local latency > latency_threshold?                 │    │
│  │    └─ YES AND Tier <= 1 ──▶ ROUTE: Cloud                      │    │
│  │    └─ NO ───────────────▶ ROUTE: Local                        │    │
│  │                                                               │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 3.2 Routing Configuration

```yaml
# config/routing.yaml

routing:
  # Default routing behavior per tier
  tier_defaults:
    0: cloud      # PUBLIC
    1: cloud      # INTERNAL  
    2: local      # CONFIDENTIAL
    3: local      # RESTRICTED (immutable)

  # Override permissions
  overrides:
    tier_2_cloud_allowed: false          # Can Tier 2 ever go to cloud?
    tier_1_local_preferred: false        # Should Tier 1 prefer local?
    
  # SLO thresholds (used by closed-loop controller)
  slo:
    max_ttft_ms: 2000                    # Time to first token
    max_itl_p95_ms: 50                   # Inter-token latency (decode)
    max_tpot_ms: 30                      # Time per output token
    max_total_latency_ms: 30000          # Total generation time
    max_cost_per_1k_tokens: 0.01         # USD
    min_local_availability: 0.95         # Local backend uptime

  # A/B shadow mode
  shadow_mode:
    enabled: false
    shadow_percentage: 10                # % of requests to dual-route
    compare_metrics:
      - ttft
      - itl                              # Decode performance comparison
      - tpot
      - total_latency  
      - tokens_per_second
      - output_similarity                # Semantic similarity of outputs

  # Backend selection within route type
  cloud_backends:
    primary: anthropic
    fallback: google
    selection_strategy: cost             # cost | latency | round_robin
    
  local_backends:
    # Multi-endpoint configuration for distributed local inference
    endpoints:
      - name: mac-mini
        host: 192.168.1.10               # Update with actual IP
        port: 11434
        model: "llama3.2:8b-instruct-q4_K_M"
        priority: 1                       # Lower = higher priority
        hardware:
          chip: "Apple M4"
          memory_gb: 16
          
      - name: macbook
        host: 192.168.1.20               # Update with actual IP
        port: 11434
        model: "gemma2:9b-instruct-q4_K_M"
        priority: 2
        hardware:
          chip: "Apple M1"
          memory_gb: 16
          
    selection_strategy: latency_best     # latency_best | round_robin | priority | model_affinity
    health_check_interval_seconds: 30
    failover_enabled: true
    
    # Model affinity rules (optional)
    model_affinity:
      # Route specific request types to specific models
      code_generation: "llama3.2:8b-instruct-q4_K_M"
      general_chat: "gemma2:9b-instruct-q4_K_M"
```

### 3.3 Routing Output Schema

```python
@dataclass
class RoutingDecision:
    route: Literal["local", "cloud", "both"]
    backend: str                          # "ollama", "anthropic", "google"
    model: str                            # Specific model identifier
    reason: str                           # Human-readable decision reason
    privacy_tier: int
    slo_override: bool                    # True if SLO overrode privacy preference
    shadow_request: bool                  # True if this is a shadow/comparison request
    decision_latency_ms: float
    
    # Audit fields
    classification_result: ClassificationResult
    config_snapshot: dict                 # Routing config at decision time
```

---

## 4. Session Stickiness

### 4.1 Purpose

Session stickiness ensures **conversational continuity** when PII is detected mid-conversation. Once a session encounters sensitive data (Tier 2+), it permanently locks to local inference for the remainder of the session.

**Problem solved**: Without session stickiness, a multi-turn conversation could alternate between cloud and local backends based on each message's classification, causing:
- Context fragmentation (local model doesn't know what cloud discussed)
- Privacy leakage (cloud sees follow-up questions referencing PII)
- Inconsistent user experience

### 4.2 State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    SESSION STATE MACHINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         PII Detected          ┌──────────────────┐
│  │                  │         (Tier >= 2)           │                  │
│  │  CLOUD_ELIGIBLE  │ ─────────────────────────────▶│  LOCAL_LOCKED    │
│  │                  │                               │                  │
│  └──────────────────┘                               └──────────────────┘
│         │                                                   │
│         │ No PII                                            │ All requests
│         ▼                                                   ▼
│    Route to Cloud                                    Route to Local
│    (or Local for Tier 3)                            (permanently)
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ ONE-WAY TRAPDOOR: Sessions NEVER unlock back to cloud     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Session Identification

```python
# Session ID generation (deterministic, privacy-preserving)
session_id = SHA256(client_ip + daily_salt)

# Daily salt rotation
salt = HMAC_SHA256(server_secret, date.today().isoformat())
```

**Properties:**
- Same client gets same session ID within a day
- Different clients cannot correlate sessions
- Salt rotation provides forward secrecy

### 4.4 Context Buffer & Handoff

When a session transitions from `CLOUD_ELIGIBLE` to `LOCAL_LOCKED`, the gateway performs a **context handoff**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      CONTEXT HANDOFF                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Rolling Buffer (last 5 turns OR 4000 chars, whichever first)   │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ <context_handoff>                                        │    │
│  │   <turn role="user">What's the weather like?</turn>      │    │
│  │   <turn role="assistant">It's sunny today...</turn>      │    │
│  │   <turn role="user">My SSN is 123-45-6789</turn>         │    │
│  │   <notice>Content scrubbed for privacy</notice>          │    │
│  │ </context_handoff>                                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                        │
│         ▼                                                        │
│  Injected as system message to local model                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Buffer constraints:**
- `buffer_max_turns`: 5 (configurable)
- `buffer_max_chars`: 4000 (configurable)
- Oldest content evicted when limits exceeded

### 4.5 Configuration

```yaml
session:
  enabled: true
  ttl_seconds: 900              # 15 min inactivity → purge
  lock_threshold_tier: 2        # Tier 2+ triggers lock
  buffer_size: 5                # Max turns in buffer
  buffer_max_chars: 4000        # Max characters in buffer
  capability_guardrail: true    # Prevent cloud lockout for Tier 3
```

**Environment variables:**
```bash
SENTINEL_SESSION__ENABLED=true
SENTINEL_SESSION__TTL_SECONDS=900
SENTINEL_SESSION__LOCK_THRESHOLD_TIER=2
SENTINEL_SESSION__BUFFER_SIZE=5
SENTINEL_SESSION__CAPABILITY_GUARDRAIL=true
```

### 4.6 Response Headers

Every response includes session metadata:

```
X-Sentinel-Session: abc123def456...   # Truncated session ID
X-Sentinel-Route: local               # Current route
X-Sentinel-Backend: ollama            # Backend used
X-Sentinel-Tier: 3                    # Privacy tier
```

### 4.7 Implementation

```
src/sentinel/session/
├── __init__.py          # Module exports
├── salt.py              # Daily salt rotation, session ID generation
├── buffer.py            # RollingBuffer, content scrubbing, handoff prompt
└── manager.py           # SessionManager, state machine, TTL cache
```

---

## 5. Closed-Loop Controller

### 4.1 Purpose

The closed-loop controller **automatically adjusts routing thresholds** based on observed performance. This prevents:

- Routing to a degraded local backend
- Over-reliance on expensive cloud during low-sensitivity periods
- SLO violations from static thresholds

### 4.2 Control Loop Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP CONTROLLER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Telemetry   │───▶│  Aggregator  │───▶│  Analyzer    │      │
│  │  (per req)   │    │  (5m window) │    │  (trends)    │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐       │
│                                          │  Controller  │       │
│                                          │  (PID-like)  │       │
│                                          └──────┬───────┘       │
│                                                 │               │
│           ┌─────────────────────────────────────┼───────────┐   │
│           │                                     │           │   │
│           ▼                                     ▼           ▼   │
│  ┌─────────────────┐               ┌─────────────────┐  ┌─────┐│
│  │ Adjust latency  │               │ Adjust backend  │  │Alert││
│  │ thresholds      │               │ weights         │  │     ││
│  └─────────────────┘               └─────────────────┘  └─────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Metrics Watched

```python
@dataclass
class ControllerMetrics:
    # Latency metrics (per backend, rolling 5m window)
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_p50_ms: float                  # Inter-token latency
    itl_p95_ms: float
    itl_p99_ms: float
    tpot_p50_ms: float                 # Time per output token
    tpot_p95_ms: float
    total_latency_p50_ms: float
    total_latency_p95_ms: float
    
    # Throughput metrics
    tokens_per_second: float
    requests_per_minute: float
    
    # Availability metrics
    success_rate: float                # 1 - (errors / total)
    timeout_rate: float
    
    # Cost metrics
    cost_per_1k_tokens: float
    total_cost_window: float           # Cost in current window
    
    # Quality metrics (when shadow mode enabled)
    output_similarity_mean: float      # vs. cloud baseline
```

### 4.4 Adjustment Algorithm

```python
class ClosedLoopController:
    """
    Simplified PID-like controller for routing thresholds.
    
    Key principles:
    1. Dampen oscillations - don't overreact to short-term spikes
    2. Gradual adjustments - max 10% change per interval
    3. Safety bounds - never exceed configured min/max thresholds
    4. Hysteresis - require sustained deviation before adjusting
    """
    
    def __init__(self, config: ControllerConfig):
        self.config = config
        self.history = deque(maxlen=12)  # 1 hour at 5m intervals
        
    def evaluate(self, metrics: ControllerMetrics) -> list[Adjustment]:
        adjustments = []
        
        # Check local backend health
        if metrics.success_rate < self.config.min_local_availability:
            if self._sustained_below(metrics.success_rate, 
                                     self.config.min_local_availability,
                                     periods=3):  # 15 minutes sustained
                adjustments.append(Adjustment(
                    parameter="local_backend_weight",
                    direction="decrease",
                    magnitude=0.1,
                    reason=f"Local success rate {metrics.success_rate:.2%} below threshold"
                ))
        
        # Check latency drift (TTFT)
        ttft_ratio = metrics.ttft_p95_ms / self.config.slo.max_ttft_ms
        if ttft_ratio > 1.2:  # 20% above SLO
            adjustments.append(Adjustment(
                parameter="latency_threshold_ms",
                direction="decrease",  # Be more aggressive about cloud routing
                magnitude=min(0.1, (ttft_ratio - 1.0) * 0.5),
                reason=f"TTFT p95 {metrics.ttft_p95_ms}ms exceeds SLO"
            ))
        elif ttft_ratio < 0.5:  # Plenty of headroom
            adjustments.append(Adjustment(
                parameter="latency_threshold_ms", 
                direction="increase",  # Can be more aggressive about local
                magnitude=0.05,
                reason=f"TTFT p95 {metrics.ttft_p95_ms}ms well under SLO"
            ))
        
        # Check decode performance (ITL)
        itl_ratio = metrics.itl_p95_ms / self.config.slo.max_itl_p95_ms
        if itl_ratio > 1.2:  # 20% above SLO
            adjustments.append(Adjustment(
                parameter="endpoint_weight",
                direction="decrease",  # Reduce load on slow decoder
                magnitude=min(0.1, (itl_ratio - 1.0) * 0.5),
                reason=f"ITL p95 {metrics.itl_p95_ms}ms exceeds SLO - decode bottleneck"
            ))
            
        # Check cost efficiency
        if metrics.cost_per_1k_tokens > self.config.slo.max_cost_per_1k_tokens:
            adjustments.append(Adjustment(
                parameter="cloud_routing_threshold",
                direction="increase",  # Route more to local
                magnitude=0.1,
                reason=f"Cost ${metrics.cost_per_1k_tokens}/1k exceeds budget"
            ))
            
        return self._apply_dampening(adjustments)
    
    def _sustained_below(self, current: float, threshold: float, periods: int) -> bool:
        """Check if metric has been below threshold for N periods."""
        if len(self.history) < periods:
            return False
        return all(h < threshold for h in list(self.history)[-periods:])
    
    def _apply_dampening(self, adjustments: list[Adjustment]) -> list[Adjustment]:
        """Prevent oscillations by limiting adjustment frequency and magnitude."""
        dampened = []
        for adj in adjustments:
            # Check if we adjusted this parameter recently
            recent_adjustments = [
                a for a in self.adjustment_history 
                if a.parameter == adj.parameter 
                and a.timestamp > now() - timedelta(minutes=15)
            ]
            if len(recent_adjustments) >= 2:
                continue  # Skip - adjusted too recently
                
            # Cap magnitude
            adj.magnitude = min(adj.magnitude, 0.1)
            dampened.append(adj)
            
        return dampened
```

### 4.5 Controller Configuration

```yaml
# config/controller.yaml

controller:
  enabled: true
  evaluation_interval_seconds: 300      # 5 minutes
  
  # Dampening
  min_adjustment_interval_seconds: 900  # 15 min between same-param adjustments
  max_adjustment_magnitude: 0.10        # 10% max change per interval
  sustained_periods_required: 3         # 15 min of deviation before acting
  
  # Safety bounds
  bounds:
    latency_threshold_ms:
      min: 500
      max: 10000
    itl_threshold_ms:
      min: 10
      max: 100
    local_backend_weight:
      min: 0.0                          # Can fully disable local
      max: 1.0
    cloud_routing_threshold:
      min: 0.0
      max: 1.0
      
  # Alerting
  alerts:
    local_availability_critical: 0.80   # Page if below
    cost_budget_exceeded: 100.00        # USD per hour
    latency_slo_breach_rate: 0.05       # 5% of requests
```

---

## 6. Telemetry Schema

### 5.1 Design Principles

1. **OpenTelemetry-native**: All telemetry follows OTel semantic conventions
2. **Three pillars**: Metrics, traces, and logs are correlated
3. **Cost attribution**: Every request has computable cost
4. **Privacy-safe**: Never log raw PII, only hashes and classifications

### 5.1.1 Latency Metrics Definitions

| Metric | Formula | What It Captures |
|--------|---------|------------------|
| **TTFT** | `first_token_time - request_start` | Prefill latency (prompt processing) |
| **ITL** | `token[n]_time - token[n-1]_time` | Decode latency (per-token generation) |
| **TPOT** | `(total_time - ttft) / output_tokens` | Average decode throughput |
| **Total Latency** | `last_token_time - request_start` | End-to-end request time |

**Why track both TTFT and ITL:**

- **TTFT** dominated by: prompt length, KV cache state, model load time
- **ITL** dominated by: memory bandwidth, batch size, model architecture
- M4 vs M1 comparison: similar TTFT, but M4 has ~50% better memory bandwidth → lower ITL
- Network latency: adds to TTFT (single round-trip), minimal impact on ITL (streaming)

**ITL Collection Strategy:**

```python
# During streaming response
token_timestamps = []
async for token in stream:
    token_timestamps.append(time.perf_counter())
    yield token

# Calculate ITL distribution
itl_values = [
    token_timestamps[i] - token_timestamps[i-1] 
    for i in range(1, len(token_timestamps))
]
itl_p50 = np.percentile(itl_values, 50)
itl_p95 = np.percentile(itl_values, 95)
itl_p99 = np.percentile(itl_values, 99)
```

### 5.2 Trace Schema

Each request generates a trace with the following span structure:

```
inference_request (root span)
├── privacy_classification
│   ├── regex_detection
│   └── ner_detection (optional)
├── routing_decision
├── backend_execution
│   ├── request_preparation
│   ├── inference (local or cloud)
│   │   ├── time_to_first_token
│   │   ├── token_generation (repeated per token)
│   │   │   └── inter_token_latency (recorded per token pair)
│   │   └── stream_complete
│   └── response_parsing
├── itl_aggregation (compute p50/p95/p99 from token timestamps)
└── response_delivery
```

**Span Attributes (inference_request):**

```python
{
    # Request identification
    "request.id": "uuid",
    "request.timestamp": "ISO8601",
    
    # Privacy classification
    "privacy.tier": 2,
    "privacy.tier_label": "CONFIDENTIAL",
    "privacy.entities_detected_count": 3,
    "privacy.entity_types": ["email", "phone", "address"],
    "privacy.classification_method": "hybrid",
    "privacy.classification_latency_ms": 12.5,
    
    # Routing decision
    "routing.decision": "local",
    "routing.backend": "ollama",
    "routing.endpoint": "mac-mini",       # Which local endpoint was used
    "routing.endpoint_host": "192.168.1.10",
    "routing.model": "llama3.2:8b-instruct-q4_K_M",
    "routing.reason": "tier_2_confidential",
    "routing.slo_override": False,
    "routing.shadow_request": False,
    "routing.decision_latency_ms": 1.2,
    "routing.network_latency_ms": 2.5,    # Network hop to local endpoint
    
    # Inference metrics
    "inference.ttft_ms": 245.0,
    "inference.itl_p50_ms": 12.3,         # Inter-token latency (median)
    "inference.itl_p95_ms": 18.7,         # Inter-token latency (p95)
    "inference.itl_p99_ms": 25.1,         # Inter-token latency (p99)
    "inference.tpot_ms": 14.2,            # Time per output token (avg)
    "inference.total_latency_ms": 3420.0,
    "inference.tokens_prompt": 150,
    "inference.tokens_completion": 280,
    "inference.tokens_total": 430,
    "inference.tokens_per_second": 81.9,
    
    # Cost attribution
    "cost.estimated_usd": 0.0,          # Local = $0
    "cost.cloud_equivalent_usd": 0.012, # What cloud would have cost
    "cost.savings_usd": 0.012,
    
    # System metrics (local inference only)
    "system.memory_used_gb": 8.2,
    "system.memory_peak_gb": 9.1,
    "system.gpu_utilization_percent": 78.5,  # Metal utilization
    
    # Error tracking
    "error": False,
    "error.type": None,
    "error.message": None,
}
```

### 5.3 Metrics Schema

**Counter Metrics:**

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `sentinel_requests_total` | `tier`, `route`, `backend`, `status` | Total requests processed |
| `sentinel_tokens_total` | `direction` (prompt/completion), `route` | Total tokens processed |
| `sentinel_routing_decisions_total` | `tier`, `route`, `reason` | Routing decisions made |
| `sentinel_privacy_detections_total` | `entity_type`, `tier` | PII entities detected |

**Histogram Metrics:**

| Metric Name | Labels | Buckets | Description |
|-------------|--------|---------|-------------|
| `sentinel_ttft_seconds` | `route`, `backend`, `model`, `endpoint` | .1, .25, .5, 1, 2, 5, 10 | Time to first token |
| `sentinel_itl_seconds` | `route`, `backend`, `model`, `endpoint` | .005, .01, .015, .025, .05, .1 | Inter-token latency |
| `sentinel_tpot_seconds` | `route`, `backend`, `model`, `endpoint` | .005, .01, .02, .05, .1, .2 | Time per output token (avg) |
| `sentinel_latency_seconds` | `route`, `backend`, `model`, `endpoint` | .5, 1, 2, 5, 10, 30, 60 | Total request latency |
| `sentinel_classification_seconds` | `method` | .001, .005, .01, .05, .1 | Classification latency |
| `sentinel_tokens_per_second` | `route`, `backend`, `model`, `endpoint` | 10, 25, 50, 100, 200 | Generation speed |
| `sentinel_network_latency_seconds` | `endpoint` | .001, .005, .01, .025, .05, .1 | Network hop to local endpoint |

**Gauge Metrics:**

| Metric Name | Labels | Description |
|-------------|--------|-------------|
| `sentinel_local_backend_healthy` | `endpoint`, `host` | 1 if healthy, 0 if degraded |
| `sentinel_local_endpoint_latency_ms` | `endpoint` | Last observed network latency to endpoint |
| `sentinel_local_memory_used_bytes` | `endpoint` | Current memory usage per endpoint |
| `sentinel_local_gpu_utilization` | `endpoint` | Metal GPU utilization (0-1) |
| `sentinel_cost_savings_usd` | | Cumulative cost savings |
| `sentinel_controller_threshold` | `parameter` | Current controller thresholds |
| `sentinel_endpoint_requests_active` | `endpoint` | In-flight requests per endpoint |

### 5.4 Log Schema

**Structured JSON logs:**

```json
{
  "timestamp": "2025-03-13T10:23:45.123Z",
  "level": "INFO",
  "logger": "sentinel.router",
  "trace_id": "abc123",
  "span_id": "def456",
  "message": "Request routed to local backend",
  "attributes": {
    "request_id": "req_789",
    "privacy_tier": 2,
    "route": "local",
    "backend": "ollama",
    "model": "llama3.2:8b",
    "reason": "tier_2_confidential"
  }
}
```

**Log levels by event:**

| Event | Level | When |
|-------|-------|------|
| Request received | DEBUG | Always |
| Classification complete | DEBUG | Always |
| Routing decision | INFO | Always |
| Inference complete | INFO | Always |
| SLO override | WARN | When privacy is overridden for SLO |
| Backend error | ERROR | On failure |
| Controller adjustment | INFO | When thresholds change |
| Privacy tier 3 cloud attempt | ERROR | Should never happen |

---

## 7. System Architecture

### 6.1 Hardware Setup

**Development/Demo Environment:**

| Device | Role | Chip | RAM | Storage | Network | Models |
|--------|------|------|-----|---------|---------|--------|
| Mac Mini | Primary host (Sentinel + Observability + Ollama) | Apple M4 | 16GB | 256GB+ | 192.168.x.10 | llama3.2:8b-q4 |
| MacBook | Secondary Ollama endpoint | Apple M1 | 16GB | 256GB | 192.168.x.20 | gemma2:9b-q4 |

**Network Topology:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOCAL NETWORK                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Mac Mini (192.168.x.10)                  │   │
│  │                                                             │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────────┐   │   │
│  │  │  Sentinel   │  │   Ollama    │  │   Observability   │   │   │
│  │  │  :8000      │  │   :11434    │  │   Stack           │   │   │
│  │  │             │  │             │  │   :3000 (Grafana) │   │   │
│  │  │             │  │ llama3.2:8b │  │   :9091 (Prom)    │   │   │
│  │  └─────────────┘  └─────────────┘  └───────────────────┘   │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              │ LAN                                  │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   MacBook (192.168.x.20)                    │   │
│  │                                                             │   │
│  │                       ┌─────────────┐                       │   │
│  │                       │   Ollama    │                       │   │
│  │                       │   :11434    │                       │   │
│  │                       │             │                       │   │
│  │                       │ gemma2:9b   │                       │   │
│  │                       └─────────────┘                       │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Benefits of this setup:**
- Demonstrates distributed inference across heterogeneous Apple Silicon
- Model diversity (Llama vs Gemma) enables quality comparisons
- Failover testing without complex infrastructure
- Realistic network latency measurements

### 6.2 Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE SENTINEL                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        GATEWAY LAYER                            │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │   │
│  │  │   FastAPI   │  │   Uvicorn   │  │  OpenTelemetry SDK      │ │   │
│  │  │   (async)   │  │   (ASGI)    │  │  (auto-instrumentation) │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      CORE SERVICES                              │   │
│  │                                                                 │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │   │
│  │  │    Classifier   │  │  Routing Engine │  │   Controller   │  │   │
│  │  │                 │  │                 │  │                │  │   │
│  │  │  • Regex rules  │  │  • Tier logic   │  │  • Metrics     │  │   │
│  │  │  • NER model    │  │  • SLO checks   │  │  • Thresholds  │  │   │
│  │  │  • Taxonomy     │  │  • A/B shadow   │  │  • Dampening   │  │   │
│  │  └─────────────────┘  └─────────────────┘  └────────────────┘  │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                    ┌───────────────┴───────────────┐                    │
│                    ▼                               ▼                    │
│  ┌──────────────────────────────┐    ┌──────────────────────────────────┐  │
│  │     LOCAL BACKENDS          │    │        CLOUD BACKENDS            │  │
│  │     (Distributed)           │    │                                  │  │
│  │                             │    │  ┌─────────────┐ ┌────────────┐ │  │
│  │  ┌────────────────────┐     │    │  │  Anthropic  │ │   Google   │ │  │
│  │  │  Mac Mini (M4)     │     │    │  │  (Claude)   │ │  (Gemini)  │ │  │
│  │  │  192.168.x.10      │     │    │  └─────────────┘ └────────────┘ │  │
│  │  │  llama3.2:8b-q4    │     │    │                                  │  │
│  │  └────────────────────┘     │    └──────────────────────────────────┘  │
│  │                             │                                          │
│  │  ┌────────────────────┐     │                                          │
│  │  │  MacBook (M1)      │     │                                          │
│  │  │  192.168.x.20      │     │                                          │
│  │  │  gemma2:9b-q4      │     │                                          │
│  │  └────────────────────┘     │                                          │
│  │                             │                                          │
│  └──────────────────────────────┘                                          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                        OBSERVABILITY STACK                              │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Prometheus  │  │    Loki     │  │    Tempo    │  │   Grafana   │   │
│  │  (metrics)  │  │   (logs)    │  │  (traces)   │  │ (dashboards)│   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Docker Compose (Development)

**Note:** This runs on the Mac Mini. The MacBook runs Ollama standalone (not in Docker) and is accessed over the network. Configure `OLLAMA_HOST=0.0.0.0` on MacBook to allow network access.

```yaml
# docker-compose.yaml

version: '3.8'

services:
  # ============== CORE APPLICATION ==============
  sentinel:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"           # API
      - "9090:9090"           # Metrics
    environment:
      - SENTINEL_ENV=development
      # Local endpoints (Mac Mini local + MacBook over network)
      - LOCAL_ENDPOINTS=mac-mini:http://ollama:11434,macbook:http://192.168.1.20:11434
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
      - OTEL_SERVICE_NAME=inference-sentinel
    volumes:
      - ./config:/app/config:ro
    depends_on:
      - ollama
      - prometheus
      - tempo
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G

  # ============== LOCAL INFERENCE (Mac Mini) ==============
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    # For Apple Silicon Macs, Ollama uses Metal automatically
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Model initialization (runs once)
  ollama-init:
    image: ollama/ollama:latest
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        ollama pull llama3.2:8b-instruct-q4_K_M
    volumes:
      - ollama_data:/root/.ollama
      
  # ============== NOTE: MacBook Ollama ==============
  # The MacBook runs Ollama natively (not in Docker) for Metal acceleration.
  # Setup on MacBook:
  #   1. brew install ollama
  #   2. OLLAMA_HOST=0.0.0.0 ollama serve
  #   3. ollama pull gemma2:9b-instruct-q4_K_M
  # 
  # Verify connectivity from Mac Mini:
  #   curl http://192.168.1.20:11434/api/tags

  # ============== OBSERVABILITY ==============
  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9091:9090"
    volumes:
      - ./observability/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
      - '--storage.tsdb.retention.time=7d'

  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    volumes:
      - ./observability/loki/loki-config.yml:/etc/loki/local-config.yaml:ro
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  tempo:
    image: grafana/tempo:2.3.0
    ports:
      - "4317:4317"           # OTLP gRPC
      - "4318:4318"           # OTLP HTTP
      - "3200:3200"           # Tempo query
    volumes:
      - ./observability/tempo/tempo-config.yml:/etc/tempo/tempo.yaml:ro
      - tempo_data:/var/tempo
    command: ["-config.file=/etc/tempo/tempo.yaml"]

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=sentinel
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    volumes:
      - ./observability/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./observability/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
      - loki
      - tempo

volumes:
  ollama_data:
  prometheus_data:
  loki_data:
  tempo_data:
  grafana_data:
```

### 6.4 Kubernetes Architecture (Production)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  namespace: inference-sentinel                                          │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                                                                   │ │
│  │  ┌─────────────────┐      ┌─────────────────────────────────────┐│ │
│  │  │    Ingress      │      │         ConfigMaps/Secrets          ││ │
│  │  │  (nginx/istio)  │      │  • routing-config                   ││ │
│  │  │                 │      │  • privacy-taxonomy                 ││ │
│  │  │  /v1/inference  │      │  • controller-config                ││ │
│  │  │  /health        │      │  • cloud-credentials (sealed)       ││ │
│  │  │  /metrics       │      │                                     ││ │
│  │  └────────┬────────┘      └─────────────────────────────────────┘│ │
│  │           │                                                       │ │
│  │           ▼                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────┐ │ │
│  │  │                    Deployment: sentinel                     │ │ │
│  │  │                                                             │ │ │
│  │  │  replicas: 3 (HPA: 2-10 based on CPU/request rate)         │ │ │
│  │  │                                                             │ │ │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │ │ │
│  │  │  │   Pod 1     │ │   Pod 2     │ │   Pod 3     │           │ │ │
│  │  │  │             │ │             │ │             │           │ │ │
│  │  │  │ sentinel:   │ │ sentinel:   │ │ sentinel:   │           │ │ │
│  │  │  │   8000      │ │   8000      │ │   8000      │           │ │ │
│  │  │  │             │ │             │ │             │           │ │ │
│  │  │  │ otel-       │ │ otel-       │ │ otel-       │           │ │ │
│  │  │  │ collector   │ │ collector   │ │ collector   │           │ │ │
│  │  │  │ (sidecar)   │ │ (sidecar)   │ │ (sidecar)   │           │ │ │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘           │ │ │
│  │  │                                                             │ │ │
│  │  │  Resources per pod:                                         │ │ │
│  │  │    requests: 500m CPU, 1Gi memory                          │ │ │
│  │  │    limits: 2 CPU, 4Gi memory                               │ │ │
│  │  │                                                             │ │ │
│  │  └─────────────────────────────────────────────────────────────┘ │ │
│  │                              │                                    │ │
│  │           ┌──────────────────┴───────────────────┐               │ │
│  │           ▼                                      ▼               │ │
│  │  ┌─────────────────────┐            ┌─────────────────────────┐ │ │
│  │  │ Service: ollama     │            │  External Cloud APIs    │ │ │
│  │  │ (ClusterIP)         │            │                         │ │ │
│  │  │                     │            │  • api.anthropic.com    │ │ │
│  │  │ ┌─────────────────┐ │            │  • generativelanguage.  │ │ │
│  │  │ │ StatefulSet:    │ │            │    googleapis.com       │ │ │
│  │  │ │ ollama          │ │            │                         │ │ │
│  │  │ │                 │ │            │  (via NetworkPolicy     │ │ │
│  │  │ │ replicas: 1     │ │            │   egress allowlist)     │ │ │
│  │  │ │ (GPU node)      │ │            │                         │ │ │
│  │  │ │                 │ │            └─────────────────────────┘ │ │
│  │  │ │ resources:      │ │                                        │ │
│  │  │ │  nvidia.com/gpu │ │                                        │ │
│  │  │ │  OR             │ │                                        │ │
│  │  │ │  CPU-only       │ │                                        │ │
│  │  │ │                 │ │                                        │ │
│  │  │ │ PVC: 50Gi       │ │                                        │ │
│  │  │ │ (model storage) │ │                                        │ │
│  │  │ └─────────────────┘ │                                        │ │
│  │  └─────────────────────┘                                        │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  namespace: observability                                               │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐  │ │
│  │  │Prometheus │  │   Loki    │  │   Tempo   │  │    Grafana    │  │ │
│  │  │ (or Mimir │  │           │  │           │  │               │  │ │
│  │  │ for scale)│  │           │  │           │  │  Dashboards:  │  │ │
│  │  │           │  │           │  │           │  │  • Overview   │  │ │
│  │  │ ServiceMon│  │ Promtail  │  │ OTel Coll │  │  • Privacy    │  │ │
│  │  │ itor CRDs │  │ DaemonSet │  │ (central) │  │  • Cost       │  │ │
│  │  └───────────┘  └───────────┘  └───────────┘  │  • Controller │  │ │
│  │                                               └───────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.5 Key Kubernetes Manifests

**HorizontalPodAutoscaler:**

```yaml
# k8s/sentinel-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentinel-hpa
  namespace: inference-sentinel
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentinel
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: sentinel_requests_per_second
        target:
          type: AverageValue
          averageValue: "50"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120
```

**NetworkPolicy:**

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sentinel-network-policy
  namespace: inference-sentinel
spec:
  podSelector:
    matchLabels:
      app: sentinel
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    # Allow Ollama (internal)
    - to:
        - podSelector:
            matchLabels:
              app: ollama
      ports:
        - port: 11434
    # Allow cloud APIs
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443
    # Allow observability
    - to:
        - namespaceSelector:
            matchLabels:
              name: observability
      ports:
        - port: 4317  # OTLP
        - port: 9090  # Prometheus
```

---

## 8. Benchmark Methodology

### 7.1 Objectives

The benchmark answers these questions:

1. **Privacy overhead**: How much latency does classification add?
2. **Routing accuracy**: Does the classifier correctly identify sensitive content?
3. **Local vs. cloud tradeoff**: At what prompt complexity does cloud win on latency?
4. **Cost savings**: How much money does intelligent routing save?
5. **Closed-loop effectiveness**: Do adaptive thresholds improve SLO compliance?

### 7.2 Synthetic Dataset

**Dataset composition:**

| Category | Count | Description |
|----------|-------|-------------|
| Tier 0 (PUBLIC) | 500 | General knowledge, coding, creative writing |
| Tier 1 (INTERNAL) | 300 | Business context, internal project discussions |
| Tier 2 (CONFIDENTIAL) | 300 | Contains PII (names, emails, phones, addresses) |
| Tier 3 (RESTRICTED) | 200 | SSNs, credit cards, health records, credentials |
| Adversarial | 100 | Edge cases (partial SSNs, fake PII, obfuscated) |
| **Total** | 1,400 | |

**Prompt characteristics:**

```yaml
prompt_distribution:
  length:
    short: 30%      # < 100 tokens
    medium: 50%     # 100-500 tokens
    long: 20%       # 500-2000 tokens
    
  complexity:
    simple: 40%     # Single-turn, factual
    moderate: 40%   # Multi-step reasoning
    complex: 20%    # Long-context, analysis
    
  expected_output:
    short: 30%      # < 100 tokens
    medium: 50%     # 100-500 tokens
    long: 20%       # 500-2000 tokens
```

**Generation methodology:**

- Tier 0-1: Sample from open datasets (Alpaca, ShareGPT) + synthetic generation
- Tier 2-3: Synthetic generation with controlled PII injection using Faker
- Adversarial: Hand-crafted edge cases + automated perturbations

### 7.3 Experiment Protocol

**Experiment 1: Classification Accuracy**

```yaml
experiment_1:
  name: "Classification Accuracy"
  objective: "Measure precision/recall of privacy classification"
  
  setup:
    - Use labeled synthetic dataset (1,400 prompts)
    - Ground truth: human-labeled tier assignments
    
  metrics:
    - precision_per_tier
    - recall_per_tier
    - f1_per_tier
    - confusion_matrix
    - classification_latency_p50_p95_p99
    
  variations:
    - regex_only: true
    - ner_only: true
    - hybrid: true
    
  success_criteria:
    - tier_3_recall > 0.99       # Must catch restricted content
    - tier_2_recall > 0.95
    - overall_f1 > 0.90
    - classification_p99 < 100ms
```

**Experiment 2: Routing Performance**

```yaml
experiment_2:
  name: "Routing Performance Comparison"
  objective: "Compare latency/throughput across routing strategies"
  
  setup:
    - 1,000 requests per configuration
    - Warm-up: 100 requests (discarded)
    - Cooldown: 60s between configurations
    
  configurations:
    - all_local: Force all requests to Ollama
    - all_cloud_anthropic: Force all to Claude
    - all_cloud_google: Force all to Gemini
    - privacy_routed: Use inference-sentinel routing
    
  metrics:
    - ttft_p50_p95_p99
    - itl_p50_p95_p99                    # Inter-token latency
    - tpot_p50_p95_p99                   # Time per output token
    - total_latency_p50_p95_p99
    - tokens_per_second
    - error_rate
    - cost_total
    - cost_per_request
    
  controls:
    - Same prompts across all configurations
    - Sequential execution (no parallel interference)
    - System load monitoring (reject if CPU > 80% baseline)
```

**Experiment 3: Cost Attribution**

```yaml
experiment_3:
  name: "Cost Savings Analysis"
  objective: "Quantify cost savings from privacy-aware routing"
  
  setup:
    - Run privacy_routed configuration
    - Calculate actual cost vs. hypothetical all-cloud cost
    
  scenarios:
    - scenario_a:
        name: "Privacy-heavy workload"
        distribution: { tier_0: 10%, tier_1: 20%, tier_2: 40%, tier_3: 30% }
    - scenario_b:
        name: "Balanced workload"
        distribution: { tier_0: 40%, tier_1: 30%, tier_2: 20%, tier_3: 10% }
    - scenario_c:
        name: "Public-heavy workload"
        distribution: { tier_0: 70%, tier_1: 20%, tier_2: 7%, tier_3: 3% }
        
  metrics:
    - total_cost_actual
    - total_cost_cloud_baseline
    - savings_absolute
    - savings_percentage
    - cost_per_tier_breakdown
```

**Experiment 4: Closed-Loop Adaptation**

```yaml
experiment_4:
  name: "Closed-Loop Controller Effectiveness"
  objective: "Validate controller improves SLO compliance under load"
  
  setup:
    - 2-hour sustained load test
    - Inject performance degradation at T+30m (slow Ollama responses)
    - Remove degradation at T+90m
    
  configurations:
    - static_thresholds: Controller disabled
    - adaptive_thresholds: Controller enabled
    
  metrics:
    - slo_compliance_rate (% requests within latency SLO)
    - threshold_adjustment_count
    - time_to_adapt (how fast controller responds)
    - overshoot_rate (unnecessary cloud routing after recovery)
    
  success_criteria:
    - adaptive_slo_compliance > static_slo_compliance + 10%
    - time_to_adapt < 15 minutes
    - overshoot_rate < 5%
```

**Experiment 5: Multi-Node Local Inference**

```yaml
experiment_5:
  name: "Multi-Node Local Inference Performance"
  objective: "Evaluate distributed local inference across heterogeneous hardware"
  
  setup:
    - Mac Mini (M4, 16GB) running Ollama with llama3.2:8b-q4
    - MacBook (M1, 16GB) running Ollama with gemma2:9b-q4
    - Both on same local network
    - 500 requests per configuration
    
  configurations:
    - single_node_mini:
        endpoints: [mac-mini]
        description: "Baseline: Mac Mini only"
    - single_node_macbook:
        endpoints: [macbook]
        description: "Baseline: MacBook only"
    - multi_node_round_robin:
        endpoints: [mac-mini, macbook]
        strategy: round_robin
        description: "Load balanced across both"
    - multi_node_latency_best:
        endpoints: [mac-mini, macbook]
        strategy: latency_best
        description: "Route to fastest available"
    - multi_node_failover:
        endpoints: [mac-mini, macbook]
        strategy: priority
        inject_failure: mac-mini  # Simulate primary failure
        description: "Failover behavior"
        
  metrics:
    - throughput_requests_per_second
    - ttft_p50_p95_p99_per_endpoint
    - itl_p50_p95_p99_per_endpoint       # Inter-token latency per endpoint
    - tpot_p50_p95_p99_per_endpoint      # Time per output token per endpoint
    - total_latency_p50_p95_p99_per_endpoint
    - network_overhead_ms              # Time added by network hop
    - load_distribution_percentage     # Requests per endpoint
    - failover_latency_ms              # Time to detect failure and reroute
    - model_comparison:
        - llama_vs_gemma_ttft
        - llama_vs_gemma_itl             # Decode performance comparison
        - llama_vs_gemma_tpot
        - llama_vs_gemma_quality       # If shadow mode enabled
        
  success_criteria:
    - multi_node_throughput > single_node_throughput * 1.5
    - failover_latency < 5000ms
    - network_overhead_p95 < 50ms      # LAN should be fast
```

### 7.4 Reproducibility Requirements

Every benchmark run produces:

```
benchmark_results/
├── run_metadata.json          # Git SHA, timestamp, system specs
├── system_snapshot.json       # Docker versions, model checksums
├── config_snapshot/           # All config files used
│   ├── routing.yaml
│   ├── privacy_taxonomy.yaml
│   └── controller.yaml
├── raw_data/                  # Per-request telemetry
│   ├── experiment_1_results.jsonl
│   ├── experiment_2_results.jsonl
│   ├── experiment_3_results.jsonl
│   └── experiment_4_results.jsonl
├── analysis/                  # Computed metrics
│   ├── summary_statistics.json
│   ├── latency_distributions.png
│   ├── cost_breakdown.png
│   └── controller_adaptation.png
└── BENCHMARK_REPORT.md        # Human-readable summary
```

**System requirements documented:**

```json
{
  "hardware": {
    "cpu": "Apple M4",
    "memory_gb": 16,
    "storage_type": "SSD"
  },
  "software": {
    "os": "macOS 15.x",
    "docker_version": "25.x",
    "ollama_version": "0.x.x",
    "python_version": "3.12.x"
  },
  "models": {
    "local": {
      "name": "llama3.2:8b-instruct-q4_K_M",
      "sha256": "abc123..."
    }
  }
}
```

---

## 9. Phased Roadmap

### Phase 0: Foundation (Week 1) ✅

**Goal**: Repository setup, basic FastAPI skeleton, Docker Compose with Ollama

**Deliverables:**
- [x] Repository initialized with structure (Section 9)
- [x] FastAPI app with `/health`, `/v1/inference` endpoints
- [x] Docker Compose with sentinel + ollama services
- [x] Basic request/response schema (no routing logic yet)
- [x] Ollama responding to hardcoded requests
- [x] README with setup instructions

**Exit criteria**: Can send a request to localhost:8000/v1/inference and get a response from Ollama

---

### Phase 1: Privacy Classification (Week 2) ✅

**Goal**: Implement privacy taxonomy and classification pipeline

**Deliverables:**
- [x] Privacy taxonomy YAML schema and loader
- [x] Regex-based classifier (Tier 3 patterns)
- [x] Classification result schema
- [x] Unit tests for all Tier 3 patterns (100% coverage)
- [x] Classification endpoint for testing: `POST /v1/classify`
- [x] Benchmark: classification latency (target: p99 < 50ms)

**Exit criteria**: Classifier correctly identifies all Tier 3 entities with >99% recall

---

### Phase 2: Routing Engine (Week 3) ✅

**Goal**: Implement routing logic and cloud backend integration

**Deliverables:**
- [x] Routing configuration schema and loader
- [x] Routing decision engine (tier-based + SLO checks)
- [x] Anthropic backend adapter
- [x] Google backend adapter  
- [x] Backend health checking
- [x] Routing decision logging
- [x] Integration tests: tier → correct backend

**Exit criteria**: Requests correctly route to local or cloud based on classification

---

### Phase 3: Observability (Week 4) ✅

**Goal**: Full telemetry pipeline with Grafana dashboards

**Deliverables:**
- [x] OpenTelemetry instrumentation (traces)
- [x] Prometheus metrics (counters, histograms, gauges)
- [x] Structured JSON logging (Loki-compatible)
- [x] Tempo integration for trace storage
- [x] Grafana dashboards:
  - [x] Overview (requests, latency, routes)
  - [x] Privacy (tier distribution, detections)
  - [x] Cost (savings, per-backend breakdown)
- [x] Docker Compose with full observability stack

**Exit criteria**: Can visualize end-to-end request flow in Grafana with correlated metrics/traces/logs

---

### Phase 4: Advanced Features (Week 5-6) ✅

**Goal**: NER classifier, A/B shadow mode, closed-loop controller, session stickiness

**Deliverables:**
- [x] NER model integration (Tier 1-2 classification)
- [x] Hybrid classification pipeline (regex → NER)
- [x] A/B shadow mode implementation
- [x] Output similarity computation
- [x] Closed-loop controller
- [x] Controller metrics + dashboard
- [x] Configuration hot-reload
- [x] Session stickiness with context handoff
- [x] Round-robin cloud backend selection

**Exit criteria**: Controller automatically adjusts thresholds in response to simulated degradation

---

### Phase 5: Benchmark Suite (Week 7)

**Goal**: Reproducible benchmark methodology and initial results

**Deliverables:**
- [x] Synthetic dataset generator
- [x] Benchmark harness (experiment runner)
- [x] Experiment 1: Classification accuracy
- [x] Experiment 2: Routing performance
- [x] Experiment 3: Cost attribution
- [x] Experiment 4: Closed-loop effectiveness
- [x] Experiment 5: Session stickiness
- [ ] Automated report generation
- [ ] Reproducibility documentation

**Exit criteria**: Full benchmark suite runs and produces documented results

---

### Phase 6: Production Hardening (Week 8)

**Goal**: Kubernetes manifests, security hardening, documentation

**Deliverables:**
- [ ] Kubernetes manifests (Deployment, Service, HPA, NetworkPolicy)
- [ ] Helm chart (optional)
- [ ] Security: API authentication, rate limiting
- [ ] Security: Secrets management (sealed secrets or external-secrets)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Comprehensive documentation
- [ ] Demo video / blog post outline

**Exit criteria**: Production-ready deployment with CI/CD and security controls

---

### v1.0 Release Checklist

- [ ] All Phase 0-6 deliverables complete
- [ ] Benchmark results documented
- [ ] README polished with architecture diagrams
- [ ] Demo video recorded
- [ ] Blog post published (optional but recommended)
- [ ] Tagged release on GitHub

---

## 10. Repository Structure

```
inference-sentinel/
├── README.md
├── LICENSE                          # Apache 2.0 or MIT
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml                   # Python packaging (uv/poetry)
├── Dockerfile
├── docker-compose.yaml
├── .github/
│   └── workflows/
│       ├── ci.yaml                  # Lint, test, type-check
│       ├── benchmark.yaml           # Weekly benchmark runs
│       └── release.yaml             # Tagged releases
│
├── src/
│   └── sentinel/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app entrypoint
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py          # Pydantic settings
│       │   └── loader.py            # YAML config loading
│       │
│       ├── classification/
│       │   ├── __init__.py
│       │   ├── taxonomy.py          # Privacy taxonomy model
│       │   ├── regex_classifier.py  # Fast-pass regex patterns
│       │   ├── ner_classifier.py    # NER model wrapper
│       │   ├── hybrid.py            # Combined pipeline
│       │   └── schemas.py           # ClassificationResult, Entity
│       │
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── engine.py            # Routing decision logic
│       │   ├── schemas.py           # RoutingDecision
│       │   └── shadow.py            # A/B shadow mode
│       │
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract backend interface
│       │   ├── ollama.py            # Ollama adapter
│       │   ├── anthropic.py         # Claude adapter
│       │   ├── google.py            # Gemini adapter
│       │   └── health.py            # Health checking
│       │
│       ├── controller/
│       │   ├── __init__.py
│       │   ├── metrics.py           # ControllerMetrics aggregation
│       │   ├── adjustments.py       # Adjustment algorithm
│       │   └── loop.py              # Background control loop
│       │
│       ├── telemetry/
│       │   ├── __init__.py
│       │   ├── tracing.py           # OpenTelemetry setup
│       │   ├── metrics.py           # Prometheus metrics
│       │   └── logging.py           # Structured logging
│       │
│       └── api/
│           ├── __init__.py
│           ├── routes.py            # FastAPI routes
│           ├── middleware.py        # Request ID, timing
│           └── schemas.py           # API request/response models
│
├── config/
│   ├── routing.yaml
│   ├── privacy_taxonomy.yaml
│   ├── controller.yaml
│   └── logging.yaml
│
├── observability/
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── loki/
│   │   └── loki-config.yml
│   ├── tempo/
│   │   └── tempo-config.yml
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── datasources.yaml
│       │   └── dashboards/
│       │       └── dashboards.yaml
│       └── dashboards/
│           ├── overview.json
│           ├── privacy.json
│           ├── cost.json
│           └── controller.json
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmaps.yaml
│   ├── secrets.yaml                 # Template only, real secrets sealed
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── network-policy.yaml
│   ├── ollama-statefulset.yaml
│   └── kustomization.yaml
│
├── benchmarks/
│   ├── README.md
│   ├── datasets/
│   │   ├── synthetic_generator.py
│   │   └── .gitkeep               # Generated data not committed
│   ├── experiments/
│   │   ├── experiment_1_classification.py
│   │   ├── experiment_2_routing.py
│   │   ├── experiment_3_cost.py
│   │   └── experiment_4_controller.py
│   ├── harness.py                  # Experiment runner
│   ├── analysis.py                 # Results processing
│   └── results/                    # Committed benchmark results
│       └── .gitkeep
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_regex_classifier.py
│   │   ├── test_taxonomy.py
│   │   ├── test_routing_engine.py
│   │   └── test_controller.py
│   ├── integration/
│   │   ├── test_classification_pipeline.py
│   │   ├── test_backend_adapters.py
│   │   └── test_end_to_end.py
│   └── fixtures/
│       ├── prompts/
│       │   ├── tier_0_samples.json
│       │   ├── tier_1_samples.json
│       │   ├── tier_2_samples.json
│       │   └── tier_3_samples.json
│       └── expected_responses/
│
└── docs/
    ├── architecture.md
    ├── configuration.md
    ├── deployment.md
    ├── benchmarks.md
    └── diagrams/
        ├── system-overview.svg
        └── routing-flow.svg
```

---

## Appendix A: API Reference (Draft)

### POST /v1/inference

**Request:**

```json
{
  "model": "auto",                    // "auto", "local", "cloud", or specific model
  "messages": [
    {"role": "user", "content": "..."}
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false,
  "routing_override": null,           // Force "local" or "cloud"
  "classification_only": false        // If true, return classification without inference
}
```

**Response:**

```json
{
  "id": "req_abc123",
  "model": "llama3.2:8b-instruct-q4_K_M",
  "choices": [
    {
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 280,
    "total_tokens": 430
  },
  "sentinel": {
    "route": "local",
    "backend": "ollama",
    "endpoint": "mac-mini",
    "privacy_tier": 2,
    "privacy_tier_label": "CONFIDENTIAL",
    "entities_detected": ["email", "phone"],
    "classification_latency_ms": 12.5,
    "routing_latency_ms": 1.2,
    "inference_latency_ms": 3420.0,
    "ttft_ms": 245.0,
    "itl_p50_ms": 12.3,
    "itl_p95_ms": 18.7,
    "tpot_ms": 14.2,
    "tokens_per_second": 81.9,
    "cost_usd": 0.0,
    "cost_savings_usd": 0.012
  }
}
```

---

## Appendix B: Grafana Dashboard Panels

### Overview Dashboard

1. **Request Rate** - Requests/second, split by route (local/cloud)
2. **TTFT Distribution** - Heatmap of time to first token
3. **ITL Distribution** - Heatmap of inter-token latency (decode performance)
4. **TTFT vs ITL by Backend** - Side-by-side comparison (prefill vs decode)
5. **TPOT by Endpoint** - Time per output token (M4 vs M1 comparison)
6. **Route Distribution** - Pie chart of local vs. cloud %
7. **Error Rate** - Error % over time
8. **Active Backends** - Health status indicators

### Privacy Dashboard

1. **Tier Distribution** - Bar chart of requests per tier
2. **Entity Types Detected** - Top 10 entity types
3. **Classification Latency** - p50/p95/p99 over time
4. **Tier 3 Detections** - Restricted content alerts
5. **SLO Overrides** - When privacy was overridden for SLO

### Cost Dashboard

1. **Cumulative Savings** - Running total of $ saved
2. **Cost by Backend** - Actual spend per backend
3. **Cost Avoidance** - What cloud would have cost
4. **Cost per Request** - Average cost trending
5. **Savings Rate** - % saved vs. all-cloud baseline

### Controller Dashboard

1. **Current Thresholds** - Gauge of active thresholds
2. **Threshold History** - Changes over time
3. **SLO Compliance** - % of requests within SLO
4. **Adjustment Events** - Log of controller actions
5. **Backend Health** - Availability over time

---

*End of Design Specification*