"""NER-based classifier for detecting named entities.

Uses transformer models to detect entities that regex cannot reliably catch:
- Person names (PERSON)
- Organization names (ORG)
- Locations/Addresses (LOC, GPE)
- Misc entities (MISC)

This classifier is designed to be optional and async to minimize latency impact.
"""

import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from sentinel.telemetry import get_logger

logger = get_logger("sentinel.classification.ner")

# Entity type to privacy tier mapping
NER_ENTITY_TIERS: dict[str, int] = {
    # Tier 2 - Confidential (PII that should route local by default)
    "PERSON": 2,      # Person names
    "PER": 2,         # Alternative tag for person
    "B-PER": 2,
    "I-PER": 2,
    
    # Tier 1 - Internal (business context, cloud allowed)
    "ORG": 1,         # Organizations
    "B-ORG": 1,
    "I-ORG": 1,
    "GPE": 1,         # Geo-political entities (countries, cities)
    "LOC": 1,         # Locations
    "B-LOC": 1,
    "I-LOC": 1,
    
    # Tier 0 - Public (no concern)
    "MISC": 0,        # Miscellaneous
    "B-MISC": 0,
    "I-MISC": 0,
    "DATE": 0,
    "TIME": 0,
    "MONEY": 0,
    "PERCENT": 0,
    "QUANTITY": 0,
}


@dataclass
class NEREntity:
    """A detected named entity."""
    
    text: str
    entity_type: str
    tier: int
    start_pos: int
    end_pos: int
    confidence: float
    source: Literal["ner"] = "ner"


@dataclass
class NERResult:
    """Result of NER classification."""
    
    entities: list[NEREntity] = field(default_factory=list)
    highest_tier: int = 0
    latency_ms: float = 0.0
    model_name: str = ""
    error: str | None = None
    
    @property
    def has_pii(self) -> bool:
        """Check if any PII (tier >= 2) was detected."""
        return self.highest_tier >= 2
    
    @property
    def entity_types(self) -> list[str]:
        """Get unique entity types detected."""
        return list(set(e.entity_type for e in self.entities))


class NERClassifier:
    """Named Entity Recognition classifier using transformers.
    
    Lazy-loads the model on first use to avoid startup delay.
    Supports async execution to not block the event loop.
    """
    
    # Supported models (smallest to largest)
    MODELS = {
        "fast": "dslim/bert-base-NER",           # ~400MB, good accuracy
        "accurate": "Jean-Baptiste/roberta-large-ner-english",  # ~1.3GB, best accuracy
        "multilingual": "Davlan/bert-base-multilingual-cased-ner-hrl",  # ~700MB, multi-lang
    }
    
    def __init__(
        self,
        model_name: str = "fast",
        device: str = "cpu",  # "cpu", "cuda", "mps"
        confidence_threshold: float = 0.7,
        enabled: bool = True,
    ):
        self.model_key = model_name
        self.model_name = self.MODELS.get(model_name, model_name)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        
        self._pipeline = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the NER pipeline (lazy load)."""
        if not self.enabled:
            logger.info("NER classifier disabled")
            return
        
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Import here to make transformers optional
                from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
                
                logger.info(
                    "Loading NER model",
                    model=self.model_name,
                    device=self.device
                )
                
                # Load in a thread pool to not block
                loop = asyncio.get_event_loop()
                self._pipeline = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        "ner",
                        model=self.model_name,
                        device=self.device if self.device != "cpu" else -1,
                        aggregation_strategy="simple",  # Merge B-/I- tags
                    )
                )
                
                self._initialized = True
                logger.info("NER model loaded successfully")
                
            except ImportError:
                logger.warning(
                    "transformers not installed. Install with: pip install transformers torch"
                )
                self.enabled = False
            except Exception as e:
                logger.error("Failed to load NER model", error=str(e))
                self.enabled = False
    
    async def classify(self, text: str) -> NERResult:
        """Classify text for named entities.
        
        Args:
            text: Text to analyze
            
        Returns:
            NERResult with detected entities
        """
        import time
        start = time.perf_counter()
        
        if not self.enabled:
            return NERResult(error="NER disabled")
        
        if not self._initialized:
            await self.initialize()
        
        if not self._pipeline:
            return NERResult(error="NER pipeline not available")
        
        try:
            # Run inference in thread pool
            loop = asyncio.get_event_loop()
            raw_entities = await loop.run_in_executor(
                None,
                lambda: self._pipeline(text)
            )
            
            # Convert to our format
            entities = []
            highest_tier = 0
            
            for ent in raw_entities:
                confidence = ent.get("score", 0.0)
                
                # Skip low-confidence entities
                if confidence < self.confidence_threshold:
                    continue
                
                entity_type = ent.get("entity_group", ent.get("entity", "UNKNOWN"))
                tier = NER_ENTITY_TIERS.get(entity_type, 0)
                highest_tier = max(highest_tier, tier)
                
                entities.append(NEREntity(
                    text=ent.get("word", ""),
                    entity_type=entity_type,
                    tier=tier,
                    start_pos=ent.get("start", 0),
                    end_pos=ent.get("end", 0),
                    confidence=confidence,
                ))
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            return NERResult(
                entities=entities,
                highest_tier=highest_tier,
                latency_ms=latency_ms,
                model_name=self.model_name,
            )
            
        except Exception as e:
            logger.error("NER classification failed", error=str(e))
            return NERResult(
                error=str(e),
                latency_ms=(time.perf_counter() - start) * 1000,
            )
    
    def classify_sync(self, text: str) -> NERResult:
        """Synchronous classification (for testing)."""
        return asyncio.run(self.classify(text))


# Global singleton
_ner_classifier: NERClassifier | None = None


def get_ner_classifier() -> NERClassifier:
    """Get the global NER classifier instance."""
    global _ner_classifier
    if _ner_classifier is None:
        _ner_classifier = NERClassifier(enabled=False)  # Disabled by default
    return _ner_classifier


def configure_ner(
    model_name: str = "fast",
    device: str = "cpu",
    confidence_threshold: float = 0.7,
    enabled: bool = True,
) -> NERClassifier:
    """Configure and return the global NER classifier."""
    global _ner_classifier
    _ner_classifier = NERClassifier(
        model_name=model_name,
        device=device,
        confidence_threshold=confidence_threshold,
        enabled=enabled,
    )
    return _ner_classifier
