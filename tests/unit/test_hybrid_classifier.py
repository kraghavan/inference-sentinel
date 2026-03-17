"""Unit tests for hybrid classifier (regex + NER)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sentinel.classification.hybrid_classifier import (
    HybridClassifier,
    HybridResult,
    get_hybrid_classifier,
    configure_hybrid_classifier,
)
from sentinel.classification.ner_classifier import NERClassifier, NERResult, NEREntity
from sentinel.classification.regex_classifier import RegexClassifier
from sentinel.classification.schemas import ClassificationResult, DetectedEntity


class TestHybridResult:
    """Test HybridResult dataclass."""
    
    def test_default_result(self):
        """Default result should be public tier."""
        result = HybridResult()
        assert result.tier == 0
        assert result.tier_label == "PUBLIC"
        assert result.is_sensitive is False
        assert result.requires_local is False
    
    def test_tier_2_requires_local(self):
        """Tier 2 should require local routing."""
        result = HybridResult(tier=2, tier_label="CONFIDENTIAL")
        assert result.is_sensitive is True
        assert result.requires_local is True
    
    def test_tier_3_requires_local(self):
        """Tier 3 should require local routing."""
        result = HybridResult(tier=3, tier_label="RESTRICTED")
        assert result.is_sensitive is True
        assert result.requires_local is True
    
    def test_to_classification_result(self):
        """Should convert to ClassificationResult."""
        hybrid = HybridResult(
            tier=2,
            tier_label="CONFIDENTIAL",
            entities_detected=[
                DetectedEntity(
                    entity_type="email",
                    tier=2,
                    start_pos=0,
                    end_pos=10,
                    confidence=1.0,
                )
            ],
            entity_types=["email"],
            entity_count=1,
            total_latency_ms=5.5,
            detection_method="hybrid",
        )
        
        result = hybrid.to_classification_result()
        
        assert isinstance(result, ClassificationResult)
        assert result.tier == 2
        assert result.tier_label == "CONFIDENTIAL"
        assert result.entity_count == 1


class TestHybridClassifier:
    """Test HybridClassifier class."""
    
    @pytest.fixture
    def mock_regex_classifier(self):
        """Create a mock regex classifier."""
        mock = MagicMock(spec=RegexClassifier)
        mock.classify.return_value = ClassificationResult(
            tier=0,
            tier_label="PUBLIC",
            entities_detected=[],
            entity_types=[],
            entity_count=0,
        )
        return mock
    
    @pytest.fixture
    def mock_ner_classifier(self):
        """Create a mock NER classifier."""
        mock = MagicMock(spec=NERClassifier)
        mock.enabled = True
        mock.classify = AsyncMock(return_value=NERResult(
            entities=[],
            highest_tier=0,
            latency_ms=50.0,
            model_name="test-model",
        ))
        return mock
    
    def test_classifier_init_with_ner_disabled(self):
        """Classifier should work with NER disabled."""
        classifier = HybridClassifier(ner_enabled=False)
        assert classifier.ner_enabled is False
    
    @pytest.mark.asyncio
    async def test_classify_regex_only_when_ner_disabled(
        self, mock_regex_classifier, mock_ner_classifier
    ):
        """Should use only regex when NER is disabled."""
        classifier = HybridClassifier(
            regex_classifier=mock_regex_classifier,
            ner_classifier=mock_ner_classifier,
            ner_enabled=False,
        )
        
        result = await classifier.classify("Hello world")
        
        assert result.ner_skipped is True
        assert result.ner_skipped_reason == "NER disabled in config"
        assert result.detection_method == "regex"
        mock_ner_classifier.classify.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_skip_ner_when_regex_finds_tier_3(
        self, mock_regex_classifier, mock_ner_classifier
    ):
        """Should skip NER when regex finds tier 3."""
        mock_regex_classifier.classify.return_value = ClassificationResult(
            tier=3,
            tier_label="RESTRICTED",
            entities_detected=[
                DetectedEntity(
                    entity_type="ssn",
                    tier=3,
                    start_pos=0,
                    end_pos=11,
                    confidence=1.0,
                )
            ],
            entity_types=["ssn"],
            entity_count=1,
        )
        
        classifier = HybridClassifier(
            regex_classifier=mock_regex_classifier,
            ner_classifier=mock_ner_classifier,
            ner_enabled=True,
            skip_ner_on_tier3=True,
        )
        
        result = await classifier.classify("My SSN is 123-45-6789")
        
        assert result.tier == 3
        assert result.ner_skipped is True
        assert "Tier 3 detected" in result.ner_skipped_reason
        mock_ner_classifier.classify.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_run_ner_when_regex_finds_tier_0(
        self, mock_regex_classifier, mock_ner_classifier
    ):
        """Should run NER when regex finds tier 0."""
        mock_ner_classifier.classify = AsyncMock(return_value=NERResult(
            entities=[
                NEREntity(
                    text="John Doe",
                    entity_type="PER",
                    tier=2,
                    start_pos=10,
                    end_pos=18,
                    confidence=0.95,
                )
            ],
            highest_tier=2,
            latency_ms=50.0,
            model_name="test-model",
        ))
        
        classifier = HybridClassifier(
            regex_classifier=mock_regex_classifier,
            ner_classifier=mock_ner_classifier,
            ner_enabled=True,
        )
        
        result = await classifier.classify("I am John Doe")
        
        assert result.tier == 2  # NER upgraded tier
        assert result.ner_skipped is False
        assert result.detection_method == "hybrid"
        assert result.ner_tier == 2
        mock_ner_classifier.classify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_merge_results_combines_entities(
        self, mock_regex_classifier, mock_ner_classifier
    ):
        """Should merge entities from both classifiers."""
        mock_regex_classifier.classify.return_value = ClassificationResult(
            tier=2,
            tier_label="CONFIDENTIAL",
            entities_detected=[
                DetectedEntity(
                    entity_type="email",
                    tier=2,
                    start_pos=0,
                    end_pos=15,
                    confidence=1.0,
                )
            ],
            entity_types=["email"],
            entity_count=1,
        )
        
        mock_ner_classifier.classify = AsyncMock(return_value=NERResult(
            entities=[
                NEREntity(
                    text="Anthropic",
                    entity_type="ORG",
                    tier=1,
                    start_pos=20,
                    end_pos=29,
                    confidence=0.9,
                )
            ],
            highest_tier=1,
            latency_ms=50.0,
            model_name="test-model",
        ))
        
        classifier = HybridClassifier(
            regex_classifier=mock_regex_classifier,
            ner_classifier=mock_ner_classifier,
            ner_enabled=True,
            skip_ner_on_tier3=True,  # tier 2 won't skip
            ner_threshold_tier=3,
        )
        
        result = await classifier.classify("test@email.com at Anthropic")
        
        assert result.tier == 2  # Max of regex (2) and NER (1)
        assert result.entity_count == 2
        assert "email" in result.entity_types
        assert "ner_org" in result.entity_types
    
    @pytest.mark.asyncio
    async def test_classify_messages(self, mock_regex_classifier, mock_ner_classifier):
        """Should classify combined message content."""
        classifier = HybridClassifier(
            regex_classifier=mock_regex_classifier,
            ner_classifier=mock_ner_classifier,
            ner_enabled=False,
        )
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "My email is test@example.com"},
        ]
        
        result = await classifier.classify_messages(messages)
        
        # Should have combined all content
        mock_regex_classifier.classify.assert_called_once()
        call_text = mock_regex_classifier.classify.call_args[0][0]
        assert "Hello" in call_text
        assert "test@example.com" in call_text


class TestHybridClassifierGlobal:
    """Test global hybrid classifier functions."""
    
    def test_get_hybrid_classifier_returns_instance(self):
        """get_hybrid_classifier should return a classifier instance."""
        classifier = get_hybrid_classifier()
        assert isinstance(classifier, HybridClassifier)
    
    def test_configure_hybrid_classifier(self):
        """configure_hybrid_classifier should create configured classifier."""
        classifier = configure_hybrid_classifier(
            ner_enabled=False,
            ner_model="fast",
            ner_device="cpu",
            ner_confidence_threshold=0.8,
            ner_threshold_tier=3,
            skip_ner_on_tier3=True,
        )
        
        assert isinstance(classifier, HybridClassifier)
        assert classifier.ner_enabled is False
        assert classifier.ner_threshold_tier == 3
        assert classifier.skip_ner_on_tier3 is True
