"""Unit tests for NER classifier."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sentinel.classification.ner_classifier import (
    NERClassifier,
    NEREntity,
    NERResult,
    NER_ENTITY_TIERS,
    get_ner_classifier,
    configure_ner,
)


class TestNEREntityTiers:
    """Test NER entity tier mappings."""
    
    def test_person_entities_are_tier_2(self):
        """Person names should be tier 2 (confidential)."""
        assert NER_ENTITY_TIERS["PERSON"] == 2
        assert NER_ENTITY_TIERS["PER"] == 2
        assert NER_ENTITY_TIERS["B-PER"] == 2
        assert NER_ENTITY_TIERS["I-PER"] == 2
    
    def test_org_entities_are_tier_1(self):
        """Organizations should be tier 1 (internal)."""
        assert NER_ENTITY_TIERS["ORG"] == 1
        assert NER_ENTITY_TIERS["B-ORG"] == 1
    
    def test_location_entities_are_tier_1(self):
        """Locations should be tier 1 (internal)."""
        assert NER_ENTITY_TIERS["LOC"] == 1
        assert NER_ENTITY_TIERS["GPE"] == 1
    
    def test_misc_entities_are_tier_0(self):
        """Miscellaneous entities should be tier 0 (public)."""
        assert NER_ENTITY_TIERS["MISC"] == 0
        assert NER_ENTITY_TIERS["DATE"] == 0


class TestNERResult:
    """Test NERResult dataclass."""
    
    def test_empty_result_has_no_pii(self):
        """Empty result should not have PII."""
        result = NERResult()
        assert result.has_pii is False
        assert result.highest_tier == 0
        assert result.entity_types == []
    
    def test_result_with_person_has_pii(self):
        """Result with person entity should have PII."""
        result = NERResult(
            entities=[
                NEREntity(
                    text="John Doe",
                    entity_type="PERSON",
                    tier=2,
                    start_pos=0,
                    end_pos=8,
                    confidence=0.95,
                )
            ],
            highest_tier=2,
        )
        assert result.has_pii is True
        assert result.entity_types == ["PERSON"]
    
    def test_result_with_org_has_no_pii(self):
        """Result with only org entity should not flag PII (tier < 2)."""
        result = NERResult(
            entities=[
                NEREntity(
                    text="Anthropic",
                    entity_type="ORG",
                    tier=1,
                    start_pos=0,
                    end_pos=9,
                    confidence=0.9,
                )
            ],
            highest_tier=1,
        )
        assert result.has_pii is False


class TestNERClassifier:
    """Test NERClassifier class."""
    
    def test_classifier_disabled_by_default(self):
        """Classifier should be disabled when enabled=False."""
        classifier = NERClassifier(enabled=False)
        assert classifier.enabled is False
    
    def test_classifier_model_selection(self):
        """Classifier should map model keys to model names."""
        classifier = NERClassifier(model_name="fast")
        assert classifier.model_name == "dslim/bert-base-NER"
        
        classifier = NERClassifier(model_name="accurate")
        assert classifier.model_name == "Jean-Baptiste/roberta-large-ner-english"
    
    def test_custom_model_name_passthrough(self):
        """Custom model names should pass through."""
        classifier = NERClassifier(model_name="my-custom/ner-model")
        assert classifier.model_name == "my-custom/ner-model"
    
    @pytest.mark.asyncio
    async def test_classify_when_disabled_returns_error(self):
        """Classification when disabled should return error."""
        classifier = NERClassifier(enabled=False)
        result = await classifier.classify("Hello world")
        
        assert result.error == "NER disabled"
        assert result.entities == []
        assert result.highest_tier == 0
    
    @pytest.mark.asyncio
    async def test_classify_with_mock_pipeline(self):
        """Classification should work with mocked pipeline."""
        classifier = NERClassifier(enabled=True)
        
        # Mock the pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {
                "word": "John Doe",
                "entity_group": "PER",
                "score": 0.95,
                "start": 10,
                "end": 18,
            },
            {
                "word": "Anthropic",
                "entity_group": "ORG",
                "score": 0.88,
                "start": 30,
                "end": 39,
            },
        ]
        
        classifier._pipeline = mock_pipeline
        classifier._initialized = True
        
        result = await classifier.classify("I am John Doe working at Anthropic")
        
        assert result.error is None
        assert len(result.entities) == 2
        assert result.highest_tier == 2  # PER is tier 2
        
        # Check person entity
        person = result.entities[0]
        assert person.text == "John Doe"
        assert person.entity_type == "PER"
        assert person.tier == 2
        assert person.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_classify_filters_low_confidence(self):
        """Low confidence entities should be filtered."""
        classifier = NERClassifier(enabled=True, confidence_threshold=0.8)
        
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"word": "John", "entity_group": "PER", "score": 0.9, "start": 0, "end": 4},
            {"word": "maybe", "entity_group": "PER", "score": 0.5, "start": 5, "end": 10},  # Low confidence
        ]
        
        classifier._pipeline = mock_pipeline
        classifier._initialized = True
        
        result = await classifier.classify("John maybe")
        
        assert len(result.entities) == 1
        assert result.entities[0].text == "John"


class TestNERClassifierGlobal:
    """Test global NER classifier functions."""
    
    def test_get_ner_classifier_returns_instance(self):
        """get_ner_classifier should return a classifier instance."""
        classifier = get_ner_classifier()
        assert isinstance(classifier, NERClassifier)
    
    def test_configure_ner_creates_new_instance(self):
        """configure_ner should create and configure a new classifier."""
        classifier = configure_ner(
            model_name="fast",
            device="cpu",
            confidence_threshold=0.8,
            enabled=True,
        )
        
        assert classifier.model_name == "dslim/bert-base-NER"
        assert classifier.device == "cpu"
        assert classifier.confidence_threshold == 0.8
        assert classifier.enabled is True
