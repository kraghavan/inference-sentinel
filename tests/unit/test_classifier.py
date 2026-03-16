"""Tests for the regex classifier."""

import pytest

from sentinel.classification import (
    ClassificationResult,
    RegexClassifier,
    classify,
    get_tier_label,
    load_taxonomy,
)


@pytest.fixture
def classifier() -> RegexClassifier:
    """Create a classifier with the default taxonomy."""
    taxonomy = load_taxonomy()
    return RegexClassifier(taxonomy)


class TestTier3Detection:
    """Tests for Tier 3 (RESTRICTED) entity detection."""

    def test_ssn_with_dashes(self, classifier: RegexClassifier) -> None:
        """Test SSN detection with dash format."""
        result = classifier.classify("My SSN is 123-45-6789")
        assert result.tier == 3
        assert result.tier_label == "RESTRICTED"
        assert "ssn" in result.entity_types

    def test_ssn_with_spaces(self, classifier: RegexClassifier) -> None:
        """Test SSN detection with space format."""
        result = classifier.classify("SSN: 123 45 6789")
        assert result.tier == 3
        assert "ssn" in result.entity_types

    def test_ssn_with_label(self, classifier: RegexClassifier) -> None:
        """Test SSN with explicit label."""
        result = classifier.classify("ssn:123-45-6789")
        assert result.tier == 3
        assert "ssn" in result.entity_types

    def test_credit_card_visa(self, classifier: RegexClassifier) -> None:
        """Test Visa card detection."""
        result = classifier.classify("Card: 4111111111111111")
        assert result.tier == 3
        assert result.tier_label == "RESTRICTED"
        assert "credit_card" in result.entity_types

    def test_credit_card_mastercard(self, classifier: RegexClassifier) -> None:
        """Test Mastercard detection."""
        result = classifier.classify("Pay with 5500000000000004")
        assert result.tier == 3
        assert "credit_card" in result.entity_types

    def test_credit_card_amex(self, classifier: RegexClassifier) -> None:
        """Test Amex card detection."""
        result = classifier.classify("Amex: 378282246310005")
        assert result.tier == 3
        assert "credit_card" in result.entity_types

    def test_bank_account(self, classifier: RegexClassifier) -> None:
        """Test bank account detection."""
        result = classifier.classify("Account number: 12345678901234")
        assert result.tier == 3
        assert "bank_account" in result.entity_types

    def test_routing_number(self, classifier: RegexClassifier) -> None:
        """Test routing number detection."""
        result = classifier.classify("Routing: 021000021")
        assert result.tier == 3
        assert "bank_account" in result.entity_types

    def test_medical_record_number(self, classifier: RegexClassifier) -> None:
        """Test MRN detection."""
        result = classifier.classify("Patient MRN: 12345678")
        assert result.tier == 3
        assert "health_record" in result.entity_types

    def test_npi_number(self, classifier: RegexClassifier) -> None:
        """Test NPI detection."""
        result = classifier.classify("Provider NPI: 1234567890")
        assert result.tier == 3
        assert "health_record" in result.entity_types


class TestTier2Detection:
    """Tests for Tier 2 (CONFIDENTIAL) entity detection."""

    def test_email_address(self, classifier: RegexClassifier) -> None:
        """Test email detection."""
        result = classifier.classify("Contact me at john.doe@example.com")
        assert result.tier == 2
        assert result.tier_label == "CONFIDENTIAL"
        assert "email" in result.entity_types

    def test_phone_number_dashes(self, classifier: RegexClassifier) -> None:
        """Test phone number with dashes."""
        result = classifier.classify("Call me at 555-123-4567")
        assert result.tier == 2
        assert "phone" in result.entity_types

    def test_phone_number_parens(self, classifier: RegexClassifier) -> None:
        """Test phone number with parentheses."""
        result = classifier.classify("Phone: (555) 123-4567")
        assert result.tier == 2
        assert "phone" in result.entity_types

    def test_street_address(self, classifier: RegexClassifier) -> None:
        """Test street address detection."""
        result = classifier.classify("Ship to 123 Main Street")
        assert result.tier == 2
        assert "address" in result.entity_types

    def test_dob_with_label(self, classifier: RegexClassifier) -> None:
        """Test date of birth detection."""
        result = classifier.classify("DOB: 01/15/1990")
        assert result.tier == 2
        assert "dob" in result.entity_types

    def test_openai_api_key(self, classifier: RegexClassifier) -> None:
        """Test OpenAI API key detection."""
        result = classifier.classify("OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456")
        assert result.tier == 2
        assert "api_key" in result.entity_types

    def test_anthropic_api_key(self, classifier: RegexClassifier) -> None:
        """Test Anthropic API key detection."""
        result = classifier.classify("Key: sk-ant-api03-abcdefghijklmnopqrstuvwxyz12345678901234")
        assert result.tier == 2
        assert "api_key" in result.entity_types

    def test_aws_access_key(self, classifier: RegexClassifier) -> None:
        """Test AWS access key detection."""
        result = classifier.classify("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
        assert result.tier == 2
        assert "api_key" in result.entity_types

    def test_github_token(self, classifier: RegexClassifier) -> None:
        """Test GitHub token detection."""
        result = classifier.classify("GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert result.tier == 2
        assert "api_key" in result.entity_types

    def test_password_in_text(self, classifier: RegexClassifier) -> None:
        """Test password detection."""
        result = classifier.classify("password: mysecretpassword123")
        assert result.tier == 2
        assert "password" in result.entity_types

    def test_private_key_header(self, classifier: RegexClassifier) -> None:
        """Test private key detection."""
        result = classifier.classify("-----BEGIN RSA PRIVATE KEY-----\nMIIE...")
        assert result.tier == 2
        assert "private_key" in result.entity_types

    def test_internal_ip_10(self, classifier: RegexClassifier) -> None:
        """Test internal IP detection (10.x.x.x)."""
        result = classifier.classify("Server at 10.0.1.25")
        assert result.tier == 2
        assert "ip_address_internal" in result.entity_types

    def test_internal_ip_192(self, classifier: RegexClassifier) -> None:
        """Test internal IP detection (192.168.x.x)."""
        result = classifier.classify("Connect to 192.168.1.100")
        assert result.tier == 2
        assert "ip_address_internal" in result.entity_types


class TestTier1Detection:
    """Tests for Tier 1 (INTERNAL) entity detection."""

    def test_employee_id(self, classifier: RegexClassifier) -> None:
        """Test employee ID detection."""
        result = classifier.classify("Employee ID: EMP12345")
        assert result.tier == 1
        assert result.tier_label == "INTERNAL"
        assert "employee_id" in result.entity_types

    def test_project_code(self, classifier: RegexClassifier) -> None:
        """Test project code detection."""
        result = classifier.classify("Project code: PROJ-2024-001")
        assert result.tier == 1
        assert "project_code" in result.entity_types

    def test_internal_url(self, classifier: RegexClassifier) -> None:
        """Test internal URL detection."""
        result = classifier.classify("See https://internal.company.com/docs")
        assert result.tier == 1
        assert "internal_url" in result.entity_types


class TestTier0PublicContent:
    """Tests for Tier 0 (PUBLIC) - no sensitive content."""

    def test_general_question(self, classifier: RegexClassifier) -> None:
        """Test general knowledge question."""
        result = classifier.classify("What is the capital of France?")
        assert result.tier == 0
        assert result.tier_label == "PUBLIC"
        assert len(result.entities_detected) == 0

    def test_code_snippet(self, classifier: RegexClassifier) -> None:
        """Test code snippet without secrets."""
        result = classifier.classify("def hello(): print('Hello, World!')")
        assert result.tier == 0
        assert len(result.entities_detected) == 0

    def test_general_text(self, classifier: RegexClassifier) -> None:
        """Test general prose."""
        result = classifier.classify(
            "The weather today is sunny with a high of 75 degrees."
        )
        assert result.tier == 0
        assert len(result.entities_detected) == 0


class TestMultipleEntities:
    """Tests for detecting multiple entities."""

    def test_multiple_tier3_entities(self, classifier: RegexClassifier) -> None:
        """Test multiple Tier 3 entities."""
        result = classifier.classify(
            "SSN: 123-45-6789, Card: 4111111111111111"
        )
        assert result.tier == 3
        assert "ssn" in result.entity_types
        assert "credit_card" in result.entity_types
        assert result.entity_count >= 2

    def test_mixed_tier_entities(self, classifier: RegexClassifier) -> None:
        """Test that highest tier is returned."""
        result = classifier.classify(
            "Email: john@example.com, SSN: 123-45-6789"
        )
        assert result.tier == 3  # SSN is tier 3, email is tier 2
        assert "ssn" in result.entity_types
        assert "email" in result.entity_types

    def test_multiple_emails(self, classifier: RegexClassifier) -> None:
        """Test multiple emails detected."""
        result = classifier.classify(
            "Contact alice@example.com or bob@example.com"
        )
        assert result.tier == 2
        assert result.entity_types.count("email") == 1  # Unique types
        assert result.entity_count >= 2  # But multiple entities


class TestClassifyMessages:
    """Tests for classify_messages function."""

    def test_classify_single_message(self, classifier: RegexClassifier) -> None:
        """Test classifying a single message."""
        messages = [{"role": "user", "content": "My SSN is 123-45-6789"}]
        result = classifier.classify_messages(messages)
        assert result.tier == 3
        assert "ssn" in result.entity_types

    def test_classify_multiple_messages(self, classifier: RegexClassifier) -> None:
        """Test classifying multiple messages."""
        messages = [
            {"role": "user", "content": "My email is test@example.com"},
            {"role": "assistant", "content": "Got it!"},
            {"role": "user", "content": "And my SSN is 123-45-6789"},
        ]
        result = classifier.classify_messages(messages)
        assert result.tier == 3
        assert "email" in result.entity_types
        assert "ssn" in result.entity_types


class TestQuickCheck:
    """Tests for quick_check method."""

    def test_quick_check_positive(self, classifier: RegexClassifier) -> None:
        """Test quick_check returns True for sensitive content."""
        assert classifier.quick_check("SSN: 123-45-6789") is True

    def test_quick_check_negative(self, classifier: RegexClassifier) -> None:
        """Test quick_check returns False for safe content."""
        assert classifier.quick_check("Hello, World!") is False

    def test_quick_check_min_tier(self, classifier: RegexClassifier) -> None:
        """Test quick_check with min_tier parameter."""
        text = "Email: test@example.com"  # Tier 2
        assert classifier.quick_check(text, min_tier=2) is True
        assert classifier.quick_check(text, min_tier=3) is False


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_tier_label(self) -> None:
        """Test get_tier_label function."""
        assert get_tier_label(0) == "PUBLIC"
        assert get_tier_label(1) == "INTERNAL"
        assert get_tier_label(2) == "CONFIDENTIAL"
        assert get_tier_label(3) == "RESTRICTED"
        assert get_tier_label(99) == "PUBLIC"  # Unknown tier

    def test_classify_convenience_function(self) -> None:
        """Test the module-level classify function."""
        result = classify("SSN: 123-45-6789")
        assert result.tier == 3


class TestClassificationResult:
    """Tests for ClassificationResult properties."""

    def test_is_sensitive_true(self, classifier: RegexClassifier) -> None:
        """Test is_sensitive property returns True for tier > 0."""
        result = classifier.classify("Email: test@example.com")
        assert result.is_sensitive is True

    def test_is_sensitive_false(self, classifier: RegexClassifier) -> None:
        """Test is_sensitive property returns False for tier 0."""
        result = classifier.classify("Hello, World!")
        assert result.is_sensitive is False

    def test_requires_local_true(self, classifier: RegexClassifier) -> None:
        """Test requires_local property returns True for tier >= 2."""
        result = classifier.classify("SSN: 123-45-6789")
        assert result.requires_local is True

    def test_requires_local_false(self, classifier: RegexClassifier) -> None:
        """Test requires_local property returns False for tier < 2."""
        result = classifier.classify("Project code: PROJ-123")
        assert result.requires_local is False

    def test_to_dict(self, classifier: RegexClassifier) -> None:
        """Test to_dict serialization."""
        result = classifier.classify("Email: test@example.com")
        d = result.to_dict()
        assert d["tier"] == 2
        assert d["tier_label"] == "CONFIDENTIAL"
        assert "email" in d["entity_types"]


class TestPerformance:
    """Performance-related tests."""

    def test_classification_latency(self, classifier: RegexClassifier) -> None:
        """Test that classification latency is recorded."""
        result = classifier.classify("Some text with SSN: 123-45-6789")
        assert result.detection_latency_ms > 0
        assert result.detection_latency_ms < 100  # Should be fast

    def test_large_text_performance(self, classifier: RegexClassifier) -> None:
        """Test classification of large text is still fast."""
        # Generate a large text (~100KB)
        large_text = "Hello, World! " * 10000
        large_text += "SSN: 123-45-6789"  # Add sensitive content at end

        result = classifier.classify(large_text)
        assert result.tier == 3
        assert result.detection_latency_ms < 1000  # Should complete in <1s
