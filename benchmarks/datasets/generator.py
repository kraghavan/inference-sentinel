"""Synthetic dataset generator for privacy classification benchmarks.

Uses Faker to generate prompts with known PII at exact positions,
enabling precise accuracy measurement with zero ambiguity.

Usage:
    python -m benchmarks.datasets.generator --count 200 --output benchmarks/datasets/privacy_prompts.json
"""

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from faker import Faker

fake = Faker()
Faker.seed(42)  # Reproducibility
random.seed(42)


@dataclass
class Entity:
    """A detected/expected entity in a prompt."""
    type: str           # e.g., "SSN", "EMAIL", "CREDIT_CARD"
    value: str          # The actual value
    start: int          # Character offset start
    end: int            # Character offset end


@dataclass
class LabeledPrompt:
    """A prompt with ground truth labels."""
    id: str
    text: str
    expected_tier: int
    tier_name: str
    entities: list[Entity] = field(default_factory=list)
    category: str = ""  # e.g., "customer_support", "code_review", "general"


# =============================================================================
# PROMPT TEMPLATES BY TIER
# =============================================================================

TIER_0_TEMPLATES = [
    # General knowledge
    "What is the capital of {country}?",
    "Explain how {concept} works in simple terms.",
    "Write a short poem about {topic}.",
    "What are the main differences between {thing1} and {thing2}?",
    "Summarize the history of {historical_topic}.",
    "How do I cook {dish}?",
    "What's the weather like in {city} during {season}?",
    "Recommend some books about {genre}.",
    "Explain the rules of {sport}.",
    "What are the benefits of {healthy_habit}?",
    # Code/technical (no PII)
    "Write a Python function to sort a list.",
    "Explain the difference between REST and GraphQL.",
    "How do I center a div in CSS?",
    "What is the time complexity of quicksort?",
    "Explain dependency injection in simple terms.",
    # Business (no PII)
    "Draft an agenda for a team meeting about project planning.",
    "What are best practices for code reviews?",
    "How should I structure a product roadmap?",
    "Explain agile vs waterfall methodology.",
    "What metrics should I track for a SaaS product?",
]

TIER_1_TEMPLATES = [
    # Internal business identifiers - using patterns that regex can catch
    "Look up employee id: {employee_id} in the HR system.",
    "Project code: {project_code}. What's the status?",
    "Check https://internal.company.com/tickets/{ticket_num} for updates.",
    "Our internal API is at {internal_url}. Is it responding?",
    "Access the project at https://intranet.corp.local/projects/{project_id}",
    "Employee number: {employee_id} needs access to the system.",
    "Visit https://private.company.internal/docs for documentation.",
    "The project id: {project_code} deadline is approaching.",
    "Employee id {employee_id} submitted a request.",
    "Go to https://corp.internal.net/wiki for the runbook.",
]

TIER_2_TEMPLATES = [
    # Contact information
    "Send a follow-up email to {email} about the proposal.",
    "Call {name} at {phone} to confirm the meeting.",
    "Ship the package to {name} at {address}.",
    "My email is {email}. Please send me the report.",
    "Contact {name} ({email}) for the interview schedule.",
    "The customer's phone number is {phone}. Call them back.",
    "Deliver to: {name}, {address}",
    "Please email {email} with the meeting notes.",
    "Reach out to {name} at {phone} or {email}.",
    "The billing address is {address} for account holder {name}.",
]

TIER_3_TEMPLATES = [
    # High sensitivity - SSN, credit cards, medical, financial
    "My social security number is {ssn}. Process my application.",
    "Please charge credit card number {credit_card} for the purchase.",
    "Patient {name}, MRN: {mrn}, has diagnosis requiring treatment.",
    "Bank account number: {bank_account}, routing number: {routing_number} for wire transfer.",
    "SSN: {ssn}, DOB: {dob}. Verify identity for loan application.",
    "Credit card {credit_card} expiring {exp_date} with security code {cvv}.",
    "Medical record MRN {mrn} for patient {name}: Blood type {blood_type}, allergies: {allergies}.",
    "Tax filing for SSN {ssn}, taxpayer {name}. Process refund.",
    "Patient {name}, health insurance ID: {insurance_id}, prescribed {medication} {dosage}.",
    "Wire transfer: account number {bank_account}, routing {routing_number}, amount $50,000.",
]


# =============================================================================
# ENTITY GENERATORS
# =============================================================================

def generate_tier_0_prompt() -> LabeledPrompt:
    """Generate a public prompt with no PII."""
    template = random.choice(TIER_0_TEMPLATES)
    
    # Fill in placeholders with generic content
    text = template.format(
        country=fake.country(),
        concept=random.choice(["photosynthesis", "machine learning", "blockchain", "quantum computing"]),
        topic=random.choice(["nature", "technology", "space", "music"]),
        thing1=random.choice(["Python", "cats", "summer", "coffee"]),
        thing2=random.choice(["JavaScript", "dogs", "winter", "tea"]),
        historical_topic=random.choice(["the Roman Empire", "the Renaissance", "World War II"]),
        dish=random.choice(["pasta carbonara", "pad thai", "tacos"]),
        city=fake.city(),
        season=random.choice(["spring", "summer", "fall", "winter"]),
        genre=random.choice(["science fiction", "mystery", "history"]),
        sport=random.choice(["basketball", "soccer", "tennis"]),
        healthy_habit=random.choice(["meditation", "exercise", "reading"]),
    )
    
    return LabeledPrompt(
        id=f"tier0_{fake.uuid4()[:8]}",
        text=text,
        expected_tier=0,
        tier_name="PUBLIC",
        entities=[],
        category="general",
    )


def generate_tier_1_prompt() -> LabeledPrompt:
    """Generate an internal prompt with business identifiers."""
    template = random.choice(TIER_1_TEMPLATES)
    entities = []
    
    # Generate fake internal identifiers that match regex patterns
    # employee_id pattern: (?i)\b(?:employee|emp)[\s_-]?(?:id|number|no|#)[:\s]*[A-Z0-9]{4,10}\b
    employee_id = f"{fake.random_number(digits=6, fix_len=True)}"
    
    # project_code pattern: (?i)\b(?:project|proj)[\s_-]?(?:id|code|number)[:\s]*[A-Z0-9-]{4,15}\b
    project_code = f"{fake.lexify('???').upper()}-{fake.random_number(digits=3)}"
    
    # internal_url pattern: https://internal.xxx or https://xxx.internal
    internal_url = f"https://internal.company.com/api/v{fake.random_int(1,3)}/{fake.slug()}"
    
    # Other values for templates
    ticket_num = fake.random_number(digits=5)
    project_id = fake.lexify('????').lower()
    
    text = template.format(
        employee_id=employee_id,
        project_code=project_code,
        internal_url=internal_url,
        ticket_num=ticket_num,
        project_id=project_id,
    )
    
    # Track entities (find their positions in the final text)
    for entity_type, value in [
        ("EMPLOYEE_ID", employee_id),
        ("PROJECT_CODE", project_code),
        ("INTERNAL_URL", internal_url),
    ]:
        start = text.find(str(value))
        if start != -1:
            entities.append(Entity(entity_type, str(value), start, start + len(str(value))))
    
    return LabeledPrompt(
        id=f"tier1_{fake.uuid4()[:8]}",
        text=text,
        expected_tier=1,
        tier_name="INTERNAL",
        entities=entities,
        category="internal",
    )


def generate_tier_2_prompt() -> LabeledPrompt:
    """Generate a confidential prompt with contact information."""
    template = random.choice(TIER_2_TEMPLATES)
    entities = []
    
    name = fake.name()
    email = fake.email()
    # Generate phone in standard format (no extensions)
    phone = f"{fake.random_int(200, 999)}-{fake.random_int(200, 999)}-{fake.random_int(1000, 9999)}"
    # Use simpler address format that regex can catch
    street_num = fake.random_int(100, 9999)
    street_name = fake.last_name()
    street_type = random.choice(["Street", "Avenue", "Road", "Boulevard", "Drive"])
    address = f"{street_num} {street_name} {street_type}, {fake.city()}, {fake.state_abbr()} {fake.zipcode()}"
    
    text = template.format(
        name=name,
        email=email,
        phone=phone,
        address=address,
    )
    
    # Track entities
    for entity_type, value in [
        ("PERSON_NAME", name),
        ("EMAIL", email),
        ("PHONE", phone),
        ("ADDRESS", address),
    ]:
        start = text.find(value)
        if start != -1:
            entities.append(Entity(entity_type, value, start, start + len(value)))
    
    return LabeledPrompt(
        id=f"tier2_{fake.uuid4()[:8]}",
        text=text,
        expected_tier=2,
        tier_name="CONFIDENTIAL",
        entities=entities,
        category="contact",
    )


def generate_tier_3_prompt() -> LabeledPrompt:
    """Generate a restricted prompt with high-sensitivity PII."""
    template = random.choice(TIER_3_TEMPLATES)
    entities = []
    
    name = fake.name()
    ssn = fake.ssn()  # Already generates XXX-XX-XXXX format
    
    # Generate valid credit card format (Visa starts with 4, 16 digits)
    credit_card = f"4{fake.random_number(digits=15, fix_len=True)}"
    exp_date = fake.credit_card_expire()
    cvv = fake.credit_card_security_code()
    
    # Generate bank account with proper format (8-17 digits)
    bank_account = str(fake.random_number(digits=10, fix_len=True))
    routing_number = str(fake.random_number(digits=9, fix_len=True))
    
    dob = fake.date_of_birth(minimum_age=18, maximum_age=80).strftime("%m/%d/%Y")
    blood_type = random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
    allergies = random.choice(["Penicillin", "Peanuts", "None", "Latex"])
    medication = random.choice(["Metformin", "Lisinopril", "Atorvastatin"])
    dosage = random.choice(["10mg daily", "25mg twice daily", "500mg with meals"])
    
    # Generate MRN (medical record number)
    mrn = str(fake.random_number(digits=8, fix_len=True))
    
    # Health insurance ID
    insurance_id = f"{fake.lexify('???').upper()}{fake.random_number(digits=9, fix_len=True)}"
    
    text = template.format(
        name=name,
        ssn=ssn,
        credit_card=credit_card,
        exp_date=exp_date,
        cvv=cvv,
        bank_account=bank_account,
        routing_number=routing_number,
        dob=dob,
        blood_type=blood_type,
        allergies=allergies,
        medication=medication,
        dosage=dosage,
        mrn=mrn,
        insurance_id=insurance_id,
    )
    
    # Track high-sensitivity entities
    for entity_type, value in [
        ("SSN", ssn),
        ("CREDIT_CARD", credit_card),
        ("BANK_ACCOUNT", bank_account),
        ("ROUTING_NUMBER", routing_number),
        ("MRN", mrn),
        ("PERSON_NAME", name),
    ]:
        start = text.find(value)
        if start != -1:
            entities.append(Entity(entity_type, value, start, start + len(value)))
        if start != -1:
            entities.append(Entity(entity_type, value, start, start + len(value)))
    
    return LabeledPrompt(
        id=f"tier3_{fake.uuid4()[:8]}",
        text=text,
        expected_tier=3,
        tier_name="RESTRICTED",
        entities=entities,
        category="sensitive",
    )


# =============================================================================
# DATASET GENERATION
# =============================================================================

GENERATORS = {
    0: generate_tier_0_prompt,
    1: generate_tier_1_prompt,
    2: generate_tier_2_prompt,
    3: generate_tier_3_prompt,
}


def generate_dataset(
    count: int = 200,
    distribution: dict[int, float] | None = None,
) -> list[LabeledPrompt]:
    """Generate a balanced dataset of labeled prompts.
    
    Args:
        count: Total number of prompts to generate.
        distribution: Optional dict mapping tier -> proportion (must sum to 1.0).
                     Default is equal distribution across all 4 tiers.
    
    Returns:
        List of LabeledPrompt objects with ground truth.
    """
    if distribution is None:
        distribution = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    
    dataset = []
    
    for tier, proportion in distribution.items():
        tier_count = int(count * proportion)
        generator = GENERATORS[tier]
        
        for _ in range(tier_count):
            prompt = generator()
            dataset.append(prompt)
    
    # Shuffle to avoid tier clustering
    random.shuffle(dataset)
    
    return dataset


def save_dataset(dataset: list[LabeledPrompt], output_path: Path) -> None:
    """Save dataset to JSON file."""
    data = {
        "metadata": {
            "version": "1.0",
            "total_count": len(dataset),
            "tier_distribution": {
                tier: sum(1 for p in dataset if p.expected_tier == tier)
                for tier in range(4)
            },
        },
        "prompts": [
            {
                **asdict(p),
                "entities": [asdict(e) for e in p.entities],
            }
            for p in dataset
        ],
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(dataset)} prompts → {output_path}")
    print(f"Distribution: {data['metadata']['tier_distribution']}")


def load_dataset(input_path: Path) -> list[LabeledPrompt]:
    """Load dataset from JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    
    prompts = []
    for p in data["prompts"]:
        entities = [Entity(**e) for e in p.get("entities", [])]
        prompts.append(LabeledPrompt(
            id=p["id"],
            text=p["text"],
            expected_tier=p["expected_tier"],
            tier_name=p["tier_name"],
            entities=entities,
            category=p.get("category", ""),
        ))
    
    return prompts


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic privacy benchmark dataset")
    parser.add_argument("--count", type=int, default=200, help="Total prompts to generate")
    parser.add_argument("--output", type=str, default="benchmarks/datasets/privacy_prompts.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Reset seeds
    Faker.seed(args.seed)
    random.seed(args.seed)
    
    dataset = generate_dataset(count=args.count)
    save_dataset(dataset, Path(args.output))


if __name__ == "__main__":
    main()
