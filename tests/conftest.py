from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperweight.models import Spec

FIXTURES = Path(__file__).parent / "fixtures"

SPEC_JSON = {
    "project_name": "fast_smoothing",
    "summary": "Online exponential smoothing with O(1) updates.",
    "schemas": [
        {
            "name": "SmootherState",
            "fields": [{"name": "value", "type": "float", "constraints": ["finite"]}],
        }
    ],
    "components": [
        {
            "name": "Smoother",
            "module": "fast_smoothing.smoother",
            "functions": [
                {
                    "name": "update",
                    "signature": "def update(state: float, x: float, alpha: float) -> float",
                    "preconditions": ["0 < alpha <= 1"],
                    "invariants": ["INV-1"],
                    "complexity": {"operation": "update", "time": "O(1)", "space": "O(1)"},
                    "pseudocode": "return alpha * x + (1 - alpha) * state",
                }
            ],
        }
    ],
    "invariants": [
        {
            "id": "INV-1",
            "statement": "Output lies between min and max of inputs.",
            "formal": "min(x) <= s_t <= max(x)",
            "source": "Theorem 1",
        }
    ],
    "complexity": [{"operation": "update", "time": "O(1)", "space": "O(1)"}],
    "edge_cases": [
        {"id": "EC-1", "description": "alpha = 0", "expected_behavior": "raise ValueError"}
    ],
    "test_plan": [
        {"id": "T-1", "kind": "property", "description": "bounded output", "invariants": ["INV-1"]}
    ],
}


@pytest.fixture
def sample_tex() -> str:
    return (FIXTURES / "sample.tex").read_text()


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURES


@pytest.fixture
def spec_json() -> dict:
    return json.loads(json.dumps(SPEC_JSON))


@pytest.fixture
def spec(spec_json: dict) -> Spec:
    return Spec.model_validate(spec_json)


class FakeLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 16000) -> str:
        self.calls.append((system, user))
        return self.replies.pop(0)


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
