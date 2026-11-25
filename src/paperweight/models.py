"""Typed artifacts passed between pipeline stages."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    arxiv_id: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    source_url: str | None = None


class Equation(BaseModel):
    latex: str
    label: str | None = None
    environment: str
    section: str | None = None


class Algorithm(BaseModel):
    caption: str | None = None
    label: str | None = None
    body: str
    section: str | None = None


class Figure(BaseModel):
    caption: str
    label: str | None = None
    kind: str = "image"
    files: list[str] = Field(default_factory=list)


class Section(BaseModel):
    title: str
    level: int
    text: str


class Paper(BaseModel):
    """Stage 1 output: the structured content extracted from a paper."""

    metadata: PaperMetadata
    source_format: str
    sections: list[Section] = Field(default_factory=list)
    equations: list[Equation] = Field(default_factory=list)
    algorithms: list[Algorithm] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    theorems: list[str] = Field(default_factory=list)


class TypeField(BaseModel):
    name: str
    type: str = Field(description="Type in Python typing notation, e.g. 'list[float]'.")
    description: str = ""
    constraints: list[str] = Field(default_factory=list)


class DataSchema(BaseModel):
    name: str
    description: str = ""
    fields: list[TypeField] = Field(default_factory=list)


class Invariant(BaseModel):
    id: str = Field(description="Stable identifier such as 'INV-1'.")
    statement: str = Field(description="Precise, testable natural-language statement.")
    formal: str = Field(default="", description="Formula form, LaTeX or Python expression.")
    source: str = Field(default="", description="Where in the paper this comes from.")
    applies_to: list[str] = Field(default_factory=list)


class ComplexityBound(BaseModel):
    operation: str
    time: str = ""
    space: str = ""
    source: str = ""


class FunctionSpec(BaseModel):
    name: str
    signature: str = Field(description="Python signature with type hints.")
    description: str = ""
    pseudocode: str = ""
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list, description="Invariant ids.")
    complexity: ComplexityBound | None = None


class Component(BaseModel):
    name: str
    module: str = Field(description="Dotted module path inside the target package.")
    responsibility: str = ""
    functions: list[FunctionSpec] = Field(default_factory=list)


class EdgeCase(BaseModel):
    id: str
    description: str
    expected_behavior: str
    component: str = ""


class TestKind(StrEnum):
    property = "property"
    unit = "unit"
    benchmark = "benchmark"


class TestCase(BaseModel):
    __test__ = False

    id: str
    kind: TestKind
    description: str
    target: str = Field(default="", description="Function or component under test.")
    invariants: list[str] = Field(default_factory=list)
    generator: str = Field(default="", description="Input generation strategy, if any.")


class Spec(BaseModel):
    """Stage 2 output: an engineering spec derived from the paper."""

    project_name: str = Field(description="Python package name, snake_case.")
    summary: str
    language: str = "python"
    schemas: list[DataSchema] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    complexity: list[ComplexityBound] = Field(default_factory=list)
    edge_cases: list[EdgeCase] = Field(default_factory=list)
    test_plan: list[TestCase] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class SessionResult(BaseModel):
    """Stage 3/4 output: the outcome of a Devin session."""

    stage: str
    session_id: str
    session_url: str
    status: str
    status_detail: str | None = None
    pr_urls: list[str] = Field(default_factory=list)
    branch: str | None = None
    structured_output: dict | None = None
