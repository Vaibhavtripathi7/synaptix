"""Typed outputs of the analysis agents. Every item carries `evidence`: the passage numbers it
was drawn from, so an analysis is auditable back to pages of the source document."""

from pydantic import BaseModel, Field


class Point(BaseModel):
    text: str
    evidence: list[int] = Field(default_factory=list, description="passage numbers [n] used")


class Concept(BaseModel):
    name: str
    explanation: str = Field(description="two or three sentences, in the document's own terms")
    evidence: list[int] = Field(default_factory=list)


class Formula(BaseModel):
    name: str
    latex: str = Field(description="the expression as LaTeX, without surrounding $ delimiters")
    symbols: str = Field(description="what each symbol means, and the units where given")
    evidence: list[int] = Field(default_factory=list)


class Question(BaseModel):
    question: str
    answer: str = Field(description="the worked answer, with the reasoning stated briefly")
    difficulty: str = Field(description="recall | application | analysis")
    evidence: list[int] = Field(default_factory=list)


class StudyOutline(BaseModel):
    overview: str = Field(description="what this material covers, in three or four sentences")
    key_concepts: list[Concept]
    formulas: list[Formula]
    worked_examples: list[Point]
    pitfalls: list[Point] = Field(description="conditions, assumptions and common mistakes")


class PracticeSet(BaseModel):
    questions: list[Question]


class PaperComparison(BaseModel):
    problem: list[Point] = Field(description="the problem each paper addresses")
    methods: list[Point]
    data_and_setup: list[Point] = Field(description="datasets, benchmarks, experimental setup")
    results: list[Point]
    strengths_and_limitations: list[Point]
    takeaway: str = Field(description="how the two relate, and when to prefer each")
