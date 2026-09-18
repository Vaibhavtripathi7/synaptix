"""Typed outputs of the analysis agents. Every item carries `evidence`: the passage numbers it
was drawn from, so an analysis is auditable back to pages of the source document."""

from pydantic import BaseModel, Field


class Point(BaseModel):
    text: str
    evidence: list[int] = Field(default_factory=list, description="passage numbers [n] used")


class CompetitorProfile(BaseModel):
    name: str = Field(description="the company the document is about")
    basic_information: list[Point] = Field(description="location, industry, history, size")
    products_and_services: list[Point]
    market_presence: list[Point] = Field(description="audience, share, reach, marketing")
    sales: list[Point] = Field(description="revenue, pricing, channels, customers")


class Swot(BaseModel):
    strengths: list[Point]
    weaknesses: list[Point]
    opportunities: list[Point]
    threats: list[Point]


class Strategy(BaseModel):
    likely_strategy: list[Point] = Field(description="the actions the company is likely to take")
    impact: list[Point] = Field(description="threats and opportunities this creates for others")
    counter_strategies: list[Point]


class KeyAnalysis(BaseModel):
    executive_summary: str
    advantages: list[Point]
    disadvantages: list[Point]
    recommendations: list[Point]


class Comparison(BaseModel):
    differences: list[Point]
    similarities: list[Point]
    first_can_learn: list[Point] = Field(description="what the first can learn from the second")
    second_can_learn: list[Point] = Field(description="what the second can learn from the first")
