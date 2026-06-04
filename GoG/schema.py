from pydantic import BaseModel, Field
from typing import List, Optional, Set, Tuple, Union, Literal


class QueryEntry(BaseModel):
    id: str
    question: str
    answers: set[str] = Field(description="Contain answers, even hard answers")
    hard_answer: set[str]
    prediction: set[str] | None = None

    @property
    def overlap(self) -> set[str]:
        if self.prediction is None:
            return set()
        return self.answers.intersection(self.prediction)

    @property
    def hit_any(self) -> bool:
        return len(self.overlap) > 0
    
    @property
    def hit_hard(self) -> bool:
        if self.prediction is None:
            return False
        return len(self.hard_answer.intersection(self.prediction)) > 0
    
    @property
    def precision(self) -> float:
        if self.prediction is None or len(self.prediction) == 0:
            return 0.0
        return len(self.overlap) / len(self.prediction)
    
    @property
    def recall(self) -> float:
        if len(self.answers) == 0:
            return 0.0
        return len(self.overlap) / len(self.answers)
    
    @property
    def f1_score(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    

class ReasoningChain(BaseModel):
    id: str
    triplets: list[tuple[str, str, str]] | list[list[str]]  # Allow both tuple and list representations of triples
    # candidates: set[str] | None = None
    # status: Literal["expandable", "complete_promised", "complete_unpromised", None] = None


class SearchParams(BaseModel):
    entities: list[str] = Field(description="The entities to search in the knowledge graph.")


class GenerateParams(BaseModel):
    query: str = Field(description="The open-ended natural language query to generate triples from.")


class FinishParams(BaseModel):
    entities: list[str] = Field(description="The final answer entities.")


class SearchAction(BaseModel):
    type: Literal["search"]
    params: SearchParams


class GenerateAction(BaseModel):
    type: Literal["generate"]
    params: GenerateParams


class FinishAction(BaseModel):
    type: Literal["finish"]
    params: FinishParams


Action = Union[SearchAction, GenerateAction, FinishAction]


class PlannerResponse(BaseModel):
    thought: str = Field(description="Your thought process and reasoning for the current step.")
    action: Action


class FallbackOutput(BaseModel):
    thought: str = Field(description="Your thought process and reasoning for selecting the candidates.")
    candidates: list[str] = Field(description="The list of selected candidate entities.")


class Triplet(BaseModel):
    subject: str
    relation: str
    object: str


class ChainExpansion(BaseModel):
    id: str
    triplets_to_append: list[Triplet] = Field(description="List of triplets to append to the existing chain.")


class NewChain(BaseModel):
    id: str
    triplets: list[Triplet] = Field(description="List of triplets to include in the new chain.")


class GatherChainOutput(BaseModel):
    thought: str = Field(description="Your thought process and reasoning for gathering the triplets into reasoning chains.")
    chain_expansion: list[ChainExpansion] = Field(description="List of existing chains to expand with the triplets to append.")
    new_chains: list[NewChain] = Field(description="List of new chains to create with the triplets.")


class FilterRelationOutput(BaseModel):
    thought: str = Field(description="Your thought process and reasoning for selecting the relations.")
    relations: list[str] = Field(description="The list of selected relations.")
