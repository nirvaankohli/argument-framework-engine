from dataclasses import dataclass
from typing import Literal, Optional

RelationType = Literal["supports", "attacks"]


@dataclass
class Argument:

    id: str  # A1, A2, ...
    text: str  # the claim itself
    utterance_id: str  # which comment/line it came from
    speaker: Optional[str] = None
    confidence: Optional[float] = None  # 0–1 from the LLM

@dataclass
class Relation:

    source_id: str  # id of the source argument
    target_id: str  # id of the target argument
    type: RelationType  # supports or attacks
    confidence: Optional[float] = None  # 0–1 from the LLM

