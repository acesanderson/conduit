from pydantic import BaseModel
from typing import Optional


class PerplexityCitation(BaseModel):
    title: str
    url: str
    date: Optional[str]


class PerplexityContent(BaseModel):
    text: str
    citations: list[PerplexityCitation]
