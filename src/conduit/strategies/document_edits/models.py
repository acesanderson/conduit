from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class EditType(str, Enum):
    replace = "replace"
    insert = "insert"
    delete = "delete"


class EditOp(BaseModel):
    type: EditType
    search: str
    replace: str = ""


class DocumentEdits(BaseModel):
    edits: list[EditOp] = Field(..., max_length=20)
    summary: str = Field(..., max_length=200)
