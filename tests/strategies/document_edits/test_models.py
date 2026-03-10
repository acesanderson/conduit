from __future__ import annotations
import pytest
from pydantic import ValidationError
from conduit.strategies.document_edits.models import EditType, EditOp, DocumentEdits


def test_edit_type_values():
    assert EditType.replace == "replace"
    assert EditType.insert == "insert"
    assert EditType.delete == "delete"


def test_edit_op_fields():
    op = EditOp(type=EditType.replace, search="foo", replace="bar")
    assert op.type == EditType.replace
    assert op.search == "foo"
    assert op.replace == "bar"


def test_edit_op_replace_defaults_to_empty_string():
    op = EditOp(type=EditType.delete, search="foo")
    assert op.replace == ""


def test_document_edits_max_length_edits():
    ops = [EditOp(type=EditType.replace, search=f"s{i}", replace=f"r{i}") for i in range(21)]
    with pytest.raises(ValidationError):
        DocumentEdits(edits=ops, summary="too many")


def test_document_edits_max_length_summary():
    op = EditOp(type=EditType.replace, search="x", replace="y")
    with pytest.raises(ValidationError):
        DocumentEdits(edits=[op], summary="x" * 201)


def test_document_edits_valid():
    op = EditOp(type=EditType.replace, search="hello", replace="world")
    doc_edits = DocumentEdits(edits=[op], summary="swap hello for world")
    assert len(doc_edits.edits) == 1
    assert doc_edits.summary == "swap hello for world"


def test_prompt_template_renders_with_required_variables():
    from conduit.strategies.document_edits.prompt import PROMPT_TEMPLATE
    from conduit.core.prompt.prompt import Prompt
    rendered = Prompt(PROMPT_TEMPLATE).render(
        {"user_prompt": "Fix the typo", "document": "Helo world"}
    )
    assert "Fix the typo" in rendered
    assert "Helo world" in rendered
