from __future__ import annotations
import pytest
from conduit.strategies.document_edits.models import DocumentEdits, EditOp, EditType
from conduit.strategies.document_edits.apply import apply_edits, EditApplicationError


def test_replace_op_substitutes_first_occurrence():
    doc = "the cat sat on the mat"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.replace, search="cat", replace="dog")],
        summary="replace cat",
    )
    result = apply_edits(doc, edits.edits)
    assert result == "the dog sat on the mat"


def test_replace_ops_are_sequential_against_current_state():
    doc = "aaa bbb"
    edits = DocumentEdits(
        edits=[
            EditOp(type=EditType.replace, search="aaa", replace="xxx"),
            EditOp(type=EditType.replace, search="xxx", replace="zzz"),
        ],
        summary="chain",
    )
    result = apply_edits(doc, edits.edits)
    assert result == "zzz bbb"


def test_insert_op_appends_after_search_anchor():
    doc = "hello world"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.insert, search="hello", replace=" beautiful")],
        summary="insert",
    )
    result = apply_edits(doc, edits.edits)
    assert result == "hello beautiful world"


def test_delete_op_removes_search_text():
    doc = "the quick brown fox"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.delete, search=" brown")],
        summary="delete",
    )
    result = apply_edits(doc, edits.edits)
    assert result == "the quick fox"


def test_raises_when_search_not_found():
    doc = "hello world"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.replace, search="missing", replace="x")],
        summary="not found",
    )
    with pytest.raises(EditApplicationError, match="not found"):
        apply_edits(doc, edits.edits)


def test_raises_when_search_is_ambiguous():
    doc = "cat and cat"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.replace, search="cat", replace="dog")],
        summary="ambiguous",
    )
    with pytest.raises(EditApplicationError, match="ambiguous"):
        apply_edits(doc, edits.edits)
