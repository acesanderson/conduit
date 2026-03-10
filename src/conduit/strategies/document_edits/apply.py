from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.strategies.document_edits.models import EditOp


class EditApplicationError(Exception):
    pass


def apply_edits(document: str, edits: list[EditOp]) -> str:
    from conduit.strategies.document_edits.models import EditType

    current = document
    for op in edits:
        count = current.count(op.search)
        if count == 0:
            raise EditApplicationError(
                f"Search string not found in document: {op.search!r}"
            )
        if count > 1:
            raise EditApplicationError(
                f"Search string is ambiguous ({count} occurrences): {op.search!r}"
            )
        if op.type == EditType.replace:
            current = current.replace(op.search, op.replace, 1)
        elif op.type == EditType.insert:
            idx = current.index(op.search)
            insert_at = idx + len(op.search)
            current = current[:insert_at] + op.replace + current[insert_at:]
        elif op.type == EditType.delete:
            current = current.replace(op.search, "", 1)
    return current
