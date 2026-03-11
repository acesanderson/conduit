# DocumentEdit Strategy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a stateless `DocumentEditStrategy` that asks an LLM for a structured list of surgical edits and applies them to a document, returning the final text.

**Architecture:** A `Strategy` subclass with `@step` on `__call__`. All runtime config (model, temperature, etc.) flows through `get_param()` / `ConduitHarness`. The strategy constructs its own fresh `GenerationParams` and `ConduitOptions` internally, calls `ConduitAsync.run()`, extracts the `parsed` field from the last assistant message, and applies edits sequentially against the running document state.

**Tech Stack:** Python 3.12+, Pydantic v2, `conduit.core.conduit.ConduitAsync`, `conduit.core.workflow.step` (`@step`, `get_param`), `conduit.core.workflow.harness.ConduitHarness`, pytest + `unittest.mock`.

---

## Acceptance Criteria

| ID | Criterion |
|----|-----------|
| AC-1 | `EditType`, `EditOp`, `DocumentEdits` models exist; `edits` has `max_length=20`; `summary` has `max_length=200` |
| AC-2 | `apply_edits()` applies `replace` op against current doc state (first occurrence of `search` → `replace`) |
| AC-3 | `apply_edits()` applies `insert` op: appends `replace` text immediately after `search` anchor |
| AC-4 | `apply_edits()` applies `delete` op: removes first occurrence of `search` |
| AC-5 | `apply_edits()` raises `EditApplicationError` when `search` is not found |
| AC-6 | `apply_edits()` raises `EditApplicationError` when `search` appears more than once |
| AC-7 | `DocumentEditStrategy.__call__` constructs fresh `GenerationParams` with `output_type="structured_response"` and `response_model=DocumentEdits` |
| AC-8 | `DocumentEditStrategy.__call__` constructs `ConduitOptions` with `include_history=False` |
| AC-9 | `DocumentEditStrategy.__call__` returns the document text after calling `ConduitAsync.run()` and applying parsed edits |
| AC-10 | `DocumentEditStrategy.__call__` raises `TypeError` if the `parsed` field of the last message is not a `DocumentEdits` instance |
| AC-11 | `DocumentEditStrategy` is a concrete `Strategy` subclass; `@step` on `__call__` satisfies `Strategy.__init_subclass__` enforcement |
| AC-12 | `PROMPT_TEMPLATE` renders with Jinja2 variables `user_prompt` and `document` without error |

---

## File Layout

```
src/conduit/strategies/document_edits/   ← Python package (underscore, not hyphen)
  __init__.py
  models.py       ← EditType, EditOp, DocumentEdits
  apply.py        ← apply_edits(), EditApplicationError
  prompt.py       ← PROMPT_TEMPLATE constant
  strategy.py     ← DocumentEditStrategy

tests/strategies/
  __init__.py
  document_edits/
    __init__.py
    test_models.py
    test_apply.py
    test_strategy.py
```

**Note:** The design doc lives at `src/conduit/strategies/document-edits/SPEC.md` (hyphen, already exists). The Python module uses an underscore: `document_edits`. These are siblings in the `strategies/` directory.

---

## Key background for implementers

### How `@step` and `get_param` work

`@step` wraps `__call__` in a `StepWrapper`. When the strategy is called, `StepWrapper.__call__` binds all positional and keyword arguments and stores them in `context.args` (a `ContextVar`). `get_param("model", default="gpt3")` reads from `context.args` first, then the harness config, then falls back to the code default **only if** `use_defaults=True` was set in the harness.

In tests, always run the strategy through `ConduitHarness(use_defaults=True)`:

```python
harness = ConduitHarness(use_defaults=True)
result = await harness.run(strategy, document="...", user_prompt="...")
```

This sets `context.use_defaults = True` so code-level defaults in `get_param(key, default=...)` are honored.

### How `Strategy.__init_subclass__` enforces `@step`

`Strategy` checks that any concrete subclass that defines `__call__` wraps it with `StepWrapper`. Defining a concrete `DocumentEditStrategy` without `@step` will raise `TypeError` at class definition time — you don't even need to test it explicitly (AC-11 is satisfied by the class definition compiling without error).

### How to inspect `GenerationParams` / `ConduitOptions` in tests

Patch `ConduitAsync` at the strategy's import site, capture the args passed to `.run()`, and assert on the captured params/options:

```python
from unittest.mock import AsyncMock, patch, MagicMock
from conduit.domain.message.message import AssistantMessage
from conduit.domain.conversation.conversation import Conversation
from conduit.strategies.document_edits.models import DocumentEdits, EditOp, EditType

def _make_conversation_with_parsed(edits_obj: DocumentEdits) -> Conversation:
    msg = AssistantMessage(parsed=edits_obj)
    conv = Conversation()
    # conversation.last returns the last message; mock or build it
    conv.messages.append(msg)
    return conv
```

Patch target: `conduit.strategies.document_edits.strategy.ConduitAsync`

---

## Chunk 1: Data Layer (models, apply, prompt)

### Task 1: Pydantic models

**Files:**
- Create: `src/conduit/strategies/document_edits/__init__.py`
- Create: `src/conduit/strategies/document_edits/models.py`
- Create: `tests/strategies/__init__.py`
- Create: `tests/strategies/document_edits/__init__.py`
- Create: `tests/strategies/document_edits/test_models.py`

- [ ] **Step 1: Write the failing test — AC-1**

```python
# tests/strategies/document_edits/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/strategies/document_edits/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'conduit.strategies.document_edits'`

- [ ] **Step 3: Create package init files**

```bash
touch src/conduit/strategies/document_edits/__init__.py
touch tests/strategies/__init__.py
touch tests/strategies/document_edits/__init__.py
```

- [ ] **Step 4: Write minimal implementation — AC-1**

```python
# src/conduit/strategies/document_edits/models.py
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/strategies/document_edits/test_models.py -v
```

Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/conduit/strategies/document_edits/ tests/strategies/
git commit -m "feat: add DocumentEdits pydantic models (AC-1)"
```

---

### Task 2: apply_edits — replace op

**Files:**
- Create: `src/conduit/strategies/document_edits/apply.py`
- Create: `tests/strategies/document_edits/test_apply.py`

- [ ] **Step 1: Write the failing test — AC-2**

```python
# tests/strategies/document_edits/test_apply.py
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
    # Second op targets the text as modified by the first
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_replace_op_substitutes_first_occurrence tests/strategies/document_edits/test_apply.py::test_replace_ops_are_sequential_against_current_state -v
```

Expected: `ModuleNotFoundError: No module named 'conduit.strategies.document_edits.apply'`

- [ ] **Step 3: Write minimal implementation — AC-2**

```python
# src/conduit/strategies/document_edits/apply.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.strategies.document_edits.models import EditOp


class EditApplicationError(Exception):
    pass


def apply_edits(document: str, edits: list[EditOp]) -> str:
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
        from conduit.strategies.document_edits.models import EditType
        if op.type == EditType.replace:
            current = current.replace(op.search, op.replace, 1)
        elif op.type == EditType.insert:
            idx = current.index(op.search)
            insert_at = idx + len(op.search)
            current = current[:insert_at] + op.replace + current[insert_at:]
        elif op.type == EditType.delete:
            current = current.replace(op.search, "", 1)
    return current
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_replace_op_substitutes_first_occurrence tests/strategies/document_edits/test_apply.py::test_replace_ops_are_sequential_against_current_state -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/conduit/strategies/document_edits/apply.py
git commit -m "feat: apply_edits replace op, sequential against current state (AC-2)"
```

---

### Task 3: apply_edits — insert op

- [ ] **Step 1: Write the failing test — AC-3**

```python
# append to tests/strategies/document_edits/test_apply.py

def test_insert_op_appends_after_search_anchor():
    doc = "hello world"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.insert, search="hello", replace=" beautiful")],
        summary="insert",
    )
    result = apply_edits(doc, edits.edits)
    assert result == "hello beautiful world"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_insert_op_appends_after_search_anchor -v
```

Expected: FAIL (the `insert` branch doesn't exist yet)

- [ ] **Step 3: Verify implementation already handles it**

The `insert` branch was included in Task 2's implementation. Run the test — it should pass without additional code.

- [ ] **Step 4: Run test to verify it passes — AC-3**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_insert_op_appends_after_search_anchor -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tests/strategies/document_edits/test_apply.py
git commit -m "test: verify insert op appends after search anchor (AC-3)"
```

---

### Task 4: apply_edits — delete op

- [ ] **Step 1: Write the failing test — AC-4**

```python
# append to tests/strategies/document_edits/test_apply.py

def test_delete_op_removes_search_text():
    doc = "the quick brown fox"
    edits = DocumentEdits(
        edits=[EditOp(type=EditType.delete, search=" brown")],
        summary="delete",
    )
    result = apply_edits(doc, edits.edits)
    assert result == "the quick fox"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_delete_op_removes_search_text -v
```

Expected: FAIL

- [ ] **Step 3: Verify implementation already handles it**

The `delete` branch was included in Task 2's implementation. Run the test.

- [ ] **Step 4: Run test to verify it passes — AC-4**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_delete_op_removes_search_text -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tests/strategies/document_edits/test_apply.py
git commit -m "test: verify delete op removes search text (AC-4)"
```

---

### Task 5: apply_edits — error cases

- [ ] **Step 1: Write the failing tests — AC-5, AC-6**

```python
# append to tests/strategies/document_edits/test_apply.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strategies/document_edits/test_apply.py::test_raises_when_search_not_found tests/strategies/document_edits/test_apply.py::test_raises_when_search_is_ambiguous -v
```

Expected: FAIL (error handling not yet implemented)

- [ ] **Step 3: Verify implementation already handles it**

Both error cases were included in Task 2's implementation. Run the tests.

- [ ] **Step 4: Run tests to verify they pass — AC-5, AC-6**

```bash
pytest tests/strategies/document_edits/test_apply.py -v
```

Expected: all apply tests pass

- [ ] **Step 5: Commit**

```bash
git add tests/strategies/document_edits/test_apply.py
git commit -m "test: verify EditApplicationError on not-found and ambiguous search (AC-5, AC-6)"
```

---

### Task 6: Prompt template

**Files:**
- Create: `src/conduit/strategies/document_edits/prompt.py`

- [ ] **Step 1: Write the failing test — AC-12**

```python
# tests/strategies/document_edits/test_models.py  (append)
from conduit.core.prompt.prompt import Prompt


def test_prompt_template_renders_with_required_variables():
    from conduit.strategies.document_edits.prompt import PROMPT_TEMPLATE
    rendered = Prompt(PROMPT_TEMPLATE).render(
        {"user_prompt": "Fix the typo", "document": "Helo world"}
    )
    assert "Fix the typo" in rendered
    assert "Helo world" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_models.py::test_prompt_template_renders_with_required_variables -v
```

Expected: `ModuleNotFoundError: No module named 'conduit.strategies.document_edits.prompt'`

- [ ] **Step 3: Write minimal implementation — AC-12**

```python
# src/conduit/strategies/document_edits/prompt.py
PROMPT_TEMPLATE = """\
<user_prompt>{{ user_prompt }}</user_prompt>

<document>{{ document }}</document>

Return ONLY valid JSON conforming to the DocumentEdits schema.
Apply edits sequentially; each search string must match the document \
as it exists after all prior edits have been applied.\
"""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/strategies/document_edits/test_models.py -v
```

Expected: all model + prompt tests pass

- [ ] **Step 5: Commit**

```bash
git add src/conduit/strategies/document_edits/prompt.py tests/strategies/document_edits/test_models.py
git commit -m "feat: add PROMPT_TEMPLATE with user_prompt and document variables (AC-12)"
```

---

## Chunk 2: Strategy

### Task 7: Strategy class structure

**Files:**
- Create: `src/conduit/strategies/document_edits/strategy.py`
- Create: `tests/strategies/document_edits/test_strategy.py`

- [ ] **Step 1: Write the failing test — AC-11**

```python
# tests/strategies/document_edits/test_strategy.py
from __future__ import annotations
from conduit.core.workflow.step import StepWrapper
from conduit.strategies.document_edits.strategy import DocumentEditStrategy


def test_strategy_call_is_wrapped_with_step():
    # Strategy.__init_subclass__ enforces @step at class definition time.
    # If this import succeeds without TypeError, AC-11 is satisfied.
    # This test also verifies the wrapper is in place at runtime.
    strategy = DocumentEditStrategy()
    assert isinstance(DocumentEditStrategy.__call__, StepWrapper)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_call_is_wrapped_with_step -v
```

Expected: `ModuleNotFoundError: No module named 'conduit.strategies.document_edits.strategy'`

- [ ] **Step 3: Write minimal strategy skeleton — AC-11**

```python
# src/conduit/strategies/document_edits/strategy.py
from __future__ import annotations
from conduit.core.workflow.protocols import Strategy
from conduit.core.workflow.step import step, get_param
import logging

logger = logging.getLogger(__name__)


class DocumentEditStrategy(Strategy):
    @step
    async def __call__(self, document: str, user_prompt: str, **kwargs) -> str:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_call_is_wrapped_with_step -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/conduit/strategies/document_edits/strategy.py tests/strategies/document_edits/test_strategy.py
git commit -m "feat: DocumentEditStrategy skeleton with @step (AC-11)"
```

---

### Task 8: Strategy builds correct GenerationParams and ConduitOptions

- [ ] **Step 1: Write the failing tests — AC-7, AC-8**

```python
# append to tests/strategies/document_edits/test_strategy.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from conduit.core.workflow.harness import ConduitHarness
from conduit.domain.message.message import AssistantMessage
from conduit.domain.conversation.conversation import Conversation
from conduit.strategies.document_edits.models import DocumentEdits, EditOp, EditType
from conduit.strategies.document_edits.strategy import DocumentEditStrategy


def _make_fake_conversation(parsed_obj: DocumentEdits) -> Conversation:
    """Build a Conversation whose last message has .parsed set."""
    msg = AssistantMessage(parsed=parsed_obj)
    conv = MagicMock(spec=Conversation)
    conv.last = msg
    return conv


@pytest.mark.asyncio
async def test_strategy_builds_params_with_structured_response_output_type(mock_model_store_validation):
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    fake_edits = DocumentEdits(edits=[], summary="no changes")
    fake_conv = _make_fake_conversation(fake_edits)

    with patch(
        "conduit.strategies.document_edits.strategy.ConduitAsync"
    ) as MockConduitAsync:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = fake_conv
        MockConduitAsync.return_value = mock_instance

        await harness.run(strategy, document="hello world", user_prompt="do nothing")

    _, run_args, _ = mock_instance.run.mock_calls[0]
    params = run_args[1]  # second positional arg to .run()
    assert params.output_type == "structured_response"
    assert params.response_model is DocumentEdits


@pytest.mark.asyncio
async def test_strategy_builds_options_with_include_history_false(mock_model_store_validation):
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    fake_edits = DocumentEdits(edits=[], summary="no changes")
    fake_conv = _make_fake_conversation(fake_edits)

    with patch(
        "conduit.strategies.document_edits.strategy.ConduitAsync"
    ) as MockConduitAsync:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = fake_conv
        MockConduitAsync.return_value = mock_instance

        await harness.run(strategy, document="hello world", user_prompt="do nothing")

    _, run_args, _ = mock_instance.run.mock_calls[0]
    options = run_args[2]  # third positional arg to .run()
    assert options.include_history is False
```

Note: `pytest-asyncio` is required. Check if it is already a dev dependency:

```bash
cd /Users/bianders/Brian_Code/conduit-project
grep asyncio pyproject.toml
```

If not present, add it to `[project.optional-dependencies]` dev group:

```bash
uv add --dev pytest-asyncio
```

Also add `asyncio_mode = "auto"` to `pyproject.toml` under `[tool.pytest.ini_options]` if not already set, to avoid needing `@pytest.mark.asyncio` decorators.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_builds_params_with_structured_response_output_type tests/strategies/document_edits/test_strategy.py::test_strategy_builds_options_with_include_history_false -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement params and options construction — AC-7, AC-8**

```python
# src/conduit/strategies/document_edits/strategy.py
from __future__ import annotations
from conduit.core.workflow.protocols import Strategy
from conduit.core.workflow.step import step, get_param
import logging

logger = logging.getLogger(__name__)


class DocumentEditStrategy(Strategy):
    @step
    async def __call__(self, document: str, user_prompt: str, **kwargs) -> str:
        from conduit.core.conduit.conduit_async import ConduitAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.strategies.document_edits.models import DocumentEdits
        from conduit.strategies.document_edits.prompt import PROMPT_TEMPLATE
        from conduit.strategies.document_edits.apply import apply_edits

        model = get_param("model", default="gpt3")
        temperature = get_param("temperature", default=None)
        max_tokens = get_param("max_tokens", default=None)
        project_name = get_param("project_name", default="conduit")

        params = GenerationParams(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            output_type="structured_response",
            response_model=DocumentEdits,
        )
        options = ConduitOptions(
            project_name=project_name,
            include_history=False,
        )

        prompt = Prompt(PROMPT_TEMPLATE)
        conduit = ConduitAsync(prompt=prompt)
        conversation = await conduit.run(
            input_variables={"user_prompt": user_prompt, "document": document},
            params=params,
            options=options,
        )

        last = conversation.last
        if not isinstance(getattr(last, "parsed", None), DocumentEdits):
            raise TypeError(
                f"Expected DocumentEdits from LLM, got {type(getattr(last, 'parsed', None))}"
            )

        return apply_edits(document, last.parsed.edits)
```

- [ ] **Step 4: Run tests to verify they pass — AC-7, AC-8**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_builds_params_with_structured_response_output_type tests/strategies/document_edits/test_strategy.py::test_strategy_builds_options_with_include_history_false -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/conduit/strategies/document_edits/strategy.py
git commit -m "feat: build GenerationParams with structured_response, ConduitOptions with include_history=False (AC-7, AC-8)"
```

---

### Task 9: Strategy returns edited document

- [ ] **Step 1: Write the failing test — AC-9**

```python
# append to tests/strategies/document_edits/test_strategy.py

@pytest.mark.asyncio
async def test_strategy_returns_document_with_edits_applied(mock_model_store_validation):
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    fake_edits = DocumentEdits(
        edits=[EditOp(type=EditType.replace, search="hello", replace="goodbye")],
        summary="replaced greeting",
    )
    fake_conv = _make_fake_conversation(fake_edits)

    with patch(
        "conduit.strategies.document_edits.strategy.ConduitAsync"
    ) as MockConduitAsync:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = fake_conv
        MockConduitAsync.return_value = mock_instance

        result = await harness.run(
            strategy, document="hello world", user_prompt="replace greeting"
        )

    assert result == "goodbye world"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_returns_document_with_edits_applied -v
```

Expected: FAIL (implementation not complete yet)

- [ ] **Step 3: Verify implementation already handles it**

The full `__call__` body from Task 8 already calls `apply_edits` and returns the result. Run the test.

- [ ] **Step 4: Run test to verify it passes — AC-9**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_returns_document_with_edits_applied -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add tests/strategies/document_edits/test_strategy.py
git commit -m "test: verify strategy returns document with edits applied (AC-9)"
```

---

### Task 10: Strategy raises TypeError on bad parsed output

- [ ] **Step 1: Write the failing test — AC-10**

```python
# append to tests/strategies/document_edits/test_strategy.py

@pytest.mark.asyncio
async def test_strategy_raises_type_error_when_parsed_is_not_document_edits(mock_model_store_validation):
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    # Simulate LLM returning None parsed (e.g. malformed JSON)
    bad_msg = AssistantMessage(content="not json")
    bad_conv = MagicMock(spec=Conversation)
    bad_conv.last = bad_msg

    with patch(
        "conduit.strategies.document_edits.strategy.ConduitAsync"
    ) as MockConduitAsync:
        mock_instance = AsyncMock()
        mock_instance.run.return_value = bad_conv
        MockConduitAsync.return_value = mock_instance

        with pytest.raises(TypeError, match="Expected DocumentEdits"):
            await harness.run(
                strategy, document="hello world", user_prompt="do something"
            )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_raises_type_error_when_parsed_is_not_document_edits -v
```

Expected: FAIL (TypeError not yet raised in the skeleton)

- [ ] **Step 3: Verify implementation already handles it**

The `isinstance` check in Task 8's implementation raises `TypeError` when `parsed` is not `DocumentEdits`. Run the test.

- [ ] **Step 4: Run test to verify it passes — AC-10**

```bash
pytest tests/strategies/document_edits/test_strategy.py::test_strategy_raises_type_error_when_parsed_is_not_document_edits -v
```

Expected: 1 passed

- [ ] **Step 5: Run full test suite for the strategy module**

```bash
pytest tests/strategies/document_edits/ -v
```

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/strategies/document_edits/test_strategy.py
git commit -m "test: verify TypeError raised when parsed is not DocumentEdits (AC-10)"
```

---

## Final verification

- [ ] **Run the complete document_edits test suite**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/strategies/document_edits/ -v
```

Expected output (all passing):
```
tests/strategies/document_edits/test_models.py::test_edit_type_values PASSED
tests/strategies/document_edits/test_models.py::test_edit_op_fields PASSED
tests/strategies/document_edits/test_models.py::test_edit_op_replace_defaults_to_empty_string PASSED
tests/strategies/document_edits/test_models.py::test_document_edits_max_length_edits PASSED
tests/strategies/document_edits/test_models.py::test_document_edits_max_length_summary PASSED
tests/strategies/document_edits/test_models.py::test_document_edits_valid PASSED
tests/strategies/document_edits/test_models.py::test_prompt_template_renders_with_required_variables PASSED
tests/strategies/document_edits/test_apply.py::test_replace_op_substitutes_first_occurrence PASSED
tests/strategies/document_edits/test_apply.py::test_replace_ops_are_sequential_against_current_state PASSED
tests/strategies/document_edits/test_apply.py::test_insert_op_appends_after_search_anchor PASSED
tests/strategies/document_edits/test_apply.py::test_delete_op_removes_search_text PASSED
tests/strategies/document_edits/test_apply.py::test_raises_when_search_not_found PASSED
tests/strategies/document_edits/test_apply.py::test_raises_when_search_is_ambiguous PASSED
tests/strategies/document_edits/test_strategy.py::test_strategy_call_is_wrapped_with_step PASSED
tests/strategies/document_edits/test_strategy.py::test_strategy_builds_params_with_structured_response_output_type PASSED
tests/strategies/document_edits/test_strategy.py::test_strategy_builds_options_with_include_history_false PASSED
tests/strategies/document_edits/test_strategy.py::test_strategy_returns_document_with_edits_applied PASSED
tests/strategies/document_edits/test_strategy.py::test_strategy_raises_type_error_when_parsed_is_not_document_edits PASSED
```

- [ ] **Run existing tests to verify no regressions**

```bash
pytest tests/unit/ tests/cli/ -v
```

Expected: all previously passing tests still pass
