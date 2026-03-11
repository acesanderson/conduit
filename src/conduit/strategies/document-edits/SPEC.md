# DocumentEdit Strategy

## Purpose

A stateless strategy that applies LLM-generated edits to a document without
returning the full text. The LLM returns a structured list of surgical ops
(`replace`, `insert`, `delete`), which are applied sequentially to the document.

This avoids the token waste of asking the LLM to rewrite the entire document
when only a few targeted changes are needed.

---

## Interface

```python
class DocumentEditStrategy(Strategy):
    @step
    async def __call__(self, document: str, user_prompt: str, **kwargs) -> str:
        ...
```

All runtime configuration comes via `**kwargs` / `get_param()`. No `__init__`,
no stored state — consistent with `OneShotSummarizer`, `MapReduceSummarizer`, etc.

### Inputs

| Name | Type | Source |
|---|---|---|
| `document` | `str` | positional arg |
| `user_prompt` | `str` | positional arg |
| `model` | `str` | `get_param("model", default="gpt3")` |
| `temperature` | `float \| None` | `get_param(...)` |
| `max_tokens` | `int \| None` | `get_param(...)` |
| `project_name` | `str` | `get_param(..., default="conduit")` |

### Output

`str` — the document with all edits applied.

---

## Pydantic Models

```python
class EditType(str, Enum):
    replace = "replace"
    insert = "insert"
    delete = "delete"

class EditOp(BaseModel):
    type: EditType
    search: str      # unique substring anchoring the edit location
    replace: str     # new text; empty string for delete

class DocumentEdits(BaseModel):
    edits: list[EditOp] = Field(..., max_length=20)
    summary: str     = Field(..., max_length=200)
```

`response_model = DocumentEdits` is injected into `GenerationParams` inside
`__call__`. The caller's params object is not mutated — a new one is constructed.

---

## Prompt Template

```jinja2
<user_prompt>{{ user_prompt }}</user_prompt>

<document>{{ document }}</document>

Return ONLY valid JSON conforming to the DocumentEdits schema.
Apply edits sequentially; each search string must match the document
as it exists after all prior edits have been applied.
```

Stored as a module-level constant string. No file loading required at this scale.

---

## Execution Flow

```
__call__(document, user_prompt, **kwargs)
  │
  ├── build GenerationParams(model, temperature, max_tokens,
  │       output_type="structured_response", response_model=DocumentEdits)
  ├── build ConduitOptions(project_name, include_history=False)
  ├── render Prompt with {user_prompt, document}
  ├── ConduitAsync(prompt).run(input_variables, params, options)
  │
  ├── extract conversation.last_assistant_message.parsed
  ├── assert isinstance(parsed, DocumentEdits)
  │
  └── apply_edits(document, parsed.edits) → str
```

---

## Apply Function

```
apply_edits(document: str, edits: list[EditOp]) -> str
```

- Iterates `edits` in order
- For each op, searches the **current document state** (not the original)
- `replace` / `delete`: finds first occurrence of `search`, substitutes `replace`
- `insert`: finds first occurrence of `search`, appends `replace` immediately after
- Raises `EditApplicationError` if `search` is not found
- Raises `EditApplicationError` if `search` is ambiguous (found > 1 time)

Both failure modes are explicit — silent wrong edits are worse than a crash.

---

## Key Design Decisions

**Sequential against current state, not original.**
Each `search` targets the document as modified by prior ops. The prompt instructs
the LLM accordingly. This allows ops to chain (e.g., insert a line, then edit it).

**No history.**
`ConduitOptions` always sets `include_history=False`. This is a one-shot
transformation; loading prior session context would be incorrect.

**Caller params are never mutated.**
`GenerationParams` is constructed fresh inside `__call__` using `get_param()`
values. The structured output fields (`output_type`, `response_model`) are set
at construction time, not via `model_copy`, avoiding the Pydantic v2 validator
re-run issue.

**Fail loudly on ambiguous search.**
A search string appearing twice means the LLM produced a non-unique anchor.
Silently applying to the first occurrence would produce unpredictably wrong
output. Better to surface this as an error the caller can handle.

---

## File Layout

```
strategies/document-edits/
  SPEC.md          ← this file
  strategy.py      ← DocumentEditStrategy + @step __call__
  models.py        ← EditType, EditOp, DocumentEdits
  apply.py         ← apply_edits(), EditApplicationError
  prompt.py        ← PROMPT_TEMPLATE constant
```
