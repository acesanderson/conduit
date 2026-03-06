# Query --search Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `-S/--search` flag to `conduit query` that gives the model `web_search` and `fetch_url` as tools and runs the existing Engine's multi-turn FSM (GENERATE → EXECUTE → TERMINATE) before producing a final response.

**Architecture:** The flag threads through the Click command layer → handler → `CLIQueryFunctionInputs`. When `search=True`, a private `_search_query_function` builds a `ToolRegistry` with `web_search` and `fetch_url`, constructs `ConduitSync` directly via `__init__` (not the `.create()` factory), and delegates to `conduit.run()`. The Protocol return type is corrected from the lying `GenerationResponse` to `Conversation`, which is what all query paths already return. No changes to Engine, ConduitOptions, or the tools themselves.

**Tech Stack:** Click, `ConduitSync`, `ConduitOptions`, `ToolRegistry`, `GenerationParams`, `Prompt`, `fetch_url` + `web_search` from `capabilities/tools/tools/fetch/fetch.py`.

---

### Task 1: Fix Protocol return type + add `search` field to `CLIQueryFunctionInputs`

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`

No tests exist yet in this project. Verify the change doesn't break the import chain instead.

**Step 1: Fix the Protocol return type**

In `query_function.py`, the `CLIQueryFunctionProtocol.__call__` currently lies — it declares `-> GenerationResponse` but all implementations return `Conversation`. Fix it.

Change the import block (under `TYPE_CHECKING`) from:
```python
from conduit.domain.result.response import GenerationResponse
```
to:
```python
from conduit.domain.conversation.conversation import Conversation
```

And update the Protocol:
```python
@runtime_checkable
class CLIQueryFunctionProtocol(Protocol):
    def __call__(
        self,
        inputs: CLIQueryFunctionInputs,
    ) -> Conversation: ...
```

Also update `default_query_function`'s return annotation:
```python
def default_query_function(
    inputs: CLIQueryFunctionInputs,
) -> Conversation:
```

**Step 2: Add `search` field to `CLIQueryFunctionInputs`**

Add after `ephemeral`:
```python
search: bool = False  # Enable web search + URL fetch tools
```

**Step 3: Verify import chain is intact**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python -c "from conduit.apps.cli.query.query_function import CLIQueryFunctionInputs, CLIQueryFunctionProtocol, default_query_function; print('OK')"
```

Expected output: `OK`

**Step 4: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py
git commit -m "fix: correct CLIQueryFunctionProtocol return type to Conversation, add search field"
```

---

### Task 2: Implement `_search_query_function`

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`

This is the core new behavior. It mirrors `default_query_function` but injects a `ToolRegistry` with `web_search` + `fetch_url`.

**Step 1: Add the private helper at the bottom of `query_function.py`, before `default_query_function`**

```python
def _search_query_function(inputs: CLIQueryFunctionInputs) -> Conversation:
    """
    Query function variant that registers web_search and fetch_url as tools,
    enabling the Engine's multi-turn GENERATE → EXECUTE → TERMINATE loop.
    """
    from conduit.capabilities.tools.registry import ToolRegistry
    from conduit.capabilities.tools.tools.fetch.fetch import fetch_url, web_search
    from conduit.core.conduit.conduit_sync import ConduitSync
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.core.prompt.prompt import Prompt
    from conduit.config import settings

    # Build tool registry with web search + fetch capabilities
    tool_registry = ToolRegistry()
    tool_registry.register_function(web_search)
    tool_registry.register_function(fetch_url)

    # Mirror default_query_function's prompt assembly (POSIX philosophy)
    combined_query = "\n\n".join(
        [inputs.query_input, inputs.context, inputs.append]
    ).strip()
    prompt = Prompt(combined_query)

    # Build params — mirror default_query_function
    params = GenerationParams(
        model=inputs.preferred_model,
        system=inputs.system_message or None,
        temperature=inputs.temperature,
    )

    # Build options from global defaults, then override
    options = settings.default_conduit_options()
    opt_updates: dict = {
        "verbosity": inputs.verbose,
        "tool_registry": tool_registry,
        "include_history": inputs.include_history,
    }

    # Cache wiring — mirror default_query_function (cache=not local)
    if inputs.cache:
        cache_name = inputs.project_name or settings.default_project_name
        opt_updates["cache"] = settings.default_cache(project_name=cache_name)

    # Persistence wiring — mirror default_query_function (persist=not ephemeral)
    if not inputs.ephemeral:
        repo_name = inputs.project_name or settings.default_project_name
        opt_updates["repository"] = settings.default_repository(
            project_name=repo_name
        )

    options = options.model_copy(update=opt_updates)

    # Construct directly (not via .create() factory, which doesn't expose tool_registry)
    conduit = ConduitSync(prompt=prompt, params=params, options=options)
    return conduit.run()
```

**Step 2: Verify import**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python -c "from conduit.apps.cli.query.query_function import _search_query_function; print('OK')"
```

Expected output: `OK`

**Step 3: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py
git commit -m "feat: add _search_query_function with web_search + fetch_url tool registry"
```

---

### Task 3: Branch `default_query_function` on `inputs.search`

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`

**Step 1: Add branch at the top of `default_query_function`**

Immediately after the docstring and before any other logic, add:

```python
if inputs.search:
    return _search_query_function(inputs)
```

The function body after that remains entirely unchanged.

**Step 2: Verify**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python -c "
from conduit.apps.cli.query.query_function import CLIQueryFunctionInputs, default_query_function
print('OK')
"
```

Expected output: `OK`

**Step 3: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py
git commit -m "feat: branch default_query_function to _search_query_function when search=True"
```

---

### Task 4: Thread `search` through `BaseHandlers.handle_query`

**Files:**
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py`

**Step 1: Add `search` parameter to `handle_query` signature**

Current signature ends with:
```python
project_name: str = "",
```

Add after it:
```python
search: bool = False,
```

**Step 2: Thread it into `CLIQueryFunctionInputs`**

In the `inputs = CLIQueryFunctionInputs(...)` construction block, add:
```python
search=search,
```

**Step 3: Verify**

```bash
cd /Users/bianders/Brian_Code/conduit-project
python -c "from conduit.apps.cli.handlers.base_handlers import BaseHandlers; print('OK')"
```

Expected output: `OK`

**Step 4: Commit**

```bash
git add src/conduit/apps/cli/handlers/base_handlers.py
git commit -m "feat: thread search flag through BaseHandlers.handle_query"
```

---

### Task 5: Add `-S/--search` flag to the `query` Click command

**Files:**
- Modify: `src/conduit/apps/cli/commands/base_commands.py`

**Step 1: Add the option decorator**

In the `query` command definition (inside `_register_commands`), add after the `--append` option and before `@click.argument`:

```python
@click.option(
    "-S", "--search", is_flag=True,
    help="Use web search and URL fetch to inform the answer (multi-turn agent).",
)
```

**Step 2: Add `search` to the function signature**

```python
def query(
    ctx: click.Context,
    model: str | None,
    local: bool,
    raw: bool,
    temperature: float | None,
    chat: bool,
    append: str | None,
    search: bool,          # <-- add this
    query_input: tuple[str, ...],
):
```

**Step 3: Pass `search` to `handlers.handle_query`**

In the `handlers.handle_query(...)` call, add:
```python
search=search,
```

**Step 4: Verify the CLI wires up correctly**

```bash
cd /Users/bianders/Brian_Code/conduit-project
conduit query --help
```

Expected output includes: `-S, --search  Use web search and URL fetch to inform the answer (multi-turn agent).`

**Step 5: Smoke test (requires BRAVE_API_KEY in env)**

```bash
conduit query -S "What is the latest Python release?"
```

Expected: model searches, fetches a page, then returns a grounded answer. No crash.

**Step 6: Commit**

```bash
git add src/conduit/apps/cli/commands/base_commands.py
git commit -m "feat: add -S/--search flag to conduit query"
```

---

## Notes

- No changes needed to `Engine`, `ConduitOptions`, `AnthropicClient`, or the fetch tools.
- `ConduitSync.create()` is intentionally bypassed — it has no `tool_registry` param and is positioned as a scripting convenience, not the right entry point here.
- Tool-call turns (AssistantMessage with tool_calls + ToolResultMessages) will appear in conversation history when `-c` is also active. This is expected and correct; a future "ephemeral by default" change will address history pollution.
- Progress during tool execution depends on `verbosity` and the Engine's console output. No changes needed now; tracked as future work.
