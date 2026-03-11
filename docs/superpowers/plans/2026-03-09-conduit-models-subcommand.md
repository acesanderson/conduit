# `conduit models` Subcommand Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the standalone `models_cli.py` script with a `conduit models` Click subcommand, making `models` a `sys.argv` shim (like `ask`) that routes through `conduit_cli.main()`.

**Architecture:** A new `models_commands.py` defines a single `@click.command("models")` with all 6 flags and all handler logic. All `ModelStore` calls are lazy (inside the callback, not at module level). The command is attached to `ConduitCLI` in `conduit_cli.main()`. The `models` binary becomes a thin entrypoint shim identical in structure to `query_entrypoint`.

**Tech Stack:** Click (existing), Rich console (existing), `ModelStore` (existing), `HeadwaterClient` (existing), `pytest` + `click.testing.CliRunner` for tests.

---

## Acceptance Criteria Reference

| ID | Command | Expected outcome |
|----|---------|-----------------|
| AC1 | `conduit models` | Calls `ModelStore.display()` — same output as pre-migration `models` |
| AC2 | `conduit models -e` | Prints embedding model list |
| AC3 | `conduit models -r` | Prints reranker model list |
| AC4 | `conduit models -m claude-3-5-sonnet` | Model card or fuzzy suggestions |
| AC5 | `conduit models -t chat` | Filtered list via `ModelStore.by_type` |
| AC6 | `conduit models --help` | All 6 flags documented (`-m`, `-t`, `-p`, `-a`, `-e`, `-r`) |
| AC7 | `models -e` | Identical to `conduit models -e` via entrypoint shim |
| AC8 | `models models -e` | Double-injection guard fires; same as `models -e` |
| AC9 | `conduit query "hello"` | Unaffected — no `ModelStore` method calls at import time |

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `src/conduit/apps/cli/commands/models_commands.py` | Single `@click.command("models")` with all 6 flags and handler logic. All imports lazy (inside callback). |
| **Modify** | `src/conduit/apps/scripts/conduit_cli.py` | Add `models_entrypoint()` shim. Attach `models_command` in `main()`. |
| **Delete** | `src/conduit/apps/scripts/models_cli.py` | Replaced entirely by `models_commands.py`. |
| **Modify** | `pyproject.toml` | Change `models` entry point from `models_cli:main` to `conduit_cli:models_entrypoint`. |
| **Create** | `tests/cli/test_models_commands.py` | All unit tests for the Click command (CliRunner-based). |

---

## Chunk 1: `models_commands.py` — The Click Command

### Task 1: Scaffold `models_commands.py` with lazy-import no-op

**Files:**
- Create: `src/conduit/apps/cli/commands/models_commands.py`
- Create: `tests/cli/test_models_commands.py`

**AC: AC9** — `conduit query "hello"` unaffected; no `ModelStore` method calls at module import time.

- [ ] **Step 1: Write the failing test**

```python
# tests/cli/test_models_commands.py
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli():
    """Minimal Click group wrapping models_command — avoids full ConduitCLI setup."""
    from conduit.apps.cli.commands.models_commands import models_command

    @click.group()
    def _cli():
        pass

    _cli.add_command(models_command)
    return _cli


def test_no_module_level_modelstore_calls():
    """AC9: importing models_commands must not call any ModelStore methods."""
    sys.modules.pop("conduit.apps.cli.commands.models_commands", None)

    with patch("conduit.core.model.models.modelstore.ModelStore") as MockStore:
        importlib.import_module("conduit.apps.cli.commands.models_commands")
        MockStore.list_models.assert_not_called()
        MockStore.list_model_types.assert_not_called()
        MockStore.list_providers.assert_not_called()
        MockStore.display.assert_not_called()
```

- [ ] **Step 2: Run the test to verify it fails**

```
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_models_commands.py::test_no_module_level_modelstore_calls -v
```

Expected: `ERROR` or `FAILED` — module does not exist yet.

- [ ] **Step 3: Create `models_commands.py` with a no-op scaffold**

```python
# src/conduit/apps/cli/commands/models_commands.py
from __future__ import annotations

import click


@click.command("models")
@click.option("-m", "--model", type=str, help="Name of the model to retrieve details for.")
@click.option("-t", "--type", "model_type", type=str, help="Type of the model to filter by.")
@click.option("-p", "--provider", type=str, help="Provider of the model to filter by.")
@click.option("-a", "--aliases", is_flag=True, help="Display model aliases.")
@click.option("-e", "--embeddings", is_flag=True, help="Display models that support embeddings.")
@click.option("-r", "--rerankers", is_flag=True, help="Display models that support reranking.")
def models_command(
    model: str | None,
    model_type: str | None,
    provider: str | None,
    aliases: bool,
    embeddings: bool,
    rerankers: bool,
) -> None:
    """List and inspect available models."""
    pass
```

- [ ] **Step 4: Run the test to verify it passes**

```
pytest tests/cli/test_models_commands.py::test_no_module_level_modelstore_calls -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: scaffold models_commands.py with lazy-import no-op (AC9)"
```

---

### Task 2: `conduit models` with no flags calls `ModelStore.display()`

**Files:**
- Modify: `src/conduit/apps/cli/commands/models_commands.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC1** — `conduit models` (no flags) calls `ModelStore.display()`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/cli/test_models_commands.py

def test_no_flags_calls_modelstore_display(runner, cli):
    """AC1: conduit models with no flags calls ModelStore.display()."""
    with patch("conduit.core.model.models.modelstore.ModelStore.display") as mock_display:
        result = runner.invoke(cli, ["models"])
        mock_display.assert_called_once()
        assert result.exit_code == 0
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/cli/test_models_commands.py::test_no_flags_calls_modelstore_display -v
```

Expected: `FAILED` — `ModelStore.display` not called (callback is a no-op).

- [ ] **Step 3: Implement the no-flags branch**

Replace the `pass` in `models_command` with:

```python
def models_command(model, model_type, provider, aliases, embeddings, rerankers):
    """List and inspect available models."""
    if rerankers:
        pass  # placeholder — implemented in Task 7
        return

    if embeddings:
        pass  # placeholder — implemented in Task 6
        return

    from conduit.core.model.models.modelstore import ModelStore

    if aliases:
        pass  # placeholder — implemented in Task 5
        return

    if model:
        pass  # placeholder — implemented in Task 3
        return

    if model_type:
        pass  # placeholder — implemented in Task 4
        return

    if provider:
        pass  # placeholder — implemented in Task 5
        return

    ModelStore.display()
```

- [ ] **Step 4: Run the test to verify it passes**

```
pytest tests/cli/test_models_commands.py::test_no_flags_calls_modelstore_display -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: models_command no-flags branch calls ModelStore.display (AC1)"
```

---

### Task 3: `-m/--model` flag — model card or fuzzy suggestions

**Files:**
- Modify: `src/conduit/apps/cli/commands/models_commands.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC4** — `conduit models -m claude-3-5-sonnet` shows model card or fuzzy suggestions.

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/cli/test_models_commands.py

def test_model_flag_calls_get_model(runner, cli):
    """AC4: -m with a known model calls ModelStore.get_model() and accesses .card."""
    from unittest.mock import PropertyMock

    mock_spec = MagicMock()
    type(mock_spec).card = PropertyMock(return_value=None)

    with patch("conduit.core.model.models.modelstore.ModelStore.get_model", return_value=mock_spec):
        result = runner.invoke(cli, ["models", "-m", "claude-3-5-sonnet"])
        assert result.exit_code == 0
        type(mock_spec).card.assert_called_once()


def test_model_flag_fuzzy_on_unknown_model(runner, cli):
    """AC4: -m with an unknown model prints fuzzy suggestions."""
    with patch("conduit.core.model.models.modelstore.ModelStore.get_model", side_effect=ValueError("not found")):
        with patch("conduit.core.model.models.modelstore.ModelStore.list_models", return_value=["claude-3-5-sonnet", "gpt-4o"]):
            with patch("rapidfuzz.process.extract", return_value=[("claude-3-5-sonnet", 90, 0)]):
                result = runner.invoke(cli, ["models", "-m", "claud"])
                assert result.exit_code == 0
                assert "Did you mean" in result.output
```

Add `from unittest.mock import MagicMock` to the imports at the top of the test file.

- [ ] **Step 2: Run the tests to verify they fail**

```
pytest tests/cli/test_models_commands.py::test_model_flag_calls_get_model tests/cli/test_models_commands.py::test_model_flag_fuzzy_on_unknown_model -v
```

Expected: both `FAILED` — `-m` branch is a `pass`.

- [ ] **Step 3: Implement the `-m` branch**

Replace the `if model: pass` placeholder:

```python
    if model:
        from conduit.core.model.models.modelstore import ModelStore
        from rich.console import Console

        try:
            modelspec = ModelStore.get_model(model)
            modelspec.card
        except ValueError:
            from rapidfuzz import process, fuzz
            from collections import namedtuple

            Match = namedtuple("Match", ["title", "score", "rank"])
            models_list = ModelStore.list_models()
            results = process.extract(model, models_list, scorer=fuzz.WRatio, limit=3)
            matches = [
                Match(title=title, score=score, rank=rank + 1)
                for rank, (title, score, _) in enumerate(results)
            ]
            console = Console()
            console.print(f"[red]Model '{model}' not found. Did you mean:[/red]")
            for match in matches:
                console.print(f"  {match.rank}. {match.title}")
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

```
pytest tests/cli/test_models_commands.py::test_model_flag_calls_get_model tests/cli/test_models_commands.py::test_model_flag_fuzzy_on_unknown_model -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: models_command -m flag with fuzzy fallback (AC4)"
```

---

### Task 4: `-t/--type` flag — filtered list by type

**Files:**
- Modify: `src/conduit/apps/cli/commands/models_commands.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC5** — `conduit models -t chat` prints filtered model list.

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/cli/test_models_commands.py

def test_type_flag_prints_filtered_models(runner, cli):
    """AC5: -t prints models filtered by type."""
    mock_spec = MagicMock()
    mock_spec.model = "claude-3-5-sonnet"

    with patch("conduit.core.model.models.modelstore.ModelStore.list_model_types", return_value=["chat", "embedding"]):
        with patch("conduit.core.model.models.modelstore.ModelStore.by_type", return_value=[mock_spec]) as mock_by_type:
            result = runner.invoke(cli, ["models", "-t", "chat"])
            mock_by_type.assert_called_once_with("chat")
            assert "claude-3-5-sonnet" in result.output
            assert result.exit_code == 0


def test_type_flag_rejects_invalid_type(cli):
    """AC5: -t with an invalid type raises BadParameter."""
    # Use mix_stderr=False so error text lands in result.output, not result.stderr
    runner = CliRunner(mix_stderr=False)
    with patch("conduit.core.model.models.modelstore.ModelStore.list_model_types", return_value=["chat", "embedding"]):
        result = runner.invoke(cli, ["models", "-t", "nonexistent"])
        assert result.exit_code != 0
        assert "Must be one of" in result.output or "Must be one of" in (result.stderr or "")
```

- [ ] **Step 2: Run the tests to verify they fail**

```
pytest tests/cli/test_models_commands.py::test_type_flag_prints_filtered_models tests/cli/test_models_commands.py::test_type_flag_rejects_invalid_type -v
```

Expected: both `FAILED`.

- [ ] **Step 3: Implement the `-t` branch**

Replace the `if model_type: pass` placeholder:

```python
    if model_type:
        from conduit.core.model.models.modelstore import ModelStore

        modeltypes = ModelStore.list_model_types()
        if model_type not in modeltypes:
            raise click.BadParameter(
                f"Must be one of: {' | '.join(modeltypes)}",
                param_hint="'--type'",
            )
        modelspecs = ModelStore.by_type(model_type)
        for ms in modelspecs:
            click.echo(ms.model)
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

```
pytest tests/cli/test_models_commands.py::test_type_flag_prints_filtered_models tests/cli/test_models_commands.py::test_type_flag_rejects_invalid_type -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: models_command -t flag with validation (AC5)"
```

---

### Task 5: `-p/--provider`, `-a/--aliases` flags, and `--help` verification

**Files:**
- Modify: `src/conduit/apps/cli/commands/models_commands.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC6** — `conduit models --help` documents all 6 flags.

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/cli/test_models_commands.py

def test_help_documents_all_six_flags(runner, cli):
    """AC6: --help output contains all 6 flag names."""
    result = runner.invoke(cli, ["models", "--help"])
    assert result.exit_code == 0
    for flag in ["-m", "--model", "-t", "--type", "-p", "--provider",
                 "-a", "--aliases", "-e", "--embeddings", "-r", "--rerankers"]:
        assert flag in result.output, f"Missing flag {flag} in --help output"


def test_provider_flag_prints_filtered_models(runner, cli):
    """AC6 (--provider branch): -p prints models filtered by provider."""
    mock_spec = MagicMock()
    mock_spec.model = "claude-3-5-sonnet"

    with patch("conduit.core.model.models.modelstore.ModelStore.list_providers", return_value=["anthropic", "openai"]):
        with patch("conduit.core.model.models.modelstore.ModelStore.by_provider", return_value=[mock_spec]) as mock_by_prov:
            result = runner.invoke(cli, ["models", "-p", "anthropic"])
            mock_by_prov.assert_called_once_with("anthropic")
            assert "claude-3-5-sonnet" in result.output
            assert result.exit_code == 0


def test_aliases_flag_prints_aliases(runner, cli):
    """AC6 (--aliases branch): -a calls ModelStore.aliases()."""
    with patch("conduit.core.model.models.modelstore.ModelStore.aliases", return_value={"sonnet": "claude-3-5-sonnet"}) as mock_aliases:
        result = runner.invoke(cli, ["models", "-a"])
        mock_aliases.assert_called_once()
        assert result.exit_code == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```
pytest tests/cli/test_models_commands.py::test_help_documents_all_six_flags tests/cli/test_models_commands.py::test_provider_flag_prints_filtered_models tests/cli/test_models_commands.py::test_aliases_flag_prints_aliases -v
```

Expected: `test_help_documents_all_six_flags` **passes** (all 6 flags were declared in the Task 1 scaffold — this is intentional; the test verifies they stay present). `test_provider_flag_prints_filtered_models` and `test_aliases_flag_prints_aliases` fail — the branches are still `pass` placeholders.

- [ ] **Step 3: Implement `-p` and `-a` branches**

Replace `if provider: pass` placeholder:

```python
    if provider:
        from conduit.core.model.models.modelstore import ModelStore

        provider = provider.lower()
        providers_list = ModelStore.list_providers()
        if provider not in providers_list:
            raise click.BadParameter(
                f"Must be one of: {' | '.join(providers_list)}",
                param_hint="'--provider'",
            )
        modelspecs = ModelStore.by_provider(provider)
        for ms in modelspecs:
            click.echo(ms.model)
        return
```

Replace `if aliases: pass` placeholder:

```python
    if aliases:
        from conduit.core.model.models.modelstore import ModelStore
        from rich.console import Console

        console = Console()
        aliases_data = ModelStore.aliases()
        console.print(aliases_data)
        return
```

- [ ] **Step 4: Run the tests to verify they pass**

```
pytest tests/cli/test_models_commands.py::test_help_documents_all_six_flags tests/cli/test_models_commands.py::test_provider_flag_prints_filtered_models tests/cli/test_models_commands.py::test_aliases_flag_prints_aliases -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: models_command -p/-a flags and --help coverage (AC6)"
```

---

### Task 6: `-e/--embeddings` flag

**Files:**
- Modify: `src/conduit/apps/cli/commands/models_commands.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC2** — `conduit models -e` prints embedding model list.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/cli/test_models_commands.py

def test_embeddings_flag_prints_embedding_models(runner, cli):
    """AC2: -e lists embedding models from HeadwaterClient."""
    mock_spec = MagicMock()
    mock_spec.model = "sentence-transformers/all-mpnet-base-v2"

    with patch("conduit.embeddings.generate_embeddings.list_embedding_models", return_value=[mock_spec]):
        result = runner.invoke(cli, ["models", "-e"])
        assert result.exit_code == 0
        assert "sentence-transformers/all-mpnet-base-v2" in result.output
        assert "Embedding models" in result.output
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/cli/test_models_commands.py::test_embeddings_flag_prints_embedding_models -v
```

Expected: `FAILED` — embeddings branch is a `pass`.

- [ ] **Step 3: Implement the `-e` branch**

Replace `if embeddings: pass` placeholder:

```python
    if embeddings:
        from conduit.embeddings.generate_embeddings import list_embedding_models
        from rich.console import Console

        specs = list_embedding_models()
        console = Console()
        console.print("Embedding models:", style="bold green")
        for spec in specs:
            console.print(f"  - {spec.model}", style="cyan")
        return
```

- [ ] **Step 4: Run the test to verify it passes**

```
pytest tests/cli/test_models_commands.py::test_embeddings_flag_prints_embedding_models -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: models_command -e flag prints embedding models (AC2)"
```

---

### Task 7: `-r/--rerankers` flag

**Files:**
- Modify: `src/conduit/apps/cli/commands/models_commands.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC3** — `conduit models -r` prints reranker model list.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/cli/test_models_commands.py

def test_rerankers_flag_prints_reranker_models(runner, cli):
    """AC3: -r lists reranker models from HeadwaterClient."""
    mock_spec = MagicMock()
    mock_spec.name = "BAAI/bge-reranker-v2-m3"

    with patch("headwater_client.client.headwater_client.HeadwaterClient") as MockClient:
        MockClient.return_value.reranker.list_reranker_models.return_value = [mock_spec]
        result = runner.invoke(cli, ["models", "-r"])
        assert result.exit_code == 0
        assert "BAAI/bge-reranker-v2-m3" in result.output
        assert "Reranker models" in result.output
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/cli/test_models_commands.py::test_rerankers_flag_prints_reranker_models -v
```

Expected: `FAILED` — rerankers branch is a `pass`.

- [ ] **Step 3: Implement the `-r` branch**

Replace `if rerankers: pass` placeholder:

```python
    if rerankers:
        from headwater_client.client.headwater_client import HeadwaterClient
        from rich.console import Console

        specs = HeadwaterClient().reranker.list_reranker_models()
        console = Console()
        console.print("Reranker models:", style="bold green")
        for spec in specs:
            console.print(f"  - {spec.name}", style="cyan")
        return
```

- [ ] **Step 4: Run the test to verify it passes**

```
pytest tests/cli/test_models_commands.py::test_rerankers_flag_prints_reranker_models -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full test file to confirm no regressions**

```
pytest tests/cli/test_models_commands.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add src/conduit/apps/cli/commands/models_commands.py tests/cli/test_models_commands.py
git commit -m "feat: models_command -r flag prints reranker models (AC3)"
```

---

## Chunk 2: Wiring, Entrypoint, and Cleanup

### Task 8: Wire `models_command` into `conduit_cli.main()`

**Files:**
- Modify: `src/conduit/apps/scripts/conduit_cli.py`

**AC: AC1** (integration smoke — `conduit models` reachable through the real CLI group)

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/cli/test_models_commands.py

def test_models_command_registered_on_conduit_cli():
    """AC1 integration: models command is reachable via the ConduitCLI group."""
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands
    from conduit.apps.cli.commands.cache_commands import cache
    from conduit.apps.cli.commands.models_commands import models_command

    # Replicate exactly what main() does, minus run() — avoids DB/async setup
    conduit_inst = ConduitCLI()
    commands = BaseCommands()
    conduit_inst.attach(commands)
    conduit_inst.cli.add_command(cache)
    conduit_inst.cli.add_command(models_command)

    result = CliRunner().invoke(conduit_inst.cli, ["models", "--help"])
    assert result.exit_code == 0
    assert "--rerankers" in result.output
```

Note: `ConduitCLI.__init__` creates an asyncio event loop and builds the Click group but does **not** open DB connections (those are lazy via `@cached_property`). The test is safe to run without mocking. If `conduit.config.settings` raises in your environment due to missing config, wrap the instantiation in `with patch("conduit.config.settings", ...)` to provide defaults.

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/cli/test_models_commands.py::test_models_command_registered_on_conduit_cli -v
```

Expected: `FAILED` — `models_command` not yet attached.

- [ ] **Step 3: Attach `models_command` in `conduit_cli.main()`**

In `src/conduit/apps/scripts/conduit_cli.py`, add after the existing imports:

```python
from conduit.apps.cli.commands.models_commands import models_command
```

Note: this is a top-level module import of the Click command object — not a `ModelStore` call. It does not violate the lazy-import constraint (which governs `ModelStore` method calls, not the import of Click decorators). This mirrors how `cache` is imported at the top of `conduit_cli.py`.

And in `main()`, add after the existing `add_command(cache)` line:

```python
    conduit_cli.cli.add_command(models_command)
```

The updated `main()`:

```python
def main():
    conduit_cli = ConduitCLI()
    commands = BaseCommands()
    conduit_cli.attach(commands)
    conduit_cli.cli.add_command(cache)
    conduit_cli.cli.add_command(models_command)
    conduit_cli.run()
```

- [ ] **Step 4: Run the test to verify it passes**

```
pytest tests/cli/test_models_commands.py::test_models_command_registered_on_conduit_cli -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/scripts/conduit_cli.py tests/cli/test_models_commands.py
git commit -m "feat: attach models_command to ConduitCLI (AC1 integration)"
```

---

### Task 9: `models_entrypoint()` shim

**Files:**
- Modify: `src/conduit/apps/scripts/conduit_cli.py`
- Modify: `tests/cli/test_models_commands.py`

**AC: AC7** — `models -e` is identical to `conduit models -e` via the entrypoint shim.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/cli/test_models_commands.py

def test_models_entrypoint_injects_models_subcommand(monkeypatch):
    """AC7: models_entrypoint inserts 'models' into sys.argv and calls main()."""
    import sys
    from unittest.mock import patch, call

    monkeypatch.setattr(sys, "argv", ["models", "-e"])

    with patch("conduit.apps.scripts.conduit_cli.main") as mock_main:
        from conduit.apps.scripts.conduit_cli import models_entrypoint
        models_entrypoint()

        assert sys.argv == ["models", "models", "-e"]
        mock_main.assert_called_once()
```

- [ ] **Step 2: Run the test to verify it fails**

```
pytest tests/cli/test_models_commands.py::test_models_entrypoint_injects_models_subcommand -v
```

Expected: `FAILED` — `models_entrypoint` does not exist yet.

- [ ] **Step 3: Add `models_entrypoint()` to `conduit_cli.py`**

Add after `query_entrypoint()` in `src/conduit/apps/scripts/conduit_cli.py`:

```python
def models_entrypoint():
    """
    Shortcut entry point for 'models'.
    Takes us directly to "conduit models ..."
    """
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        pass
    else:
        sys.argv.insert(1, "models")

    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```
pytest tests/cli/test_models_commands.py::test_models_entrypoint_injects_models_subcommand -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/scripts/conduit_cli.py tests/cli/test_models_commands.py
git commit -m "feat: add models_entrypoint() shim (AC7)"
```

---

### Task 10: Double-injection guard

**Files:**
- Modify: `tests/cli/test_models_commands.py`

**AC: AC8** — `models models -e` does not double-inject `"models"`.

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/cli/test_models_commands.py

def test_models_entrypoint_guard_against_double_injection(monkeypatch):
    """AC8: if sys.argv[1] is already 'models', entrypoint does not inject again."""
    import sys
    from unittest.mock import patch

    monkeypatch.setattr(sys, "argv", ["models", "models", "-e"])

    with patch("conduit.apps.scripts.conduit_cli.main") as mock_main:
        from conduit.apps.scripts.conduit_cli import models_entrypoint
        models_entrypoint()

        # argv must not have a third 'models' prepended
        assert sys.argv == ["models", "models", "-e"]
        mock_main.assert_called_once()
```

- [ ] **Step 2: Run the test to verify it passes immediately**

```
pytest tests/cli/test_models_commands.py::test_models_entrypoint_guard_against_double_injection -v
```

Expected: `PASSED` — the guard is already implemented in Task 9's `if len(sys.argv) > 1 and sys.argv[1] == "models": pass`.

If it fails, the guard condition needs adjustment — re-check the `if` logic in `models_entrypoint`.

- [ ] **Step 3: Commit**

```bash
git add tests/cli/test_models_commands.py
git commit -m "test: double-injection guard for models_entrypoint (AC8)"
```

---

### Task 11: Update `pyproject.toml` and delete `models_cli.py`

**Files:**
- Modify: `pyproject.toml`
- Delete: `src/conduit/apps/scripts/models_cli.py`

**AC: All** — The `models` binary now routes through `conduit_cli:models_entrypoint`. All previous ACs remain satisfied.

- [ ] **Step 1: Update the `models` entry point in `pyproject.toml`**

Change:

```toml
models = "conduit.apps.scripts.models_cli:main"
```

To:

```toml
models = "conduit.apps.scripts.conduit_cli:models_entrypoint"
```

- [ ] **Step 2: Reinstall the package to pick up the new entry point**

```bash
cd /Users/bianders/Brian_Code/conduit-project
uv pip install -e .
```

- [ ] **Step 3: Smoke-test the new `models` binary**

```bash
models --help
models -e
models -r
models -t chat
conduit models --help
```

Expected: all commands produce output identical to pre-migration behavior.

- [ ] **Step 4: Delete `models_cli.py`**

```bash
rm src/conduit/apps/scripts/models_cli.py
```

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: all tests `PASSED` — verifies no other module was importing `models_cli` and that the entry point change did not break anything upstream.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml
git rm src/conduit/apps/scripts/models_cli.py
git commit -m "feat: wire models entry point to conduit_cli:models_entrypoint, delete models_cli.py"
```

---

## Final Verification

Run each command manually after Task 11 and check the box:

- [ ] AC1: `conduit models` — prints full model list (same as pre-migration `models`)
- [ ] AC2: `conduit models -e` — prints embedding model list with "Embedding models:" header
- [ ] AC3: `conduit models -r` — prints reranker model list with "Reranker models:" header
- [ ] AC4: `conduit models -m claude-3-5-sonnet` — prints model card
- [ ] AC4: `conduit models -m zzz` — prints fuzzy "Did you mean:" suggestions
- [ ] AC5: `conduit models -t chat` — prints filtered model list
- [ ] AC6: `conduit models --help` — all 6 flags (`-m`, `-t`, `-p`, `-a`, `-e`, `-r`) visible in output
- [ ] AC7: `models -e` — output identical to `conduit models -e`
- [ ] AC8: `models models -e` — output identical to `models -e` (no double-injection)
- [ ] AC9: `conduit query "hello"` — executes normally (no ModelStore import-time error or slowdown)
