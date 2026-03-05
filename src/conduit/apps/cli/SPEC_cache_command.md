# SPEC: `conduit cache` subcommand group

---

## 1. Goal

Add a `cache` subcommand group to the Conduit CLI that exposes read and write operations
against the Postgres-backed `conduit_cache_entries` table. The primary use cases are
inspection (how much is cached, how old is it) and manual eviction (clear one project,
prune by age). All operations default to the current project's `project_name` as the
`cache_name` scope.

---

## 2. Constraints and non-goals

**Constraints**
- All cache operations are scoped to `cache_name = project_name` by default. A `--all`
  flag enables the cross-project (global) view where noted per subcommand.
- The command layer must not contain SQL or async logic. SQL lives in `AsyncPostgresCache`;
  async orchestration lives in `CacheHandlers`, a new class in a new file
  `conduit/apps/cli/handlers/cache_handlers.py`. Do not add methods to `BaseHandlers`.
- Commands follow the existing pattern: Click context is unpacked in the command layer,
  a handler method is called with flat arguments, the handler calls
  `loop.run_until_complete`. Handlers do not import `click` or access `ctx`.
- Destructive operations (`clear`) prompt via `rich.prompt.Confirm` unless `--force`
  is passed. `--force` skips the prompt entirely; no prompt is shown, no input is read.
- Hit/miss counters (in-memory on `AsyncPostgresCache`) are NOT surfaced by the CLI.
  They reset on every process start and are meaningless here.
- No new DB schema changes. All queries target the existing `conduit_cache_entries` table.
- `cache` commands do not use the `verbosity` context value. Do not wire it up.
- All new `AsyncPostgresCache` methods must call `_ensure_ready()` at entry, exactly
  like existing methods.

**Non-goals**
- Entry-level deletion (by key). Cache keys are opaque hashes; no user-facing key
  resolution is in scope.
- TTL configuration or automatic expiration.
- Cache warming or pre-population.
- Hit/miss rate reporting.
- Export or serialization of cache payloads to file.
- Any changes to how the cache is written during `conduit query`.
- A `--database` flag. The database name is resolved from `settings`, not user input.
- A `--limit` flag on `ls`. The previous standalone `cache_cli.py` had one; it is not
  carried forward.
- `cache inspect --key` or `cache inspect --index`. Only the most-recent entry is shown.
- `--json` or `--quiet` output modes.
- `--dry-run` on `clear`.
- Resetting in-memory `_hits`, `_misses`, `_start_time` counters in `delete_older_than`.
  Only `wipe()` does this, as today.

---

## 3. Interface contracts

### Subcommand surface

```
conduit cache ls       [--all]
conduit cache clear    [--all] [--older-than DURATION] [--force]
conduit cache inspect
```

The `cache` group is created with `invoke_without_command=True`. When invoked with no
subcommand, it prints its own help text and exits 0.

---

### `cache ls`

Displays a Rich table. Columns, in order:

| Column     | Source                                                       |
|------------|--------------------------------------------------------------|
| Cache Name | `cache_name`                                                 |
| Entries    | `COUNT(*)`                                                   |
| Size       | `SUM(pg_column_size(payload))`: shown as bytes if < 1024,   |
|            | KiB if < 1,048,576, MiB otherwise; one decimal place.       |
| Oldest     | `MIN(created_at)` formatted as `YYYY-MM-DD`                  |
| Newest     | `MAX(updated_at)` formatted as `YYYY-MM-DD`                  |

**Default (no `--all`):** calls `cache_stats()` on the current project's
`AsyncPostgresCache` instance. If `total_entries` is 0, prints an informational message
("No cached entries for project '{project_name}'.") and returns without rendering a table.
If `total_entries` > 0, renders a single-row table.

**With `--all`:** calls `ls_all()` (new method, see below). If the result list is empty,
prints "No cached entries found." and returns without rendering a table. Otherwise renders
one row per result, ordered by `cache_name` ascending.

---

### `cache clear`

Flags:
- `--all` — target all `cache_name` values in the table (see SQL note below).
- `--older-than DURATION` — prune entries where `created_at < now() - DURATION`.
  `DURATION` must match the pattern `^\d+[dwh]$`. Valid examples: `7d`, `2w`, `48h`.
  `w` is converted to `N*7 days` (e.g. `2w` → `14 days`); `d` → `N days`;
  `h` → `N hours`. This normalization is done by `parse_duration` before the handler
  is called.
- `--force` — skips `Confirm.ask`. No prompt is shown.

**Mutually exclusive (hard errors, raised in command layer before handler call):**
- `--all` + `--older-than`: raises `click.UsageError`.

**`--all` clear SQL scope:** deletes `WHERE TRUE` (all rows, no `cache_name` filter).
This requires `wipe_all()` (new method), not `wipe()`. Do not loop over project names.

**`--older-than 0d` / `0h`:** rejected with `click.BadParameter`; a zero-duration prune
would delete everything and bypasses the intent of the flag.

**Success message:** always prints the count of rows deleted, e.g.:
"Cleared 42 entries." Even if 0 rows were deleted (e.g. nothing older than the cutoff),
print "Cleared 0 entries." — do not treat this as an error.

---

### `cache inspect`

No flags. Always scoped to the current project. Fetches the single most-recent entry by
`ORDER BY updated_at DESC LIMIT 1`.

Prints in this order:
1. `Cache Name: {cache_name}`
2. `Cache Key: {cache_key}`
3. `Timestamp: {updated_at}` formatted as `YYYY-MM-DD HH:MM:SS`
4. The `payload` field formatted with `rich.syntax.Syntax(json_str, "json", theme="monokai", line_numbers=True)`.

asyncpg returns JSONB columns as Python dicts, not strings. Serialize with
`json.dumps(payload, indent=2)` before passing to `Syntax`. If `json.dumps` raises
`TypeError` (non-serializable value), print the raw repr with a warning line above it.

---

### Handler signatures (`CacheHandlers` in `cache_handlers.py`)

```python
@staticmethod
def handle_cache_ls(
    project_name: str,
    all_projects: bool,
    printer: Printer,
    loop: asyncio.AbstractEventLoop,
    db_name: str,
) -> None: ...

@staticmethod
def handle_cache_clear(
    project_name: str,
    all_projects: bool,
    older_than: str | None,   # already a validated Postgres interval string, or None
    force: bool,
    printer: Printer,
    loop: asyncio.AbstractEventLoop,
    db_name: str,
) -> None: ...

@staticmethod
def handle_cache_inspect(
    project_name: str,
    printer: Printer,
    loop: asyncio.AbstractEventLoop,
    db_name: str,
) -> None: ...
```

`db_name` is sourced from `settings.db_name` and injected via `ctx.obj["db_name"]` in
`ConduitCLI._build_cli`, the same pattern as `project_name` and `preferred_model`.
Handlers do not import `settings`.

---

### New `AsyncPostgresCache` methods required

```python
async def ls_all(self) -> list[dict[str, object]]:
    """
    Return aggregate stats for every cache_name in the table.
    Each dict has exactly these keys:
        cache_name: str
        total_entries: int
        total_size_bytes: int
        oldest_record: str | None   # 'YYYY-MM-DD', None if no entries
        latest_record: str | None   # 'YYYY-MM-DD', None if no entries
    Ordered by cache_name ascending.
    Calls _ensure_ready() at entry.
    """
    ...

async def delete_older_than(self, pg_interval: str) -> int:
    """
    Delete entries for this cache_name where created_at < now() - interval.
    `pg_interval` is a validated Postgres interval string (e.g. '7 days', '48 hours').
    Returns the number of rows deleted.
    Does NOT reset _hits, _misses, or _start_time.
    Calls _ensure_ready() at entry.
    """
    ...

async def wipe_all(self) -> int:
    """
    Delete all rows in conduit_cache_entries with no cache_name filter.
    Returns the number of rows deleted.
    Calls _ensure_ready() at entry.
    """
    ...
```

`wipe()` is reused as-is for single-project full clear.

---

### Duration parsing

`parse_duration(value: str) -> str` lives in `conduit/apps/cli/utils/` as a standalone
function (not a method, not a Click callback). It is called in the command layer before
the handler is invoked.

Conversion rules:
```
"7d"   -> "7 days"
"2w"   -> "14 days"      # weeks expanded to days
"48h"  -> "48 hours"
"0d"   -> raises click.BadParameter("Duration must be greater than zero.")
"0h"   -> raises click.BadParameter("Duration must be greater than zero.")
```

Any string not matching `^\d+[dwh]$` raises `click.BadParameter` with a message that
includes the invalid value and the accepted format.

---

## 4. Acceptance criteria

Each criterion is stated as a specific, independently verifiable assertion. Tests that
require database state must seed that state explicitly via fixtures.

**`cache ls`**
- Given a DB with 5 entries for `cache_name = project_name`, `conduit cache ls` exits 0,
  stdout contains `project_name`, stdout contains `5`.
- Given a DB with 0 entries for `cache_name = project_name`, `conduit cache ls` exits 0,
  stdout contains "No cached entries for project", stderr is empty.
- Given a DB with entries for `cache_name` values `["a", "b", "c"]`, `conduit cache ls
  --all` exits 0 and stdout contains exactly three data rows (one per cache name), in
  alphabetical order.

**`cache clear`**
- Given entries for the current project, `conduit cache clear` (no `--force`) writes a
  confirmation prompt to stdout, and answering `n` exits 0 with zero rows deleted.
- `conduit cache clear --force` exits 0, stdout does NOT contain "Are you sure" or
  "confirm" (case-insensitive), and the count of rows remaining for that project is 0.
- `conduit cache clear --force` stdout contains the count of deleted rows as a numeral.
- Given a DB with 3 entries at `created_at = now() - 8 days` and 2 entries at
  `created_at = now() - 6 days` for the current project, `conduit cache clear
  --older-than 7d --force` exits 0, 3 rows are deleted, 2 rows remain.
- `conduit cache clear --older-than bad_value` exits non-zero, stderr or stdout contains
  the invalid value `bad_value`, and no rows are deleted.
- `conduit cache clear --older-than 0d` exits non-zero with a message containing
  "greater than zero". No rows are deleted.
- `conduit cache clear --all --older-than 7d` exits non-zero with a usage error before
  any DB connection is made.
- `conduit cache clear --all --force` exits 0 and all rows across all `cache_name`
  values are deleted.

**`cache inspect`**
- Given a DB with at least one entry for the current project, `conduit cache inspect`
  exits 0 and stdout contains the `cache_key` string and a substring matching
  `YYYY-MM-DD HH:MM:SS`.
- Given a DB with zero entries for the current project, `conduit cache inspect` exits 0,
  stdout contains "No cached entries", stderr is empty.

**`cache` (no subcommand)**
- `conduit cache` exits 0 and stdout contains "ls", "clear", and "inspect" (the help
  text lists available subcommands).

**Structural (verified by grep/static analysis, not runtime tests)**
- No file outside `conduit/apps/cli/commands/` imports `click.Context` or accesses
  `ctx.obj`.
- `CacheHandlers` methods do not import `click`, `settings`, or `asyncio` at module
  level (only under `TYPE_CHECKING` if needed).

---

## 5. Error handling / failure modes

| Failure                                  | Behavior                                                                         |
|------------------------------------------|----------------------------------------------------------------------------------|
| DB connection timeout (`TimeoutError`)   | Handler catches; prints error via `printer`; exits 1                             |
| DB connection refused / auth failure     | Handler catches `Exception`; logs via `logger.error`; prints error via `printer`; exits 1 |
| `conduit_cache_entries` table absent     | `_ensure_ready()` creates it transparently; no user-visible error                |
| No entries for current project           | Informational message via `printer`; exits 0                                     |
| Invalid `--older-than` format            | `click.BadParameter` raised in command layer before handler or DB call            |
| `--older-than 0d` / `0h`                | `click.BadParameter` raised in command layer before handler or DB call            |
| `--all` + `--older-than` together        | `click.UsageError` raised in command layer before handler or DB call              |
| Non-TTY stdin during `Confirm.ask`       | `EOFError` caught; prints "Operation cancelled."; exits 0                         |
| Keyboard interrupt during `Confirm`      | `KeyboardInterrupt` caught; prints "Operation cancelled."; exits 0                |
| `json.dumps` raises `TypeError` on payload | `inspect` prints a warning line, then `repr(payload)`; exits 0                 |
| Unexpected exception in handler          | Caught; logged via `logger.error`; printed via `printer.print_pretty`; exits 1   |

Handlers do not re-raise exceptions to Click. All exceptions from `loop.run_until_complete`
are caught within the handler. Exit codes are set via `sys.exit`, not by raising
`SystemExit` inside async code.

---

## 6. Style example

The following shows the conventions to follow. It is not part of the implementation.

```python
# In commands/cache_commands.py

import asyncio
import click
from conduit.apps.cli.handlers.cache_handlers import CacheHandlers
from conduit.apps.cli.utils.duration import parse_duration
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.apps.cli.utils.printer import Printer

handlers = CacheHandlers()


@click.group(invoke_without_command=True)
@click.pass_context
def cache(ctx: click.Context) -> None:
    """Inspect and manage the Postgres query cache."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cache.command()
@click.option("--all", "all_projects", is_flag=True, help="Show all cache namespaces.")
@click.pass_context
def ls(ctx: click.Context, all_projects: bool) -> None:
    """List cache entries for the current project (or all projects with --all)."""
    printer: Printer = ctx.obj["printer"]
    loop: asyncio.AbstractEventLoop = ctx.obj["loop"]
    project_name: str = ctx.obj["project_name"]
    db_name: str = ctx.obj["db_name"]

    handlers.handle_cache_ls(
        project_name=project_name,
        all_projects=all_projects,
        printer=printer,
        loop=loop,
        db_name=db_name,
    )
```

Note: `ctx.obj["db_name"]` must be set in `ConduitCLI._build_cli` from `settings`,
exactly as `project_name` and `preferred_model` are set today. The `cache` group is
attached to the root CLI via `cli.add_command(cache)`, not via a `CommandCollection`.
