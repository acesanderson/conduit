"""
Not ready yet.

Batch reporting is different from our other reporting because it is NOT handling in query @middleware; but rather managed by ConduitBatchAsync class.

Next steps:
- [ ] implement EngineResult/ConduitResult with ResponseMetadata
- [ ] move the hooks from the dummy class to ConduitBatchAsync, including silencing Verbosity when passing to Conduit
- [ ] implement different Verbosity levels
"""

import asyncio
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

# Use Rich for the actual heavy lifting
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.console import Group

# ==============================================================================
# 1. MOCK DOMAIN & RESULT OBJECTS (Unchanged)
# ==============================================================================


@dataclass
class MockResponseMetadata:
    duration: float
    model_slug: str
    input_tokens: int
    output_tokens: int
    cache_hit: bool = False
    type: Literal["generate"] = "generate"


@dataclass
class MockMessage:
    role: str
    content: str
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MockConversation:
    messages: list[MockMessage] = field(default_factory=list)

    @property
    def last(self) -> MockMessage | None:
        return self.messages[-1] if self.messages else None

    @property
    def content(self) -> str:
        return self.last.content if self.last else ""


@dataclass
class MockEngineResult:
    conversation: MockConversation
    metadata: MockResponseMetadata

    @property
    def last(self) -> MockMessage:
        return self.conversation.last

    @property
    def content(self) -> str:
        return self.conversation.content


ConduitResult = MockEngineResult

# ==============================================================================
# 2. RICH BATCH REPORTER
# ==============================================================================


class BatchReporter:
    def __init__(self, project_name: str, total: int):
        self.project_name = project_name
        self.total = total
        self.cached = 0
        self.fresh = 0
        self.active_ids = set()
        self.event_log = deque(maxlen=5)

        # Setup the progress bar component
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[project]}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• {task.completed}/{task.total}"),
            expand=False,
        )
        self.task_id = self.progress.add_task(
            "batch", total=total, project=project_name
        )

    def mark_active(self, source_id: str):
        self.active_ids.add(source_id)

    def report_completion(self, source_id: str, result: ConduitResult):
        self.active_ids.discard(source_id)
        is_cache = result.metadata.cache_hit

        if is_cache:
            self.cached += 1
            icon, label, style = "⚡", "Cached", "cyan"
        else:
            self.fresh += 1
            icon, label, style = "✓ ", "Fresh ", "green"

        self.progress.update(self.task_id, advance=1)

        # Add to scroll
        event_str = f"[{style}]{icon} {label}[/{style}]: {source_id} ({result.metadata.duration:.2f}s)"
        self.event_log.append(event_str)

    def __rich__(self) -> Group:
        """This defines the layout for rich.live"""
        pending = self.total - (self.cached + self.fresh)

        # 1. Status Summary Line
        status_line = (
            f"[bold cyan]{self.cached}⚡[/] Cached  "
            f"[bold green]{self.fresh}✓  [/] Fresh  "
            f"[bold yellow]{len(self.active_ids)}⏳[/] Active  "
            f"[dim]{pending} Pending[/]"
        )

        # 2. Recent Events List
        events = (
            "\n".join(list(self.event_log))
            if self.event_log
            else "[dim]Waiting for events...[/]"
        )

        # Combine zones into a group
        return Group(self.progress, status_line, "\n[bold]Recent Events[/]", events)


# ==============================================================================
# 3. SIMULATOR (Uses Live context manager)
# ==============================================================================


class BatchSimulator:
    def __init__(self, reporter: BatchReporter):
        self.reporter = reporter

    async def simulate_item(self, source_id: str):
        self.reporter.mark_active(source_id)
        is_cache = random.random() < 0.3
        delay = random.uniform(0.05, 0.2) if is_cache else random.uniform(1.0, 3.0)
        await asyncio.sleep(delay)

        meta = MockResponseMetadata(
            duration=delay,
            model_slug="mock",
            input_tokens=10,
            output_tokens=10,
            cache_hit=is_cache,
        )
        result = ConduitResult(conversation=MockConversation(), metadata=meta)
        self.reporter.report_completion(source_id, result)

    async def run(self, total_items: int, max_concurrent: int = 5):
        ids = [f"doc_{i:03d}_{uuid.uuid4().hex[:4]}" for i in range(total_items)]
        semaphore = asyncio.Semaphore(max_concurrent)

        # Start the LIVE display here
        with Live(self.reporter, refresh_per_second=10, vertical_overflow="visible"):

            async def sem_task(sid):
                async with semaphore:
                    return await self.simulate_item(sid)

            tasks = [sem_task(sid) for sid in ids]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    reporter = BatchReporter("gold_standard_v2", total=30)
    sim = BatchSimulator(reporter)
    asyncio.run(sim.run(total_items=30, max_concurrent=6))
