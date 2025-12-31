# AGENT CONTEXT: PROJECT "TAP"

## Tech Stack

* **Python:** 3.12+
* **UI Framework:** `textual` (latest stable)
* **Rendering:** `rich` (latest stable)
* **Core Library:** `conduit` (internal library - assume available in `PYTHONPATH`)

## Core Philosophy

* **State is a Tree:** The data model is a DAG (Directed Acyclic Graph) of messages.
* **UI is a List:** The view is a flattened linear projection of the *active branch*.
* **Immutability:** Never mutate history; always fork.

## Testing Strategy

* **NO UI MOCKS:** Do not try to test `App.run()` or simulate keystrokes in unit tests.
* **Logic First:** Test `TapConversationManager` heavily (branching, flattening, navigating).
* **Tooling:** Use `pytest`.

---

# Tap TUI — UX & Architecture Specification (v0.3.1)

> **Status:** Foundational / normative
> **Audience:** Future-you, collaborators, LLM copilots
> **Non-goal:** Styling, backend APIs, persistence format, implementation code

## 0. Purpose

Tap is the **primary interaction surface** of the Watershed system.

It is:

* terminal-native
* keyboard-first
* stateful
* optimized for long-lived, high-trust use

Tap is **not**:

* a chat toy
* a passive dashboard
* a tree visualizer
* a prompt playground

Tap renders a **linearized event stream** and allows **precise, reversible interaction** with it.

---

## 1. Core domain concepts

### 1.1 Event

An **event** is an atomic, ordered item in a transcript.

Event types include (non-exhaustive):

* `UserMessage`
* `AssistantMessage`
* `SystemEvent`
* `ToolEvent` (future)

Each event has:

* stable identity
* type
* renderable content
* metadata (branch index, cached flag, timestamps, etc.)
* declared capabilities:
* expandable
* editable
* branchable



Events are **never mutated in place** once committed to a conversation.

### 1.2 Conversation

A **conversation** is an immutable, persisted record consisting of:

* an ordered event stream
* a single active branch path

Any edit to history creates a **new conversation**.
The previous conversation remains addressable and unchanged.

### 1.3 Branch

A **branch** is an alternate event sequence diverging at a specific event.

Properties:

* local to a single event
* navigated horizontally
* never merged implicitly
* never destroyed implicitly

Branching is explicit, visible, and reversible.

---

## 2. Layout model

Tap has a **single persistent layout**.

### 2.1 Vertical split

The UI is split vertically into two panes:

#### Top pane — Transcript View

* Displays the conversation event stream
* Scrollable
* Event-based selection (not line-based)
* Rendering is derived entirely from application state
* Uses Rich renderables defined per event type

#### Bottom pane — Input Editor

* Fixed height: **exactly 5 lines**
* Always visible
* Multiline text input
* Not scroll-linked to the transcript

The bottom pane **never resizes**.
The top pane flexes to fill remaining space.

---

## 3. Interaction modes

Tap operates in **exactly one global mode** at any time.

Modes determine:

* which component has focus
* which keybindings are active
* how input is interpreted

### 3.1 Normal mode

**Purpose:** navigation, inspection, structural actions

#### Focus

* Focus is on the Transcript View
* Exactly **one event is always selected**
* Default selection on entry: latest event

Selection is **event-based**, not cursor- or line-based.

#### Navigation

* `j / k` or `↑ / ↓` → move selection vertically
* `h / l` or `← / →` → navigate branches (if available)

Navigation never mutates conversation state.

#### Actions (non-exhaustive)

* Toggle expand / collapse (only if event is expandable)
* Copy event content to clipboard
* Edit / rewrite selected message
* Regenerate selected assistant message
* Open input editor for appending
* Enter command mode (`:`)

Actions operate on **the selected event only**.

### 3.2 Insert mode

**Purpose:** authoring intent

#### Focus

* Focus moves to the Input Editor (bottom pane)
* Transcript remains visible but inert
* Selection is preserved and does not change

#### Input semantics

* Input is multiline by default
* `Shift+Enter` inserts a newline
* `Enter` submits input

#### Submission rules

* If no assistant is pending:
* submission creates a new `UserMessage`


* If an assistant is pending:
* submitted text is appended to the most recent `UserMessage`
* the assistant task is cancelled and restarted



#### Exit

* `ESC` exits Insert mode
* Focus returns to Transcript View
* Unsubmitted input is preserved

#### Constraints

* **Strict Constraint:** Do not implement Vim motions beyond `h/j/k/l`.
* Do not implement counts (e.g., `5j`).
* Do not implement verbs (e.g., `d` for delete).
* Only the specific bindings listed in this spec are allowed.

### 3.3 Command mode

**Purpose:** explicit command execution (Vim-style)

Command mode is an **overlay**, not part of the main layout.

#### Entry

* Entered explicitly (e.g. via `:` in Normal mode)

#### Behavior

* A floating command-line popup appears
* Overlay captures **all input exclusively**
* Normal and Insert bindings are suspended

#### Exit and execution

* `Enter` executes the command
* `ESC` aborts and returns to Normal mode

Command mode does not modify state unless a command is explicitly executed.

#### Command prefix compatibility

Tap supports commands written in either style:

* Vim style: `:help`, `:set model haiku`
* Slash style (legacy compatibility): `/help`, `/set model haiku`

Tap normalizes these internally before parsing.

### 3.4 Mode transitions

Global invariants:

* `ESC` always returns to Normal mode
* Mode transitions never:
* destroy unsubmitted input
* change the active conversation implicitly
* change the active branch implicitly



---

## 4. Transcript semantics

### 4.1 Transcript model

* Transcript is a vertical, ordered stream of events
* Rendering is fully derived from state
* Terminal scrollback is never the source of truth

### 4.2 Expand / collapse (accordion behavior)

* Events may declare themselves expandable
* Expand/collapse state is per-event
* Expand/collapse:
* does not affect selection
* does not affect branching
* is fully reversible



### 4.3 Compact view (rendering lens)

Compact view is **not a mode**.

Properties:

* Truncates event content (e.g. first ~90 chars)
* Does not affect:
* selection
* focus
* active branch
* conversation identity


* Resets every session

Actions:

* Expand all
* Collapse all

---

## 5. Editing & branching

### 5.1 Edit entry points (Normal mode)

There are exactly three ways to leave Normal mode to edit text:

#### `A` — amend highlighted message

* Applies to selected `UserMessage`
* Editor prefilled with existing content
* Cursor at end

#### `C` — rewrite highlighted message

* Applies to selected `UserMessage`
* Editor starts empty

#### `O` — open main input

* Focus moves to Input Editor
* Cursor positioned to append a new message

### 5.2 Edit submission semantics

On submission from any edit path:

* A **new conversation** is created
* Original conversation remains unchanged
* New conversation becomes active
* Branch provenance is preserved

### 5.3 Editing while assistant pending

If:

* assistant is streaming
* selected event is the most recent `UserMessage`
* user initiates editing

Then:

* assistant task is cancelled
* user message becomes editable
* submission creates a new conversation branch

---

## 6. Branch navigation

### 6.1 Visibility

* Branch indicators are always visible when branches exist
* When event is not selected → indicators subdued
* When selected → indicators fully visible

### 6.2 Navigation

* `h / l` cycles alternates
* Only branchable events respond

### 6.3 Active path invariant

Branching never changes the active path implicitly:

* creating a branch does not switch conversations
* viewing alternates does not commit to them
* active path changes only via explicit user action

---

## 7. Streaming behavior

### 7.1 Logical streaming

* Assistant messages may stream incrementally
* Streaming mutates render state, not event identity

### 7.2 Streaming + navigation

If:

* assistant is streaming
* selection moves away

Then:

* streaming continues logically
* streaming pauses visually for that event
* navigation remains unaffected

When selection returns:

* live streaming view resumes

### 7.3 Streaming + Scrolling (Sticky Scroll)

* **Invariant:** If the Transcript View is scrolled to the absolute bottom (showing the latest tokens), it must **automatically scroll** to keep new tokens visible as they arrive.
* **Invariant:** If the user has scrolled up to inspect history, incoming tokens must **not** hijack the scroll position. The view must remain stable.
* **Implementation:** Use `scroll_to_end(animate=False)` explicitly on stream updates *only if* the user was already at the bottom.

---

## 8. Safety & trust guarantees

Tap must never:

* lose a branch
* overwrite an event
* change active path implicitly
* clear context implicitly

Destructive actions:

* require confirmation
* are visually distinct
* are never bound to ambiguous keys

Unsubmitted input:

* preserved across mode switches
* survives redraws and overlays whenever possible

---

## 9. Input invariants

* Two `UserMessage` events may never exist consecutively
* Until an `AssistantMessage` exists:
* the most recent `UserMessage` is mutable


* Submitting while assistant pending appends to that message

---

## 10. Environment constraints

Tap must remain usable in:

* SSH sessions
* tmux panes
* mouse-less environments

Therefore:

* no mouse dependency
* no reliance on Unicode-heavy glyphs
* indicators degrade gracefully

---

## 11. Command system specification

### 11.1 Command registry

Tap maintains a registry of commands.

Properties:

* Commands may be single-token (`help`) or multi-word (`set model`).
* Each command has:
* a name (string)
* a handler (async)
* metadata (help text, usage)



Help and completion are generated from this registry.

### 11.2 Parsing and resolution rules

Given a command line entered in Command mode:

1. **Normalize prefix**
* Strip a leading `:` or `/` if present.


2. **Resolve command name by longest match**
* Match the longest registered command prefix against the input string.
* This prioritizes multi-word commands over shorter prefixes.


3. **Parse arguments**
* Arguments are tokenized by whitespace, **except** quoted substrings:
* `"like this"` is treated as one argument.




4. **Unknown commands**
* If no handler is found, Tap must render a visible error (e.g. a `SystemEvent`):
* “Unknown command: … (try :help)”





### 11.3 Execution semantics

* Commands execute asynchronously.
* Commands may:
* return renderable output
* return no output but cause state changes
* request explicit UI actions (e.g. exit, clear screen)



Command execution must be explicit and must not occur on overlay open.

### 11.4 Output routing (high-trust rule)

Tap must not rely on terminal stdout/stderr printing for user-visible output.

Command results must be routed as either:

* a persisted `SystemEvent` appended to the transcript (default), or
* an ephemeral UI notification (only for intentionally transient feedback)

Commands that only mutate state may still emit a `SystemEvent` for auditability.

---

## 12. Technical specification: implementation with Textual

This section describes **how the above UX maps onto Textual conceptually**, without prescribing code.

### 12.1 Application state

The Textual app owns:

* global mode (`Normal | Insert | Command`)
* active conversation
* selected event ID
* transcript scroll position
* input buffer state
* overlay stack (command mode, dialogs)
* streaming task state (assistant run, cancellation tokens)

Widgets **do not own domain state**.

### 12.2 Widget responsibilities

#### Transcript View widget

* **Structure:** Must be implemented as a `ListView`.
* **Children:** Each child is a `ListItem` containing a single `Static` widget.
* **Rendering:** The `Static` widget holds the `conduit` Message object and renders it via `update(message)`.
* **Selection:** Do not track index manually. Use `ListView.index` and `highlighted_child`.
* **Scrolling:** Handles sticky scrolling logic defined in Section 7.3.

#### Input Editor widget

* Maintains editable text buffer
* Emits submit events upward
* Does not decide submission semantics
* Preserves unsubmitted input across mode transitions

#### Command Overlay widget

* Captures all input while active
* Provides completion and help affordances from the command registry
* Parses and dispatches commands as application actions
* Dismisses itself explicitly on execution or abort

### 12.3 Focus and key routing

* Keybindings are resolved at the application level.
* Active mode determines which bindings are enabled.
* Focus is shifted explicitly on mode transitions.
* Overlays suspend underlying input handling entirely.

### 12.4 Rendering model

* All rendering is derived from application state.
* Rich renderables are used for event display.
* No widget relies on terminal scrollback.
* Redraws are idempotent and safe.

### 12.5 Concurrency and streaming

* Assistant streaming runs as background tasks.
* UI updates are reactive to state changes.
* Streaming can be visually suppressed per event without cancelling the task.

### 12.6 Output and logging constraints

* User-visible output must be represented as events or UI notifications.
* Stray stdout/stderr output is considered a UI corruption risk.
* Logging should be directed to a file or a dedicated in-app debug surface.

---

# Tap TUI Implementation Plan

## Phase 1: The Domain Core (The Tree)

**Goal:** Create the data structure that handles branching, which `conduit`'s native linear `Conversation` does not support.

### Chunk 1.1: The Tree Node

* **File:** `tap/domain/state.py`
* **Requirements:**
* Define `TapNode`: A wrapper class.
* **Attributes:**
* `id`: UUID (stable identity).
* `parent_id`: UUID | None.
* `children_ids`: list[UUID] (ordered).
* `message`: `conduit.domain.message.message.Message` (The actual payload).
* `is_active_path`: boolean (helper for UI rendering).
* **Invariants:** `message` content is immutable once finalized.

### Chunk 1.2: The Tree Manager

* **File:** `tap/domain/manager.py`
* **Requirements:**
* Define `TapConversationManager`.
* **State:**
* `nodes`: dict[UUID, TapNode].
* `root_id`: UUID.
* `current_leaf_id`: UUID (The tip of the currently viewed branch).
* **Methods:**
* `add_message(message: Message, parent_id: UUID) -> UUID`: Appends a new node.
* `get_thread(leaf_id: UUID) -> conduit.domain.conversation.Conversation`: Walks backwards from leaf to root to construct a linear Conduit conversation for the Engine.
* `branch_at(node_id: UUID, new_message: Message) -> UUID`: Creates a sibling for `node_id`'s child (or a child of `node_id`), automatically switching `current_leaf_id` to this new path.
* `Maps_horizontal(node_id: UUID, direction: int)`: Logic to find the next/prev sibling and recalculate the `current_leaf_id` based on the most recently used path down that branch.

### Chunk 1.3: Unit Tests (Logic Only)

* **File:** `tests/test_domain_state.py`
* **Tests:**
* Verify `get_thread` returns messages in correct chronological order.
* Verify `branch_at` creates a new path without destroying the old one.
* Verify flattening the tree produces a valid `conduit.Conversation`.

---

## Phase 2: TUI Scaffold (Read-Only)

**Goal:** Get the existing `conduit` Rich renderables showing up in a Textual App.

### Chunk 2.1: The Layout & Widget

* **File:** `tap/ui/app.py`
* **Requirements:**
* `TranscriptView(ListView)`: A widget that takes a list of `TapNode`s.
* `InputArea(TextArea)`: Fixed height (5 lines), docked bottom.
* **Rendering:** Use `ListItem` containing a `Static`. Pass `node.message` directly to the `Static`. Textual will call `__rich_console__` automatically.
* **Styling:** Minimal CSS to handle the split view.

### Chunk 2.2: State Wiring

* **File:** `tap/controller.py`
* **Requirements:**
* Initialize `TapConversationManager` with a `SystemMessage`.
* On `mount`: Populate `TranscriptView` with the initial thread.
* **Manual Check:** Run the app. Ensure the System Message renders nicely using the existing Conduit styles.

---

## Phase 3: Interaction & Engine Wiring

**Goal:** Connect the TUI to the `conduit` Engine.

### Chunk 3.1: The Controller & Async Loop

* **File:** `tap/controller.py` (Update)
* **Requirements:**
* Implement `submit_message(text: str)`:

1. Create `UserMessage(content=text)`.
2. Update Domain Tree (`add_message`).
3. Update UI (push widget).
4. **Async Task:** Call the model.

* **Implementation Detail:** Do not use `Engine.run` for this version. You must use `ModelAsync.pipe(request)` directly in the Tap Controller to receive an async generator of tokens. This allows the TUI to update the UI on every token yield without modifying the core `conduit` library.

### Chunk 3.2: Keyboard Bindings & Modes

* **File:** `tap/ui/app.py`
* **Requirements:**
* **Normal Mode:** `j/k` scroll, `i` enter insert mode.
* **Insert Mode:** `Esc` to normal, `Enter` to submit.
* **Branching:** `h/l` calls `manager.navigate_horizontal`.
