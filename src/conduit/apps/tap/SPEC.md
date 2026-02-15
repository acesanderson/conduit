# Tap TUI — UX & Architecture Specification (v0.4.0)

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

Tap renders a **linearized event stream** and allows **precise, reversible interaction** with it via a Directed Acyclic Graph (DAG) state model.

---

## 1. Core domain concepts

### 1.1 Event

An **event** is an atomic, ordered item in a transcript.
Event types: `UserMessage`, `AssistantMessage`, `SystemEvent`, `ToolEvent`.

Each event has:

* stable identity and type
* renderable content
* **Metadata Store:** includes `toc_summary` (one-liner), timestamps, and branch indices.

### 1.2 Conversation

An **immutable** record consisting of an ordered event stream and a single active branch path. Any modification (editing, deleting, branching) creates a **new conversation** object.

### 1.3 Branch

An alternate event sequence diverging at a specific event. Navigated horizontally; never implicitly merged or destroyed.

---

## 2. Layout model

Tap has a **single vertical split layout**.

### 2.1 Top pane — Transcript View

* Displays the conversation event stream.
* Event-based selection (not line-based).
* **Outline Lens:** Can toggle into a collapsed "Table of Contents" view.

### 2.2 Bottom pane — Input Editor

* **Fixed height: exactly 5 lines.**
* Always visible; multiline text input.
* Not scroll-linked to the transcript.

---

## 3. Interaction modes

### 3.1 Normal mode

**Purpose:** navigation, inspection, structural actions.

#### Navigation

* `j / k` or `↑ / ↓` → move selection vertically.
* `h / l` or `← / →` → navigate branches (cycle alternate responses).
* **`TAB`** → Toggle **Outline Lens** (TOC view).

#### Actions

* `A` → amend highlighted `UserMessage`.
* `C` → rewrite highlighted `UserMessage` (creates new branch).
* `O` → open main input for appending.
* `D` → delete selected message (confirmation required).
* `m` → mark message for range deletion.
* `:` → Enter command mode.

### 3.2 Insert mode

**Purpose:** authoring intent.

* `Enter` → Submit input.
* `Shift+Enter` → Insert newline.
* `ESC` → Exit to Normal mode.

### 3.3 Command mode

**Purpose:** explicit command execution (Vim-style).

* Floating overlay capturing all input.
* Supports prefix compatibility: `:set model` or `/set model`.

---

## 4. Transcript & Outline Semantics

### 4.1 Outline Lens (TOC)

The Outline Lens is a rendering filter for the Transcript View that collapses messages into single-line summaries for rapid scanning.

* **Selection Invariant:** The currently selected event in Normal mode remains selected when toggling the Lens.
* **Rapid Rewind:** Users can toggle the Lens, move `j/k` quickly to a previous turn, and hit `C` or `h/l` to pivot history from that specific point.

### 4.2 Asynchronous Summarization

Summarization is a non-blocking side-effect that occurs after an event is committed to history.

1. **Commit:** Main message is saved to the repository.
2. **Trigger:** An async task is dispatched to the background worker.
3. **Escalation:** If a user toggles the Outline Lens while an event is "Summary Pending," that specific task is moved to the top of the queue.
4. **Placeholder:** While pending, the TUI displays an ephemeral summary based on **Lead Sentence Extraction**.

### 4.3 Summarization Strategies

Tap uses a hierarchy of strategies to populate `metadata['toc_summary']`:

| Strategy | Type | Use Case |
| --- | --- | --- |
| **LLM Summary** | Async/LLM | Preferred for `AssistantMessage`. 5-8 word intent summary using a lightweight model. |
| **Lead Sentence** | CPU-Bound | First salient sentence. Used as an instant fallback or for `SystemEvents`. |
| **Key-Phrase Ranking** | CPU-Bound | Top 3-4 nouns (e.g., `Postgres |
| **Intent Snippet** | Heuristic | First 50 chars of `UserMessage`. Default for human prompts. |

---

## 5. Editing & Branching

### 5.1 Edit Submission

On submission from any edit path (`A`, `C`, or `O`), a **new conversation** is created (forked). The original conversation remains addressable.

### 5.2 Editing while Assistant Pending

If an assistant is streaming and the user initiates an edit on the preceding `UserMessage`, the assistant task is **cancelled immediately**, and the user enters Insert mode to modify the prompt.

---

## 6. Context Pruning (Deletion)

### 6.1 Range Deletion

* `m` marks the start of a range.
* Selecting a second message and hitting `D` deletes the range.
* **Safety:** Cannot delete the root `SystemMessage` or create a state where two `UserMessage` events are consecutive.

---

## 7. Streaming & Scrolling

### 7.1 Sticky Scroll

* If the user is at the absolute bottom, the view auto-scrolls to keep new tokens visible.
* If the user has scrolled up to inspect history, the view remains stable (no scroll-hijacking).

---

## 8. Safety & Trust

* **No Implicit Mutation:** Any destructive action creates a fork, ensuring history is never truly lost.
* **Confirmation Dialogs:** Required for `D` (deletion) and `/wipe`.
* **Persistent Input:** Unsubmitted text in the Input Editor survives mode transitions and most UI crashes.

---

## 9. Internal Design: Summarization Flow

1. **Engine** produces `AssistantMessage`.
2. **Middleware** interceptor detects `GenerationResponse`.
3. **NLP Processor** (Background Task):
* Estimates tokens.
* Selects strategy (Lead Sentence if model is slow, LLM Summary if available).
* Injects result into `message.metadata['toc_summary']`.


4. **Repository** upserts the message with enriched metadata.
5. **TUI** reacts to metadata update and refreshes the current line in the Transcript View.

---

**Next Step:** Would you like me to create the Phase 1.1 `TapNode` and `TapConversationManager` Python classes, ensuring they include the `metadata` slots for these `toc_summary` fields?





