# Chain.Chat Design Specification

**Project Name:** Chain.Chat
**Version:** 0.1
**Author:** Brian Anderson
**Last Updated:** 2025-07-13

---

## Overview

`Chain.Chat` is a terminal-based conversational assistant built around the concept of a branching message tree, inspired by Claude Desktop's editable conversation model and tmux's modal command system. It supports multiple conversation paths, rapid forking, and keyboard-only navigation, optimized for power users and developers working in a flow state.

---

## Goals

* Allow users to branch from any user message to explore alternate directions
* Maintain a tree-like structure of conversations
* Provide tmux-style leader-key driven commands
* Enable persistent storage and recall of conversation threads
* Offer fast, keyboard-driven navigation and manipulation

---

## Core Architecture

### Conversation Tree

```python
class MessageNode:
    msg_id: str
    role: Literal["user", "assistant"]
    content: str
    parent: Optional[MessageNode]
    children: list[MessageNode]

class ConversationTree:
    paths: dict[str, list[MessageNode]]
    active_path: str
```

* Each branch is a named or indexed path (e.g. `/main`, `/main/alt1`)
* Editing or branching creates a new fork with its own lineage

---

## Command UX

### Prefix Key System

* Default: `Ctrl-g` (configurable)
* Enters command mode, inspired by `tmux`

### Key Bindings

| Binding        | Action                                   |
| -------------- | ---------------------------------------- |
| `prefix + c`   | Create a new branch from current message |
| `prefix + e`   | Edit previous user message               |
| `prefix + w`   | List all branches                        |
| `prefix + n/p` | Move to next/previous branch             |
| `prefix + 0-9` | Jump to branch by index                  |
| `prefix + ,`   | Rename current branch                    |
| `prefix + &`   | Delete current branch                    |
| `prefix + [`   | Scrollback / copy mode                   |
| `prefix + :`   | Manual command entry                     |

### Manual Commands

| Command           | Description                        |
| ----------------- | ---------------------------------- |
| `/branch <U#>`    | Branch from a user message         |
| `/edit <U#>`      | Edit a previous message, fork path |
| `/goto <path>`    | Switch to another branch path      |
| `/tree`           | Show conversation tree view        |
| `/list`           | List all branches                  |
| `/rename <a> <b>` | Rename a branch                    |
| `/delete <path>`  | Delete a branch                    |
| `/save <path>`    | Save current path with name        |
| `/summary <path>` | Summarize a branch via LLM         |
| `/diff <p1> <p2>` | Compare two branches               |

---

## Thread Management UI

### Sidebar (Recent Threads List)

* Modeled after Claude Desktop's left nav
* Persistent across sessions via metadata file

```json
{
  "threads": [
    {
      "id": "main",
      "title": "Lazy Loading Client Libraries",
      "path": "~/.chainchat/threads/main.json",
      "last_active": "2025-07-13T17:45:00"
    }
  ]
}
```

* Supports keyboard navigation: Up/Down, Enter to load, `d` to delete, `r` to rename

### Conversation Storage

* Each conversation is saved as a `.json` file in `~/.chainchat/threads/`
* Example:

  * `~/.chainchat/threads/main.json`
  * `~/.chainchat/threads/alt-funny.json`

---

## Optional Features (To Be Considered)

### Fuzzy Search

* `prefix + /` — Search across all messages for keyword matches
* Returns message IDs, snippets, and paths to jump into

### Flow Mode (Rapid Forking)

* `prefix + f` — Auto-forking loop after each LLM response
* Prompts user to "Try a variation?"

### Tagging Messages / Branches

* `prefix + t` — Add tag to a message or branch
* `prefix + T` — Jump to tag, list all

### Replay / Time Travel

* `prefix + r` — Replay full conversation thread
* Options to export as JSON, Markdown, or notebook

### GPT-Assisted Naming

* When creating new branches, auto-suggest semantic names
* e.g. `/persist_cache`, `/funny_names`, `/final_version`

### Scratchpad / Notes

* `prefix + n` — Open branch-specific notes
* Stored alongside conversation data

### Session Switching

* `prefix + s` — Load from a different project/session root
* Organize threads into logical project contexts

### Macros / Prompt Chains

* Record and replay multi-message workflows
* e.g. run same set of prompts with different models or parameters

---

## Design Philosophy

> Optimize for creative flow and context exploration. Prioritize keyboard-first navigation, structural clarity, and rapid branching to support deep iteration and focused thought.

---

## Next Steps

* Implement MessageNode + ConversationTree core structure
* Build basic CLI prompt + prefix detection
* Scaffold `/tree`, `/branch`, and `/edit` commands
* Create JSON thread persistence and sidebar list viewer
* Optional: prototype with `prompt_toolkit` or `urwid`

