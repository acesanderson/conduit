# Enhanced Chat UI Design Specification

## Overview

A vim-inspired terminal UI for the Chain library chat functionality, designed to work seamlessly over SSH with support for conversation branching, history management, and advanced navigation. The architecture supports migration from Rich to Textual with minimal code changes.

## Core Design Philosophy

- **Keyboard-First**: All interactions via keyboard shortcuts (no mouse dependency)
- **Vim-Inspired**: Leader key + mode system similar to vim/tmux
- **SSH Compatible**: Works reliably over terminal connections
- **Framework Agnostic**: Business logic separated from UI implementation

## Mode System

### Mode Definitions
- **Chat Mode** (default): Normal conversation flow
- **Message Mode** (`<leader>[`): Navigate/select messages with j/k
- **Tree Mode** (`<leader>t`): Conversation browser
- **Search Mode** (`<leader>s`): Global conversation search  
- **Command Mode** (`<leader>:`): Execute chat commands
- **Buffer Navigation**: `<leader>h`/`<leader>l` for prev/next conversation, `<leader>b` for buffer search

### Key Bindings
- **Leader Key**: Configurable (default: space)
- **Universal Escape**: ESC always returns to Chat Mode
- **Help System**: `<leader>?` shows available keys in current mode

## Visual Layouts

### Chat Mode (Default)
```
┌─────────────────────────────────────────────────────────────────┐
│ system: You are a helpful assistant                            │
│                                                                 │
│ user: What is the capital of France?                          │
│                                                                 │
│ assistant: The capital of France is Paris. It has been the    │
│ political and cultural center of France for centuries and is   │
│ known for landmarks like the Eiffel Tower and the Louvre.     │
│                                                                 │
│ user: Tell me more about the Eiffel Tower                     │
│                                                                 │
│ assistant: The Eiffel Tower was built between 1887 and 1889   │
│ for the 1889 World's Fair...                                  │
│                                                                 │
│ >> _                                                           │
└─────────────────────────────────────────────────────────────────┘
│ CHAT │ openai/gpt-4o │ Tokens: 1.2k/4k ████████░░ │ "Paris Chat" (3/12) │
```

### Message Mode (`<leader>[`)
```
┌─────────────────────────────────────────────────────────────────┐
│ system: You are a helpful assistant                            │
│                                                                 │
│ user: What is the capital of France?                          │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ assistant: The capital of France is Paris. It has been the │ │ <- Selected
│ │ political and cultural center of France for centuries and  │ │
│ │ known for landmarks like the Eiffel Tower and the Louvre.  │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ user: Tell me more about the Eiffel Tower                     │
│                                                                 │
│ assistant: The Eiffel Tower was built between 1887 and 1889   │
│ for the 1889 World's Fair...                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│ MESSAGE │ j/k:nav e:edit c:copy ESC:exit │ Msg 2/4 │ anthropic/claude │
```

### Tree Mode (`<leader>t`)
```
┌─────────────────────────────────────────────────────────────────┐
│                        CONVERSATIONS                           │
│                                                                 │
│ ┌─ 📁 Today                                                    │
│ │  ├─ ► Paris Chat (4 msgs)                    [CURRENT]      │
│ │  ├─ ► Python Help (12 msgs)                               │
│ │  └─ ► Code Review (8 msgs)                                │
│ │                                                             │
│ ├─ 📁 Yesterday                                               │
│ │  ├─ ► Machine Learning (25 msgs)                          │
│ │  │   ├─ 🌿 Branch: Deep Learning (15 msgs)              │
│ │  │   └─ 🌿 Branch: Neural Networks (8 msgs)             │
│ │  └─ ► API Design (6 msgs)                                 │
│ │                                                             │
│ └─ 📁 This Week                                               │
│    ├─ ► Database Optimization (18 msgs)                     │
│    └─ ► Frontend Issues (22 msgs)                           │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
│ TREE │ j/k:nav o:open d:delete n:new ESC:exit │ openai/gpt-4o │
```

### Search Mode (`<leader>s`)
```
┌─────────────────────────────────────────────────────────────────┐
│ Search conversations: python_                                  │
│                                                                 │
│ Results (3 found):                                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ► Python Help (12 msgs) - 2 hours ago                     │ │ <- Selected
│ │   "How do I handle exceptions in Python functions?"        │ │
│ └─────────────────────────────────────────────────────────────┘ │
│ ► Python Testing (8 msgs) - Yesterday                         │
│   "What's the best testing framework for Python?"             │
│                                                                 │
│ ► Python Performance (15 msgs) - 3 days ago                   │
│   "How to optimize Python code for large datasets?"           │
│                                                                 │
│                                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│ SEARCH │ type:filter j/k:nav ENTER:open ESC:exit │ 3 results   │
```

### Command Mode (`<leader>:`)
```
┌─────────────────────────────────────────────────────────────────┐
│ system: You are a helpful assistant                            │
│                                                                 │
│ user: What is the capital of France?                          │
│                                                                 │
│ assistant: The capital of France is Paris. It has been the    │
│ political and cultural center of France for centuries and is   │
│ known for landmarks like the Eiffel Tower and the Louvre.     │
│                                                                 │
│ user: Tell me more about the Eiffel Tower                     │
│                                                                 │
│ assistant: The Eiffel Tower was built between 1887 and 1889   │
│ for the 1889 World's Fair...                                  │
│                                                                 │
│ :model gpt-4o_                                                 │
└─────────────────────────────────────────────────────────────────┘
│ CMD │ Available: model, clear, export, help │ anthropic/claude │
```

### Buffer Navigation (`<leader>b`)
```
┌─────────────────────────────────────────────────────────────────┐
│ Switch to conversation: par_                                   │
│                                                                 │
│ Matches:                                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ► Paris Chat (4 msgs) - Active                            │ │ <- Selected
│ └─────────────────────────────────────────────────────────────┘ │
│ ► Parallel Processing (12 msgs) - 2 hours ago                 │
│ ► Parameter Tuning (8 msgs) - Yesterday                       │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│ BUFFER │ type:filter j/k:nav ENTER:switch ESC:exit │ 3 matches │
```

### Message Mode with Branch Point
```
┌─────────────────────────────────────────────────────────────────┐
│ system: You are a helpful assistant                            │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ user: What is the capital of France?                      │ │ <- Selected
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ assistant: The capital of France is Paris...                  │
│ 🌿 → Branch: "More about France" (3 msgs)                     │
│                                                                 │
│ user: Tell me more about the Eiffel Tower                     │
│                                                                 │
│ assistant: The Eiffel Tower was built between 1887 and 1889   │
│ for the 1889 World's Fair...                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
│ MESSAGE │ e:edit(branch) c:copy v:view-branch │ google/gemini  │
```

## Key Visual Elements

- **Selection Boxes**: Clear rectangular borders around selected items
- **Branch Indicators**: 🌿 emoji to show conversation branches  
- **Status Icons**: ► for conversations, 📁 for date folders
- **Progress Bars**: Token usage with visual fill (████████░░)
- **Provider/Model Display**: Format: `{provider}/{model}` in status bar
- **Auto-Generated Names**: Conversations automatically named after first message

## Software Architecture

### Core Abstraction Layers

```
┌─────────────────────────────────────────────────────────────┐
│                 Chat Application                            │
├─────────────────────────────────────────────────────────────┤
│ UI Abstraction Layer (Interface)                           │
│ - UIManager (abstract base)                                │
│ - ModeRenderer (abstract base)                             │
│ - InputHandler (abstract base)                             │
├─────────────────────────────────────────────────────────────┤
│ Business Logic Layer                                       │
│ - ConversationManager                                      │
│ - ModeController                                           │
│ - ConversationNamer                                        │
│ - ConfigManager                                            │
├─────────────────────────────────────────────────────────────┤
│ Data Layer                                                 │
│ - ExtendedMessageStore                                     │
│ - ConversationIndex                                        │
│ - SearchEngine                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

**ConversationManager**: Handles multiple conversations, branching, persistence
**ModeController**: State machine for mode transitions  
**ConversationNamer**: Auto-naming service using LLM
**ExtendedMessageStore**: MessageStore + conversation metadata + unique message IDs
**ConfigManager**: Leader key and keybinding configuration

### Auto-Naming System

```python
class ConversationNamer:
    def __init__(self, naming_model: str = "gpt-4o-mini"):
        self.model = Model(naming_model)
        self.prompt = Prompt("""
        Generate a concise, descriptive title (2-4 words) for a conversation that begins with:
        
        {{first_message}}
        
        Title should be specific but brief. Examples:
        - "Python Error Help"
        - "Paris Travel Tips" 
        - "Database Design"
        """)
```

### Configuration System

```python
class ChatConfig:
    ui_framework: str = "rich"  # or "textual"
    leader_key: str = " "  # space bar default
    naming_model: str = "gpt-4o-mini"
    auto_naming: bool = True
    keybindings: dict = {
        "message_mode": "[",
        "tree_mode": "t", 
        "search_mode": "s",
        "command_mode": ":",
        "buffer_next": "l",
        "buffer_prev": "h",
        "buffer_search": "b"
    }
```

## Core Features

### Conversation Management
- **Branching**: Edit any message to create new conversation branch
- **Auto-Naming**: LLM generates conversation titles after first message
- **Persistence**: Auto-save conversation state and resume capability
- **Unique Message IDs**: UUID/hash-based identifiers for cross-conversation referencing

### Navigation & Search
- **Buffer-Style**: Conversation switching with `<leader>h`/`<leader>l`
- **Global Search**: Exact match (case-insensitive) across conversation titles/metadata
- **Tree Browser**: nvim-tree inspired conversation browser with j/k navigation
- **Context Preservation**: Return to Chat mode at conversation bottom when switching

### Visual Feedback
- **Mode Indicators**: Status bar showing current mode
- **Token Counter**: Visual progress bar showing context window utilization
- **Branch Visualization**: Clear indicators for conversation branches
- **Provider Display**: Show current `{provider}/{model}` in status bar

## Implementation Phases

### Phase 1: Rich Implementation (MVP)
**Timeline**: 4-6 weeks

**Core Components**:
- UI Abstraction Layer with Rich implementation
- Basic mode system (Chat, Message, Tree modes)
- ConversationManager with branching support
- ExtendedMessageStore with unique message IDs
- Auto-naming system with LLM integration
- Configuration system

**Deliverables**:
- Fully functional Rich-based chat UI
- All core modes working
- Conversation branching and navigation
- Auto-naming of conversations
- Configurable leader key and bindings

### Phase 2: Enhanced Features
**Timeline**: 2-3 weeks

**Components**:
- Search mode implementation
- Buffer navigation (`<leader>h`/`<leader>l`/`<leader>b`)
- Command mode with chat commands
- Enhanced status bar with token counting
- Performance optimizations

**Deliverables**:
- Complete feature set as specified
- Search across conversations
- Buffer-style conversation management
- Visual token usage indicators

### Phase 3: Textual Migration Support
**Timeline**: 3-4 weeks

**Components**:
- Textual implementation of UIManager interface
- Enhanced input handling and mouse support
- Framework selection via configuration
- Migration utilities and documentation

**Deliverables**:
- Dual framework support (Rich/Textual)
- Seamless migration path
- Enhanced features available in Textual
- Comprehensive documentation

### Phase 4: Polish & Advanced Features
**Timeline**: 2-3 weeks

**Components**:
- File attachment system design
- Performance optimizations for large conversation histories
- Advanced search capabilities
- Export/import functionality
- Comprehensive testing suite

**Deliverables**:
- Production-ready chat interface
- File attachment support
- Performance benchmarks
- Complete test coverage

## Technical Requirements

- **SSH Compatibility**: All features work over terminal connections
- **Framework Agnostic**: Business logic independent of UI framework
- **Configurable**: Leader key, bindings, and behavior customizable
- **Persistent**: Conversation state maintained across sessions
- **Responsive**: Smooth navigation even with large conversation histories
- **Extensible**: Easy to add new modes and features

This specification provides a complete roadmap for building a powerful, vim-inspired chat interface that can evolve from Rich to Textual while maintaining all functionality and user workflows.
