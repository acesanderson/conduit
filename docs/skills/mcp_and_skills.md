## **Design Doc: Skills & MCP Integration Architecture**

### 1\. Overview & Core Philosophy

This document outlines an architecture for integrating a "Skills" framework with the Model Context Protocol (MCP) (Tools, Resources, Prompts).

The primary challenge in complex agentic systems is **context window saturation** and **LLM confusion**. A "flat" architecture that provides 500 tools and 100 resources in a single prompt is expensive, inefficient, and leads to poor tool-selection accuracy.

The core philosophy of this architecture is **Progressive Disclosure** managed by a **multi-level context system**.

  * **Level 1 (Router):** A lightweight, low-cost "Router" context. Its *only* job is to analyze the user's intent and select the correct "Skill."
  * **Level 2+ (Specialist):** A "Specialist" context that is lazy-loaded when a Skill is invoked. This context *only* contains the persona, tools, and resources necessary for that specific task.

**Skills** are the central mechanism for this. A Skill is *not* a single tool; it is a **"Context Package"** that defines a specialist agent's entire environment.

### 2\. Architectural Components & Hierarchy

| Component | Definition | Role & Scope |
| :--- | :--- | :--- |
| **Base "Router" Context (L1)** | The default LLM state. | **Scope:** Global. **Job:** Routes user requests to the correct Skill or Universal Tool. Must be lightweight and cheap. |
| **Skill (L2+ Context)** | A **package** of persona, tools, & resources. | **Scope:** Specialized. A Skill is "loaded" by the `view_skill` tool, which triggers the application to *re-prompt* the LLM with a new, specialized context. |
| **MCP Tools (Verbs)** | Executable functions. | **Universal Tools:** Available in L1 (e.g., `view_skill`, `fetch_url`). <br> **Scoped Tools:** Bundled in a Skill package; only available in L2. |
| **MCP Resources (Nouns)** | Passive, file-like data. | **Global Resources:** Available in L1 (e.g., global style guide). <br> **Scoped Resources:** Bundled in a Skill package; only available in L2. |
| **MCP Prompts** | Reusable instruction sets. | **Scope:** Can be bundled in a Skill to provide reusable logic, personas, or instruction partials. |

### 3\. The "Skill Package" Definition

To enable true progressive disclosure, a "Skill" is defined as a directory package, not a single file. This allows each Skill to be a self-contained, modular unit that explicitly declares all its dependencies.

The `skills.py` loader is responsible for parsing this structure.

#### Directory Structure

```
/skills/
  /trino-debug/
    SKILL.md
    tools.json
    resources.json
  /content-creation/
    SKILL.md
    tools.json
    resources.json
```

#### File Definitions

  * **`SKILL.md`:** This file has two parts:

    1.  **YAML Head:** Contains the Level 1 metadata (`name`, `description`). This is read by the `Skills` loader to populate the *Base Router Prompt* (L1).
    2.  **Markdown Body:** Contains the **Level 2 System Prompt (Persona)**. This is the text (e.g., "You are a Trino query optimization expert...") that becomes the *new* system prompt when the skill is activated.

  * **`tools.json`:** A JSON file containing a list of **scoped MCP tool definitions** (in JSON Schema or similar format) that are *only* available when this Skill is active.

  * **`resources.json`:** A JSON file containing a list of **scoped MCP resource definitions** (e.g., `{"uri": "resource://trino/optimization_rules.md", ...}`) that are *only* exposed when this Skill is active.

### 4\. Context Engineering & Loading Workflow

The application ("Conduit") manages the transitions between context levels.

#### Level 1: The Base "Router" Context

The application starts the LLM in this lightweight context. The prompt is built dynamically using `base_router_prompt.jinja2`.

  * **Job:** The LLM's *only* goal is to call a Universal Tool or, more commonly, the `view_skill` tool.
  * **Prompt Engineering Note:** XML-style tags (`<tag>`) are used, as this is the format Anthropic's models are trained to prioritize for structuring prompts.

**`base_router_prompt.jinja2`**

```jinja
You are a top-level orchestration assistant. Your job is to analyze the user's
request and select the single best tool or skill to achieve the user's goal.

Do not try to answer complex requests. Your ONLY job is to route the
request to the correct tool or skill.

{# --- AVAILABLE SKILLS (Loaded from SKILL.md YAML) --- #}
<available_skills>
{% for skill in skills %}
    <skill>
        <name>{{ skill.name }}</name>
        <description>{{ skill.description }}</description>
    </skill>
{% endfor %}
</available_skills>

{# --- MCP TOOLS (Universal) --- #}
<mcp_tools>
    <tool>
        <name>view_skill</name>
        <description>Critical. Loads the full instructions (Level 2) for a skill from <available_skills>.</description>
        <parameters>{"name": "skill_name", "type": "string"}</parameters>
    </tool>
    {% for tool in universal_tools %}
    <tool>
        <name>{{ tool.name }}</name>
        <description>{{ tool.description }}</description>
    </tool>
    {% endfor %}
</mcp_tools>

{# --- MCP RESOURCES (Global) --- #}
<mcp_resources>
{% for resource in global_resources %}
    <resource>
        <uri>{{ resource.uri }}</uri>
        <description>{{ resource.description }}</description>
    </resource>
{% endfor %}
</mcp_resources>

## HOW TO WORK
1.  **Analyze Request:** Read the user's request.
2.  **Match to Skill:** If the request is complex and matches a description in
    `<available_skills>`, your ONLY action is to call the `view_skill`
    tool with the skill's `name`.
3.  **Match to Tool:** If the request is simple and matches a description in
    `<mcp_tools>`, call that tool directly.
```

#### The Context Bridge: `view_skill`

The `view_skill` tool is the lynchpin. When the L1 "Router" LLM calls `view_skill("trino-debug")`:

1.  Your application **intercepts** this call.
2.  It finds the Skill Package at `/skills/trino-debug/`.
3.  It **builds a new Level 2 context** by loading:
      * `SKILL.md` (for the persona text)
      * `tools.json` (for the scoped tools)
      * `resources.json` (for the scoped resources)
4.  It also "inherits" the *Universal Tools* and *Global Resources* for continuity.
5.  It **re-prompts the LLM** with this newly generated, complete Level 2 prompt.

#### Level 2: The "Specialist" Skill Context

This new context is generated from a *different* template, `level_2_skill_prompt.jinja2`.

**`level_2_skill_prompt.jinja2`**

```jinja
{# --- 1. PERSONA (from SKILL.md body) --- #}
{{ skill_persona_text }}

{# --- 2. MCP TOOLS (Scoped + Universal) --- #}
<mcp_tools>
    {# --- Scoped tools for this skill --- #}
    {% for tool in scoped_tools %}
    <tool>
        <name>{{ tool.name }}</name>
        <description>{{ tool.description }}</description>
    </tool>
    {% endfor %}

    {# --- Inherited universal tools --- #}
    {% for tool in universal_tools %}
    <tool>
        <name>{{ tool.name }}</name>
        <description>{{ tool.description }}</description>
    </tool>
    {% endfor %}
</mcp_tools>

{# --- 3. MCP RESOURCES (Scoped + Global) --- #}
<mcp_resources>
    {# --- Scoped resources for this skill --- #}
    {% for resource in scoped_resources %}
    <resource>
        <uri>{{ resource.uri }}</uri>
        <description>{{ resource.description }}</description>
    </resource>
    {% endfor %}

    {# --- Inherited global resources --- #}
    {% for resource in global_resources %}
    <resource>
        <uri>{{ resource.uri }}</uri>
        <description>{{ resource.description }}</description>
    </resource>
    {% endfor %}
</mcp_resources>

## HOW TO WORK
You are now in a specialized context. Use the tools and resources above
to complete the user's task.
```

-----

### 5\. Example Workflow Trace

**Task:** User needs to debug a slow Trino query.

1.  **User:** "My Trino query is running slow: `SELECT ...`"
2.  **LLM (L1 - Router Context):**
      * **Prompt:** Sees the `base_router_prompt`.
      * **Analysis:** Scans `<available_skills>` and finds `<name>trino-debug</name><description>Debug and optimize...</description>`.
      * **Action:** Calls the `view_skill` tool.
      * **Tool Call:** `<function_calls><invoke name="view_skill"><parameters><skill_name>trino-debug</skill_name></parameters></invoke></function_calls>`
3.  **Application (Conduit):**
      * Intercepts call.
      * Finds package `/skills/trino-debug/`.
      * Loads `SKILL.md` (Persona: "You are a Trino query expert...").
      * Loads `tools.json` (Scoped Tool: `get_explain_plan`).
      * Loads `resources.json` (Scoped Resource: `resource://trino/optimization_rules.md`).
      * Renders `level_2_skill_prompt.jinja2` with all this data.
4.  **LLM (L2 - Specialist Context):**
      * **New Prompt:** The application re-prompts the model with the new, fully-rendered Level 2 context.
      * **Analysis:** The LLM is now a "Trino expert." Its prompt instructs it to use `get_explain_plan`. It also sees the `trino_optimization_rules.md` resource is available.
      * **Action:** It calls the scoped tool `get_explain_plan`.
      * **Tool Call:** `<function_calls><invoke name="get_explain_plan"><parameters><query>SELECT ...</query></parameters></invoke></function_calls>`
5.  **Application:** Executes the tool, returns the query plan.
6.  **LLM (L2):**
      * **Analysis:** Has the plan. Now it needs the optimization rules.
      * **Action:** Calls the *inherited* universal tool `read_resource`.
      * **Tool Call:** `<function_cols><invoke name="read_resource"><parameters><uri>resource://trino/optimization_rules.md</uri></parameters></invoke></function_calls>`
7.  **Application:** Returns the content of the rules.
8.  **LLM (L2):** Now has the plan *and* the rules. It synthesizes a final, expert-level answer for the user.

