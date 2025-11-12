# Skills architecture

## XML Syntax Summary

**(1) Describing a skill (in system prompt):**
```xml
<skill>
<name>docx</name>
<description>Comprehensive document creation, editing, and analysis...</description>
<location>/mnt/skills/public/docx/SKILL.md</location>
</skill>
```

**(2) Exposing skills library (in system prompt):**
```xml
<available_skills>
<skill>
<name>docx</name>
<description>...</description>
<location>/mnt/skills/public/docx/SKILL.md</location>
</skill>

<skill>
<name>pdf</name>
<description>...</description>
<location>/mnt/skills/public/pdf/SKILL.md</location>
</skill>
<!-- ... more skills ... -->
</available_skills>
```

**(3) LLM requesting skill load:**



**(4) How skill appears once loaded:**

Pure markdown blob with YAML frontmatter and line numbers. The YAML frontmatter (lines 1-5 in the example) contains:

```yaml
---
name: docx
description: "Comprehensive document creation, editing, and analysis..."
license: Proprietary. LICENSE.txt has complete terms
---
```

**Required fields per official docs:**
- `name`: Max 64 chars, lowercase letters/numbers/hyphens only, no reserved words ("anthropic", "claude")
- `description`: Max 1024 chars, describes what the skill does AND when to use it

The content is wrapped only in `<function_results>` tags with no XML wrapping of the markdown itself. Line numbers are added by the system. The markdown is injected directly into context as plain text.

## Key Architecture Detail from Docs

The docs clarify the **three-level progressive loading model**:

**Level 1: Metadata (always loaded, ~100 tokens/skill)**
- Just the YAML frontmatter `name` and `description`
- Present in system prompt at startup
- Zero cost to have many skills installed

**Level 2: Instructions (loaded when triggered, <5k tokens)**
- Main SKILL.md body content
- Read via bash when skill matches task
- Only then enters context window

**Level 3: Resources (loaded as needed, effectively unlimited)**
- Additional markdown files (FORMS.md, REFERENCE.md)
- Executable scripts (fill_form.py)
- Reference materials (schemas, templates)
- Accessed via bash only when referenced
- Scripts execute without code entering context—only output consumes tokens

