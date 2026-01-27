"""
Skill-augmented synchronous Conduit instance for executing LLM queries with access to a library of registered skills and tools. This module wraps ConduitSync with skill and tool registry initialization, enabling models to dynamically invoke domain-specific capabilities during generation. The SkillConduit class automatically loads available skills from the configured directory, registers them as executable tools, and injects them into the generation context via system prompts and tool bindings.

This is a convenience factory for ad-hoc skill-enabled queries, useful for interactive or scripted use cases where you want skill access without manually wiring the registries. For production systems, the skill/tool registration pattern shown here can be adapted to your own orchestration layer.

Usage:
```python
prompt = Prompt("I'm thinking of quitting my job at LinkedIn, give me some advice.")
conduit = SkillConduit(prompt=prompt, model="claude")
response = conduit.run()  # Runs with skills enabled; Claude can invoke registered tools
```
"""

from conduit.capabilities.skills.registry import SkillRegistry
from conduit.capabilities.tools.registry import ToolRegistry
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.utils.progress.verbosity import Verbosity
from conduit.sync import Conduit
from conduit.core.prompt.prompt import Prompt


class SkillConduit(Conduit):
    def __init__(self, prompt: Prompt, model: str = "haiku"):
        skill_registry = SkillRegistry.from_skills_dir()
        tool_registry = ToolRegistry()
        tool_registry.enable_skills(skill_registry)
        options = ConduitOptions(
            project_name="conduit",
            tool_registry=tool_registry,
            verbosity=Verbosity.COMPLETE,
        )
        system_message = skill_registry.system
        params = GenerationParams(model=model, system=system_message)
        super().__init__(prompt=prompt, options=options, params=params)


# prompt = Prompt("Tell me my preferred python development practices.")
prompt = Prompt("I'm thinking of quitting my job at LinkedIn, give me some advice.")
conduit = SkillConduit(prompt=prompt, model="claude")
response = conduit.run()
