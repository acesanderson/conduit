from __future__ import annotations
from conduit.config import settings
from pathlib import Path
from jinja2 import Template
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.capabilities.skills.registry import SkillRegistry

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt_template.jinja2"
PROMPT_STR = SYSTEM_PROMPT_PATH.read_text()


def generate_skills_system_prompt(
    registry: SkillRegistry, system_prompt: str = settings.system_prompt
) -> str:
    """Generate the system prompt with the available skills.

    Args:
        registry (SkillRegistry): The skill registry.
        system_prompt (str, optional): The base system prompt. Defaults to settings.system_prompt.

    Returns:
        str: The generated system prompt.
    """
    template = Template(PROMPT_STR)
    skills = registry.all_skills()
    return template.render(system_prompt=system_prompt, skills=skills)
