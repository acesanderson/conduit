from __future__ import annotations
from pathlib import Path
from conduit.config import settings
from conduit.capabilities.skills.skill import Skill


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    @property
    def skills_dir(self) -> Path:
        from conduit.config import settings

        return settings.paths["SKILLS_DIR"]

    @property
    def skills(self) -> list[Skill]:
        return list(self._skills.values())

    @property
    def system(self) -> str:
        from conduit.capabilities.skills.system import generate_skills_system_prompt

        return generate_skills_system_prompt(self)

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Skill:
        return self._skills[name]

    def list_skills(self) -> list[str]:
        return list(self._skills.keys())

    @classmethod
    def from_skills_dir(
        cls, skills_dir: Path = settings.paths["SKILLS_DIR"]
    ) -> SkillRegistry:
        """
        Creates a SkillRegistry by loading all skill files from the specified directory.
        """
        registry = cls()
        # The skills directory has a directory for each skill; the SKILL.md file inside contains the skill definition.
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    skill = Skill.from_path(skill_file)
                    registry.register(skill)
        return registry
