"""
Hand-rolled implementation of Claude Skills for Conduit project.
End goal:
- a custom query_function that can be swapped into Conduit workflows (Conduit, ConduitChat) which hides all the back-and-forth.
- being able to plug in an arbitrary set of SKILL.md files (according to established directory structure) and have them automatically available as skills.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, computed_field, field_validator
from pathlib import Path
from xdg_base_dirs import (
    xdg_config_home,
)

DEFAULT_SKILLS_DIR = Path(xdg_config_home()) / "conduit" / "skills"


class Skill(BaseModel):
    name: str = Field(..., description="The name of the skill.")
    description: str = Field(
        ..., description="A brief description of the skill for the LLM."
    )
    location: str = Field(
        ...,
        description="The absolute file path where the skill can be accessed. Should be absolute + exists when placed in a Path object).",
    )
    content: str = Field(
        ...,
        description="The full markdown content of the skill file, excluding the YAML front matter.",
    )

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        return cls._validate_location(v)

    @classmethod
    def _validate_location(cls, location: str) -> str:
        p = Path(location)
        if not p.suffix == ".md":
            raise ValueError("location must point to a .md file.")
        if not p.is_absolute():
            raise ValueError("location must be an absolute file path.")
        if not p.exists():
            raise ValueError(f"location path does not exist: {location}")
        return location

    @classmethod
    def from_path(cls, skill_path: Path) -> Skill:
        """
        Create a Skill instance from a SKILL.md file located at skill_path.
        The SKILL.md file is expected to have the following format:
        """
        # Validate path
        path_string = cls._validate_location(str(skill_path.absolute()))

        # Read file contents
        # Assemble attributes
        name, description, content = cls._parse_yaml(
            path_string
        )  # Parse the YAML section
        location = path_string

        return cls(
            name=name, description=description, location=location, content=content
        )

    @classmethod
    def _parse_yaml(cls, path_string: str) -> tuple[str, str, str]:
        # Read the skill file
        skill_path = Path(path_string)
        with skill_path.open("r", encoding="utf-8") as f:
            skill_file_contents = f.read()

        # Extract the yaml section
        import re

        yaml_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.search(yaml_pattern, skill_file_contents, re.DOTALL | re.MULTILINE)
        yaml_content = match.group(1) if match else None

        skill_file_contents = re.sub(
            yaml_pattern, "", skill_file_contents, flags=re.DOTALL | re.MULTILINE
        )

        # Parse the yaml section
        if not yaml_content:
            raise ValueError(f"No YAML section found in skill file.")
        import yaml

        yaml_values: dict[str, str] = yaml.safe_load(yaml_content)
        name = yaml_values.get("name")
        if not name:
            raise ValueError("Skill YAML must contain a 'name' field.")
        description = yaml_values.get("description")
        if not description:
            raise ValueError("Skill YAML must contain a 'description' field.")
        return name, description, skill_file_contents


class Skills(BaseModel):
    dir_path: str = Field(
        default=str(DEFAULT_SKILLS_DIR),
        description="The directory path where skill files are located.",
    )

    @field_validator("dir_path")
    @classmethod
    def validate_dir_path(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("dir_path must be an absolute file path.")
        if not p.exists():
            raise ValueError(f"dir_path does not exist: {v}")
        if not p.is_dir():
            raise ValueError(f"dir_path is not a directory: {v}")
        return v

    @computed_field
    @property
    def skills(self) -> list[Skill]:
        skill_dir = Path(self.dir_path)
        # Traverse the directory structure to find all SKILL.md files
        skill_files = skill_dir.glob("*/SKILL.md")
        skills: list[Skill] = []
        for skill_file in skill_files:
            skill = Skill.from_path(skill_file)
            skills.append(skill)
        return skills

    def retrieve_skill_by_name(self, name: str) -> Skill | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None


if __name__ == "__main__":
    skills = Skills()
