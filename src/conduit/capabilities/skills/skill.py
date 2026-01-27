from __future__ import annotations
from typing import Literal, TYPE_CHECKING, override
from pydantic import BaseModel, Field
from conduit.capabilities.skills.format import format_skill_content

if TYPE_CHECKING:
    from pathlib import Path


def parse_skill(file_content: str) -> tuple[dict[str, str], str]:
    """
    Parses a string with YAML frontmatter delimited by '---'.
    Raises ValueError if format is invalid or YAML is malformed.
    """
    import yaml

    # Split by '---' max 2 times
    # Expected: [0] empty, [1] yaml, [2] body
    parts = file_content.split("---", 2)

    # 1. Validation: Must start with --- and have a closing ---
    if len(parts) < 3 or parts[0].strip() != "":
        raise ValueError(
            "Invalid Skill file format: Missing '---' frontmatter delimiters."
        )

    yaml_text = parts[1]
    body_text = parts[2].strip()

    # 2. Validation: YAML must be parseable
    try:
        metadata = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in Skill frontmatter: {e}")

    # 3. Validation: YAML must not be empty or non-dict
    if not isinstance(metadata, dict):
        raise ValueError("Skill frontmatter must contain valid key-value pairs.")

    return metadata, body_text


class Skill(BaseModel):
    """
    Represents a skill, context, or prompt to be used by an AI model.
    """

    # The YAML header
    type: Literal["skill", "context", "prompt"] = Field(
        default="skill",
        description="The type of the skill: 'skill' for actions/tools, 'context' for grounding information, 'prompt' for behavior overrides.",
    )
    name: str = Field(..., description="The name of the skill.")
    description: str = Field(..., description="A brief description of the skill.")
    # The actual content -- should be in markdown format
    body: str = Field(..., description="The content/body of the skill.")

    def render(self) -> str:
        """
        Renders the skill content based on its type.
        """
        return format_skill_content(self.type, self.name, self.body)

    @classmethod
    def from_path(cls, path: Path) -> Skill:
        """
        Load a Skill from a markdown file with YAML front matter.
        """

        # Validate path exists and is a file
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Skill file not found: {path}")
        # Read file content
        content = path.read_text(encoding="utf-8")
        # Parse frontmatter and body
        metadata, body = parse_skill(content)
        # Create Skill instance
        return cls(
            type=metadata["type"],
            name=metadata["name"],
            description=metadata["description"],
            body=body,
        )

    @override
    def __str__(self) -> str:
        """Return formatted skill content as string -- i.e. yaml frontmatter + body"""
        return self.render()

    @override
    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, type={self.type!r})"
