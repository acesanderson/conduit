from __future__ import annotations
from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel, Field
from conduit.capabilities.skills.parse import parse_skill
from conduit.capabilities.skills.format import format_skill_content

if TYPE_CHECKING:
    from pathlib import Path


class Skill(BaseModel):
    """
    Represents a skill, context, or prompt to be used by an AI model.
    """

    # The YAML header
    type: Literal["skill", "context", "prompt"] = Field(
        ...,
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
