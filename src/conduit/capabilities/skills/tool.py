from __future__ import annotations
from typing import Annotated, TYPE_CHECKING
from conduit.capabilities.tools.tool import Tool, ObjectSchema, Property

if TYPE_CHECKING:
    from conduit.capabilities.skills.registry import SkillRegistry
    from conduit.capabilities.tools.registry import ToolRegistry


async def enable_skill(
    skill_name: Annotated[str, "The name of the skill to enable"],
    _skill_registry: SkillRegistry,
    _tool_registry: ToolRegistry,
) -> str:
    """
    Enables a specified skill. This loads the skill body into the conversation, and may update your available tools.
    """
    try:
        skill = _skill_registry.get_skill(skill_name)
        return skill.render()

    except KeyError:
        return f"Skill '{skill_name}' not found."


# We need to roll a special Tool instance
## First our one Property (skill_name, NOT the registry args)
skill_name_property = Property(
    type="string",
    description="The name of the skill to enable",
)
## Now our ObjectSchema
object_schema = ObjectSchema(
    properties={
        "skill_name": skill_name_property,
    },
    required=["skill_name"],
)
## Finally our Tool
enable_skill_tool = Tool(
    name="enable_skill",
    description=enable_skill.__doc__.strip(),
    input_schema=object_schema,
    func=enable_skill,
)

__all__ = ["enable_skill_tool"]
