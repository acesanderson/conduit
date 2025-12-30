from conduit.capabilities.tools.registry import ToolRegistry
from conduit.capabilities.skills.registry import SkillRegistry
from conduit.capabilities.skills.system import generate_skills_system_prompt
from conduit.sync import Conduit, Verbosity, GenerationParams, ConduitOptions, Prompt

tool_registry = ToolRegistry()
skill_registry = SkillRegistry.from_skills_dir()
tool_registry.enable_skills(skill_registry)

SYSTEM_PROMPT = generate_skills_system_prompt(skill_registry)

# Construct our conduit
prompt = Prompt(
    "I want to leave my job at LinkedIn. Recommend another place I could apply to work at."
)
params = GenerationParams(model="gpt-4o", system=SYSTEM_PROMPT)
options = ConduitOptions(
    project_name="test",
    verbosity=Verbosity.COMPLETE,
    tool_registry=tool_registry,
)
conduit = Conduit(prompt, params, options)

response = conduit.run()
