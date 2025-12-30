from typing import Literal


def format_skill_content(
    skill_type: Literal["skill", "context", "prompt"], name: str, body: str
) -> str:
    if skill_type == "prompt":
        # Overrides the persona/behavior.
        # Injected as a System Instruction or high-priority User Message.
        return f"""
<procedure_active>
SYSTEM INSTRUCTION: PROCEDURE LOADED ({name})
You must now adopt the following reasoning protocol. 
Abandon previous default behaviors and strictly follow these steps:

{body}
</procedure_active>
"""

    elif skill_type == "context":
        # Grounds the model in facts/constraints.
        # Injected as a User Message or Context Block.
        return f"""
<context_active>
SYSTEM NOTE: CONTEXT LOADED ({name})
Use the following information as the ground truth for your next responses.
Do not treat this as an instruction to act, but as constraints to think within.

{body}
</context_active>
"""

    else:
        # type == "skill" (Action/Tool)
        # Just shows the documentation/manual for the tool.
        return f"""
<skill>
Documentation for {name}:

{body}
</skill>
"""
