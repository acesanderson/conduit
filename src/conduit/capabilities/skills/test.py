from conduit.core.parser.stream.parsers import XMLStreamParser
from conduit.capabilities.skills.skills import Skills
from conduit.sync import Prompt
from conduit.sync import Model, Verbosity
from conduit.domain.message.messages import Messages
from pathlib import Path

VERBOSITY = Verbosity.PROGRESS
MODEL_NAME = "gpt-oss:latest"
SKILLS = Skills()

SYSTEM_PROMPT_TEMPLATE = (
    Path(__file__).parent / "prompts" / "system_prompt_template.jinja2"
).read_text()
SYSTEM_MESSAGE = """
You are a helpful assistant. Follow these rules:

- Avoid excessive politeness, flattery, or empty affirmations
- Avoid over-enthusiasm or emotionally charged language
- Be direct and factual, focusing on usefulness, clarity, and logic
- Prioritize truth and clarity over appeasing me
- Challenge assumptions or offer corrections anytime you get a chance
- Point out any flaws in the questions or solutions I suggest
- Avoid going off-topic or over-explaining unless I ask for more detail

Finally: I'm trying to stay a critical and sharp analytical thinker. Whenever you see opportunities in our conversations, please push my critical thinking ability.
""".strip()


def construct_system_prompt() -> str:
    """
    Construct a system prompt with embedded skill definitions.

    Loads available skills from the Skills registry and renders them into a system prompt
    template along with the base system message. The template is filled with variables
    including the core system instructions and a list of all available skills.
    """

    skills = Skills()
    skill_list = skills.skills

    prompt = Prompt(SYSTEM_PROMPT_TEMPLATE)
    rendered = prompt.render(
        input_variables={
            "system_prompt": SYSTEM_MESSAGE,
            "skills": skill_list,
        }
    )
    return rendered


def call_tool(tool_call_xml: str) -> str:
    """
    Parse and execute a tool call from XML, returning the tool's skill content.

    Extracts the tool name from an XML function call structure, retrieves the
    corresponding skill from the global SKILLS registry, and returns its content.
    """
    logger.info("Calling tool with XML:")
    logger.debug(f"Tool call XML: {tool_call_xml}")
    import xml.etree.ElementTree as ET

    root = ET.fromstring(tool_call_xml)
    """
    xml schema:
    <function_calls>
        <invoke name="tool_name">
            <parameters>
                <parameter name="param_name">
                    value
                </parameter>
            </parameters>
        </invoke>
    </function_calls>
    """
    # Get tool name, though it should only be "file_read"
    invoke_elem = root.find("invoke")
    tool_name = invoke_elem.attrib["name"]
    logger.info(f"Invoking tool: {tool_name}")
    assert tool_name == "file_read", (
        f"Only file_read tool is supported in this example; got {tool_name}"
    )
    # Get parameter name "path", and its value (which will be the skill file location)
    parameters_elem = invoke_elem.find("parameters")
    param_elem = parameters_elem.find("parameter")
    param_name = param_elem.attrib["name"]
    assert param_name == "path", (
        f"Only 'path' parameter is supported in this example; got {param_name}"
    )
    skill_path = param_elem.text.strip()
    logger.info(f"Reading skill from path: {skill_path}")
    skill = SKILLS.retrieve_skill_by_location(skill_path)
    return skill.content


def stream_query(query: str):
    """
    Execute a streaming query with tool invocation support and context management.

    Constructs a system prompt with available skills, initiates a streaming model query,
    and orchestrates a loop that parses streamed content for tool calls. When a tool call
    is detected, invokes the corresponding skill and incorporates its output back into the
    message history before continuing the conversation. Terminates when the model response
    contains no tool calls.
    """
    logger.info("Constructing system prompt...")
    system_prompt = construct_system_prompt()
    messages = Messages()
    messages.add_new(role="system", content=system_prompt)
    messages.add_new(role="user", content=query)

    while True:
        logger.info("Initiating model query...")
        model = Model(MODEL_NAME)
        stream = model.query(
            query_input=messages.messages,
            stream=True,
            verbose=VERBOSITY,
        )
        logger.info("Parsing stream for tool calls...")
        parser = StreamToolParser(stream=stream)
        pre_tool, tool_call = parser.parse()
        logger.debug(f"Pre-tool content: {pre_tool}")
        logger.debug(f"Tool call content: {tool_call}")
        if tool_call:
            logger.info("Tool call detected, invoking tool...")
            tool_contents = call_tool(tool_call)
            messages.add_new(role="assistant", content=pre_tool + "\n" + tool_call)
            messages.add_new(role="user", content=tool_contents)
            logger.info("Tool call processed, continuing conversation...")
            continue
        else:
            logger.info("No tool call detected, finalizing response...")
            messages.add_new(role="assistant", content=pre_tool)
            break

    return messages


if __name__ == "__main__":
    user_query = "Suggest some companies I should consider applying to."
    messages = stream_query(user_query)
    print(messages)
