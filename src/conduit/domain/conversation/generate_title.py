from conduit.core.conduit.conduit_sync import ConduitSync
from conduit.core.prompt.prompt import Prompt
from conduit.utils.progress.verbosity import Verbosity
from conduit.domain.conversation.conversation import Conversation


prompt_str = """
Generate a concise title for this conversation.

<system_context>{{ system_message }}</system_context>

<user_request>{{ user_message }}</user_request>

Output a noun phrase (4-7 words) that captures the user's intent. No articles at the start. No punctuation. No quotes. No preamble. Just the title.
""".strip()

prompt = Prompt(prompt_str)
conduit = ConduitSync.create(prompt=prompt, model="gpt3", verbose=Verbosity.SILENT)


def generate_title(conversation: Conversation) -> str:
    if conversation.roles != "SU":
        raise ValueError(
            "Conversation must have roles 'SU' (System, User) to generate title."
        )

    system_string = str(conversation.system.content) if conversation.system else ""
    user_string = conversation.content
    title = _generate_title(system_string, user_string)
    return title


def _generate_title(system_message: str, user_message: str) -> str:
    response = conduit(
        system_message=system_message,
        user_message=user_message,
    )
    title = response.content.strip()
    return title
