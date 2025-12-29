from __future__ import annotations
import re
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.message.message import UserMessage
from conduit.domain.message.role import Role
from conduit.core.engine import Engine
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.apps.chat.app import ChatApp


class ChatEngine:
    """
    The asynchronous chat engine.
    """

    # Command Handlers
    async def help(self, app: ChatApp) -> str:
        """
        Displays the help message.
        """
        return "Available commands:\n/help\n/wipe\n/exit\n/history\n/models\n/model\n/set model <model_name>"

    async def history(self, app: ChatApp) -> None:
        """
        Displays the conversation history.
        """
        self.conversation.print_history()

    async def models(self, app: ChatApp) -> str:
        """
        Displays available models.
        """
        from conduit.core.model.models.modelstore import ModelStore

        model_store = ModelStore()
        return model_store._generate_renderable_model_list()

    async def model(self, app: ChatApp) -> str:
        """
        Displays the current model.
        """
        return f"Current model: {self.params.model}"

    async def set_model(self, app: ChatApp, model_name: str | None) -> str:
        """
        Sets the current model.
        """
        if model_name:
            self.params.model = model_name
            return f"Set model to {model_name}"
        return "Error: Please provide a model name (e.g., /set model gpt-4)"

    async def wipe(self, app: ChatApp) -> None:
        """
        Wipes the conversation.
        """
        self.conversation.wipe()

    async def exit(self, app: ChatApp) -> None:
        """
        Exits the application.
        """
        app.is_running = False

    def __init__(
        self,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
    ):
        self.conversation = Conversation()
        self.params = params or GenerationParams.defaults(
            "claude-3-sonnet"
        )  # Default if not provided
        # Options should always be provided at app creation, but provide a default for robustness
        self.options = options or ConduitOptions(project_name="default-chat-app")
        self.commands = {
            "/help": self.help,
            "/wipe": self.wipe,
            "/exit": self.exit,
            "/history": self.history,
            "/models": self.models,
            "/model": self.model,
            "/set model": self.set_model,
        }

    def _parse_command_args(self, command_string: str) -> tuple[str, list[str]]:
        """
        Splits a command string into the command name and its arguments,
        prioritizing multi-word commands defined in self.commands.
        Supports quoted strings for multi-word arguments.
        """
        # Remove leading '/'
        command_string = command_string[1:]

        # Find the longest matching command name
        cmd_name = ""
        remaining_args_string = command_string
        for registered_cmd in sorted(self.commands.keys(), key=len, reverse=True):
            if command_string.startswith(
                registered_cmd[1:]
            ):  # Compare without leading slash
                cmd_name = registered_cmd
                remaining_args_string = command_string[
                    len(registered_cmd[1:]) :
                ].strip()
                break

        if not cmd_name:
            # If no registered command prefix matches, try parsing the first word as command
            parts = re.findall(r'"([^"]*)"|(\S+)', command_string)
            parsed_parts = [quoted or unquoted for quoted, unquoted in parts]
            if parsed_parts:
                cmd_name = "/" + parsed_parts[0]
                remaining_args_string = " ".join(parsed_parts[1:])
            else:
                return "", []  # No command found

        # Parse remaining arguments
        args_parts = re.findall(r'"([^"]*)"|(\S+)', remaining_args_string)
        args = [quoted or unquoted for quoted, unquoted in args_parts]

        return cmd_name, args

    async def handle_query(self, user_input: str) -> str | None:
        """
        Handles a user's query.
        """
        if not user_input.strip():
            return None
        self.conversation.add(UserMessage(content=user_input))
        self.conversation = await Engine.run(
            self.conversation, params=self.params, options=self.options
        )

        if (
            self.conversation.messages
            and self.conversation.messages[-1].role == Role.ASSISTANT
        ):
            return self.conversation.content
        return None

    async def execute_command(self, command_string: str, app: ChatApp) -> str | None:
        """
        Executes a command string.
        """
        cmd_name, args = self._parse_command_args(command_string)

        handler = self.commands.get(cmd_name)
        if handler:
            # Special handling for commands that take arguments
            if cmd_name == "/set model":
                if args:
                    return await handler(app, args[0])
                else:
                    return await handler(
                        app, None
                    )  # Or raise an error, or return a usage string
            else:
                return await handler(app)
        return None
