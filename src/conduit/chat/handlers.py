"""
Mixin for our ConduitChat class.
Similar mixins can be added if you want custom commands.
"""


class Handlers:
    def command_exit(self):
        """
        Exit the chat.
        """
        self.console.print("Goodbye!", style="green")
        exit()

    def command_help(self):
        """
        Display the help message.
        """
        commands = sorted(self.get_commands())
        help_message = "Commands:\n"
        for command in commands:
            command_name = command.replace("command_", "").replace("_", " ")
            command_func = getattr(self, command)
            try:
                command_docs = command_func.__doc__.strip()
            except AttributeError:
                print(f"Command {command_name} is missing a docstring.")
                sys.exit()
            help_message += (
                f"/[purple]{command_name}[/purple]: [green]{command_docs}[/green]\n"
            )
        self.console.print(help_message)

    def command_wipe(self):
        """
        Clear the message history.
        """
        if self.message_store:
            self.message_store.clear()
            self.console.print("Message history cleared.", style="green")
        else:
            self.console.print("No message store available.", style="red")

    def command_clear(self):
        """
        Clear the screen.
        """
        self.console.clear()

    def command_show_log_level(self):
        raise NotImplementedError("Log level not implemented yet.")

    def command_set_log_level(self, param: str):
        raise NotImplementedError("Log level not implemented yet.")

    def command_show_history(self):
        """
        Display the chat history.
        """
        if self.message_store:
            self.message_store.view_history()

    def command_show_models(self):
        """
        Display available models.
        """
        from conduit.model.models.modelstore import ModelStore

        ms = ModelStore()
        ms.display()

    def command_show_model(self):
        """
        Display the current model.
        """
        self.console.print(f"Current model: {self.model.model}", style="green")

    def command_set_model(self, param: str):
        """
        Set the current model.
        """
        try:
            self.model = Model(param)
            self.console.print(f"Set model to {param}", style="green")
        except ValueError:
            self.console.print(f"Invalid model: {param}", style="red")

    def command_paste_image(self):
        """
        Use this when you have an image in clipboard that you want to submit as context for LLM.
        This gets saved as self.clipboard_image.
        """
        from Conduit.message.imagemessage import ImageMessage
        import os

        if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
            self.console.print("Image paste not available over SSH.", style="red")
            return

        import warnings
        from PIL import ImageGrab
        import base64, io

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress PIL warnings
            image = ImageGrab.grabclipboard()

        if image:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            # Save for next query
            self.console.print("Image captured! What is your query?", style="green")
            user_input = self.console.input(
                "[bold green]Query about image[/bold green]: "
            )
            # Build our ImageMessage
            text_content = user_input
            image_content = img_base64
            mime_type = "image/png"
            role = "user"
            imagemessage = ImageMessage(
                role=role,
                text_content=text_content,
                image_content=image_content,
                mime_type=mime_type,
            )
            self.message_store.append(imagemessage)
            response = self.query_model([self.message_store.last()])
            self.console.print(response)

        else:
            self.console.print("No image detected.", style="red")

    def command_wipe_image(self):
        """
        Delete image from memory.
        """
        if self.clipboard_image:
            self.clipboard_image = None
            self.console.print("Image deleted.", style="green")
        else:
            self.console.print("No image to delete.", style="red")
