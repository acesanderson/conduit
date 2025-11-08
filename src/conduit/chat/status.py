from conduit.chat.status import StatusProvider


class EnhancedInput(InputInterface):
    def __init__(self, console: Console):
        # ...
        self.status_provider: StatusProvider | None = None

    def set_status_provider(self, provider: StatusProvider):
        """Set status provider for toolbar display"""
        self.status_provider = provider

    def get_toolbar_text(self):
        if not self.status_provider:
            return [("class:bottom-toolbar", " Type /help for instructions ")]

        return [
            (
                "class:bottom-toolbar",
                f" Type /help | Model: {self.status_provider.current_model} | "
                f"Messages: {self.status_provider.message_count} | "
                f"Tokens: {self.status_provider.total_tokens} ",
            )
        ]
