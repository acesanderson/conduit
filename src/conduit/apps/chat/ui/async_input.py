from prompt_toolkit.shortcuts import PromptSession


class AsyncInput:
    """
    Asynchronous input handler using prompt_toolkit.
    """

    def __init__(self):
        self.session = PromptSession()

    async def get_input(self) -> str:
        """
        Get input from the user asynchronously.
        """
        return await self.session.prompt_async("> ")
