from rich.console import Console

console = Console()


def display(message: str) -> None:
    """
    Displays a message to the console.
    """
    console.print(message)
