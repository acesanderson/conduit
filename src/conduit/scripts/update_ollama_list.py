"""
This script updates the list of Ollama models in the Conduit module.
Use this when switching environments, ssh tunnels, or when new models are added.
Need to figure out where to automatically implement this in my Conduit package to avoid manual updates but also preserve lazy loading.
"""

from conduit.model.clients.ollama_client import OllamaClientSync
from conduit.model.models.modelstore import ModelStore
from rich import console
import logging

logger = logging.getLogger(__name__)
console = console.Console(width=80)


def main():
    console.print("[green]Updating Ollama Models...[/green]")
    client = OllamaClientSync()
    client.update_ollama_models()
    console.print(
        f"[green]Model list updated: [/green][yellow]{ModelStore.models()['ollama']}[/yellow]"
    )


if __name__ == "__main__":
    main()
