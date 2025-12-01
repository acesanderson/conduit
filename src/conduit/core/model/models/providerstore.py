"""
Repository class for managing provider specifications.
This module provides functions to create, read, update, and delete provider specifications.
"""

from conduit.model.models.providerspec import ProviderSpec
from pathlib import Path

dir_path = Path(__file__).parent
providers_file = dir_path / "providers.jsonl"


class ProviderStore:
    @classmethod
    def create_provider(cls, provider: ProviderSpec) -> None:
        """
        Create a new provider specification and save it to the JSONL file.

        Args:
            provider (ProviderSpec): The provider specification to create.
        """
        with open(providers_file, "a") as f:
            f.write(provider.model_dump_json() + "\n")

    @classmethod
    def get_all_providers(cls) -> list[ProviderSpec]:
        """
        Read all provider specifications from the JSONL file.

        Returns:
            list[ProviderSpec]: A list of provider specifications.
        """
        providers = []
        with open(providers_file, "r") as f:
            for line in f:
                providers.append(ProviderSpec.model_validate_json(line.strip()))
        return providers

    @classmethod
    def update_provider(cls, provider: ProviderSpec) -> None:
        """
        Update an existing provider specification in the JSONL file.

        Args:
            provider (ProviderSpec): The provider specification to update.
        """
        providers = cls.get_all_providers()
        updated = False
        with open(providers_file, "w") as f:
            for p in providers:
                if p.provider == provider.provider:
                    f.write(provider.model_dump_json() + "\n")
                    updated = True
                else:
                    f.write(p.model_dump_json() + "\n")
        if not updated:
            raise ValueError(f"Provider {provider.provider} not found for update.")

    @classmethod
    def delete_provider(cls, provider_name: str) -> None:
        """
        Delete a provider specification from the JSONL file.

        Args:
            provider_name (str): The name of the provider to delete.
        """
        providers = cls.get_all_providers()
        with open(providers_file, "w") as f:
            deleted = False
            for p in providers:
                if p.provider != provider_name:
                    f.write(p.model_dump_json() + "\n")
                else:
                    deleted = True
            if not deleted:
                raise ValueError(f"Provider {provider_name} not found for deletion.")

    @classmethod
    def get_provider(cls, provider_name: str) -> ProviderSpec:
        """
        Get a specific provider specification by name.

        Args:
            provider_name (str): The name of the provider to retrieve.

        Returns:
            ProviderSpec: The provider specification if found.

        Raises:
            ValueError: If the provider is not found.
        """
        providers = cls.get_all_providers()
        for p in providers:
            if p.provider == provider_name:
                return p
        raise ValueError(f"Provider {provider_name} not found.")
