"""
Configuration hierarchy:
1. Explicit parameters to functions / methods
2. Environment variables (highest priority)
3. System prompt file / settings TOML file
4. Defaults (lowest priority)
"""

from __future__ import annotations
from pathlib import Path
import tomllib
import json
from dataclasses import dataclass
from conduit.utils.progress.verbosity import Verbosity
from rich.console import Console
from xdg_base_dirs import (
    xdg_config_home,
    xdg_state_home,
    xdg_data_home,
)
import os
from collections.abc import Callable
from typing import TYPE_CHECKING
from importlib.metadata import version

if TYPE_CHECKING:
    from conduit.storage.odometer.odometer_registry import OdometerRegistry
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.storage.cache.protocol import ConduitCache
    from conduit.storage.repository.protocol import AsyncSessionRepository
    from psycopg2 import connection

# Global odometer instance
_odometer_registry: OdometerRegistry | None = None

# Directories
CONFIG_DIR = Path(xdg_config_home()) / "conduit"
STATE_DIR = Path(xdg_state_home()) / "conduit"
DATA_DIR = Path(xdg_data_home()) / "conduit"

# Version
try:
    __version__ = version("conduit")
except Exception:
    __version__ = "unknown"

# File paths
SYSTEM_PROMPT_PATH = CONFIG_DIR / "system_message.jinja2"
SETTINGS_TOML_PATH = CONFIG_DIR / "settings.toml"
SKILLS_DIR = CONFIG_DIR / "skills"
OLLAMA_CONTEXT_SIZES_PATH = CONFIG_DIR / "ollama_context_sizes.json"
SERVER_MODELS_PATH = STATE_DIR / "server_models.json"
OLLAMA_MODELS_PATH = STATE_DIR / "ollama_models.json"
DEFAULT_HISTORY_FILE = DATA_DIR / "conduit" / "history.json"
DEFAULT_LOG_FILE = DATA_DIR / "conduit" / "conduit.log"


@dataclass
class Settings:
    system_prompt: str
    preferred_model: str
    default_verbosity: Verbosity
    default_console: Console
    server_models: list[str]
    paths: dict[str, Path]
    default_project_name: str
    version: str
    # Lazy loaders
    odometer_registry: Callable[[], OdometerRegistry]
    default_params: Callable[[], GenerationParams]
    default_cache: Callable[[str], ConduitCache]
    default_repository: Callable[[str], ConversationRepository]
    default_conduit_options: Callable[[str], ConduitOptions]


def load_settings() -> Settings:
    # Defaults (lowest priority)
    config: dict[str, object] = {
        "system_prompt": "You are a helpful assistant.",
        "preferred_model": "gpt3",
        "default_verbosity": Verbosity.PROGRESS,
        "default_console": Console(stderr=True),
        "server_models": [],
        "paths": {},
        "default_project_name": "conduit",
        "version": __version__,
    }

    # Config files (medium priority)
    # Ensure critical config files exist or provide defaults if missing logic is preferred
    # (Leaving assertions as is per original file, assuming scaffolding exists)
    if SYSTEM_PROMPT_PATH.exists():
        system_prompt = SYSTEM_PROMPT_PATH.read_text()
    else:
        system_prompt = config["system_prompt"]  # Fallback

    if SETTINGS_TOML_PATH.exists():
        with SETTINGS_TOML_PATH.open("rb") as f:
            toml_config = tomllib.load(f)
        toml_dict = toml_config.get("settings", {})
        preferred_model = toml_dict.get("preferred_model", config["preferred_model"])
        default_project_name = toml_dict.get(
            "default_project_name", config["default_project_name"]
        )
        verbosity_str = toml_dict.get("verbosity", config["default_verbosity"].name)
        verbosity = Verbosity[verbosity_str.upper()]
    else:
        preferred_model = config["preferred_model"]
        default_project_name = config["default_project_name"]
        verbosity = config["default_verbosity"]

    server_models: list[str] = []
    if SERVER_MODELS_PATH.exists():
        with SERVER_MODELS_PATH.open("r") as f:
            try:
                server_models_dict = json.load(f)
                server_models = server_models_dict.get("ollama", [])
            except json.JSONDecodeError:
                pass

    # Environment variables (highest priority)
    system_prompt = (
        os.getenv("CONDUIT_SYSTEM_PROMPT", system_prompt)
        if os.getenv("CONDUIT_SYSTEM_PROMPT")
        else system_prompt
    )
    preferred_model = (
        os.getenv("CONDUIT_PREFERRED_MODEL", preferred_model)
        if os.getenv("CONDUIT_PREFERRED_MODEL")
        else preferred_model
    )
    default_verbosity = (
        Verbosity[os.getenv("CONDUIT_VERBOSITY").upper()]
        if os.getenv("CONDUIT_VERBOSITY")
        else verbosity
    )

    paths = {
        "CONFIG_DIR": CONFIG_DIR,
        "STATE_DIR": STATE_DIR,
        "DATA_DIR": DATA_DIR,
        "SKILLS_DIR": SKILLS_DIR,
        "SYSTEM_PROMPT_PATH": SYSTEM_PROMPT_PATH,
        "SETTINGS_TOML_PATH": SETTINGS_TOML_PATH,
        "SERVER_MODELS_PATH": SERVER_MODELS_PATH,
        "OLLAMA_CONTEXT_SIZES_PATH": OLLAMA_CONTEXT_SIZES_PATH,
        "OLLAMA_MODELS_PATH": OLLAMA_MODELS_PATH,
        "DEFAULT_HISTORY_FILE": DEFAULT_HISTORY_FILE,
        "DEFAULT_LOG_FILE": DEFAULT_LOG_FILE,
    }

    # Default params
    def default_params() -> GenerationParams:
        from conduit.domain.request.generation_params import GenerationParams

        default_params = GenerationParams(
            model=preferred_model,
        )
        return default_params

    # CHANGED: Sync function, returns Lazy Cache
    def default_cache(project_name: str = default_project_name) -> ConduitCache:
        """
        Lazy loader for the default AsyncPostgresCache instance.
        """
        from conduit.storage.cache.postgres_cache_async import AsyncPostgresCache

        # We just pass parameters; the pool is created when accessed inside the loop
        return AsyncPostgresCache(project_name=project_name, db_name="conduit")

    def default_repository(
        project_name: str = default_project_name,
    ) -> ConversationRepository:
        # ... [repository code remains the same] ...
        from conduit.storage.repository.postgres_repository import get_async_repository

        return get_async_repository(project_name)

    def default_conduit_options(name: str = default_project_name) -> ConduitOptions:
        """
        Assemble default ConduitOptions from settings.
        Default is NO cache, no repository.
        """
        from conduit.domain.config.conduit_options import ConduitOptions

        return ConduitOptions(
            verbosity=default_verbosity,
            project_name=name,
            cache=None,
            repository=None,
            console=config["default_console"],
        )

    def get_odometer_registry() -> OdometerRegistry:
        from conduit.storage.odometer.odometer_registry import OdometerRegistry

        global _odometer_registry
        if _odometer_registry is None:
            _odometer_registry = OdometerRegistry()
        return _odometer_registry

    config.update(
        {
            "default_cache": default_cache,
            "default_conduit_options": default_conduit_options,
            "default_params": default_params,
            "default_repository": default_repository,
            "default_verbosity": default_verbosity,
            "odometer_registry": get_odometer_registry,
            "paths": paths,
            "preferred_model": preferred_model,
            "server_models": server_models,
            "system_prompt": system_prompt,
            # Lazy loaders
        }
    )

    return Settings(**config)


# Singleton
settings = load_settings()
