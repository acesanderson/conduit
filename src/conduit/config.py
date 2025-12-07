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
from conduit.domain.request.generation_params import GenerationParams
from rich.console import Console
from xdg_base_dirs import (
    xdg_config_home,
    xdg_state_home,
    xdg_data_home,
)
import os

# Directories
CONFIG_DIR = Path(xdg_config_home()) / "conduit"
STATE_DIR = Path(xdg_state_home()) / "conduit"
DATA_DIR = Path(xdg_data_home()) / "conduit"

# File paths
SYSTEM_PROMPT_PATH = CONFIG_DIR / "system_message.jinja2"
SETTINGS_TOML_PATH = CONFIG_DIR / "settings.toml"
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
    default_params: GenerationParams


def load_settings() -> Settings:
    # Defaults (lowest priority)
    config: dict[str, object] = {
        "system_prompt": "You are a helpful assistant.",
        "preferred_model": "gpt3",
        "default_verbosity": Verbosity.PROGRESS,
        "default_console": Console(),
        "server_models": [],
        "paths": {},
        "default_params": GenerationParams(model="gpt3"),
    }

    # Config files (medium priority)
    assert SYSTEM_PROMPT_PATH.exists(), f"Missing config file: {system_prompt_path}"
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    assert SETTINGS_TOML_PATH.exists(), f"Missing config file: {SETTINGS_TOML_PATH}"
    with SETTINGS_TOML_PATH.open("rb") as f:
        toml_config = tomllib.load(f)
    toml_dict = toml_config.get("settings")
    preferred_model = toml_dict.get("preferred_model", config["preferred_model"])
    verbosity_str = toml_dict.get("verbosity", config["default_verbosity"].name)
    verbosity = Verbosity[verbosity_str.upper()]
    assert SERVER_MODELS_PATH.exists(), (
        f"Missing server models file: {SERVER_MODELS_PATH}"
    )
    with SERVER_MODELS_PATH.open("r") as f:
        server_models_dict: dict[str, list[str]] = json.load(f)
    server_models: list[str] = server_models_dict.get("ollama", [])

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
    verbosity = (
        Verbosity[os.getenv("CONDUIT_VERBOSITY").upper()]
        if os.getenv("CONDUIT_VERBOSITY")
        else verbosity
    )

    paths = {
        "CONFIG_DIR": CONFIG_DIR,
        "STATE_DIR": STATE_DIR,
        "SYSTEM_PROMPT_PATH": SYSTEM_PROMPT_PATH,
        "SETTINGS_TOML_PATH": SETTINGS_TOML_PATH,
        "SERVER_MODELS_PATH": SERVER_MODELS_PATH,
        "OLLAMA_CONTEXT_SIZES_PATH": OLLAMA_CONTEXT_SIZES_PATH,
        "OLLAMA_MODELS_PATH": OLLAMA_MODELS_PATH,
        "DEFAULT_HISTORY_FILE": DEFAULT_HISTORY_FILE,
        "DEFAULT_LOG_FILE": DEFAULT_LOG_FILE,
    }

    # Default params
    default_params = GenerationParams(
        model=preferred_model,
    )

    config.update(
        {
            "system_prompt": system_prompt,
            "preferred_model": preferred_model,
            "default_verbosity": verbosity,
            "server_models": server_models,
            "paths": paths,
            "default_params": default_params,
        }
    )

    return Settings(**config)


# Singleton
settings = load_settings()
