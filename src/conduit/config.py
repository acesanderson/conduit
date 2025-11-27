"""
Configuration hierarchy:
1. Explicit parameters to functions / methods
2. Environment variables (highest priority)
3. System prompt file / settings TOML file
4. Defaults (lowest priority)
"""

from pathlib import Path
import tomllib
from dataclasses import dataclass
from conduit.progress.verbosity import Verbosity
from xdg_base_dirs import (
    xdg_config_home,
)
import os

CONFIG_DIR = Path(xdg_config_home()) / "conduit"
SYSTEM_PROMPT_PATH = CONFIG_DIR / "system_message.jinja2"
SETTINGS_TOML_PATH = CONFIG_DIR / "settings.toml"


@dataclass
class Settings:
    system_prompt: str
    preferred_model: str
    verbosity: Verbosity


def load_settings() -> Settings:
    # Defaults (lowest priority)
    config = {
        "system_prompt": "You are a helpful assistant.",
        "preferred_model": "gpt3",
        "verbosity": Verbosity.PROGRESS,
    }

    # Config files (medium priority)
    assert SYSTEM_PROMPT_PATH.exists(), f"Missing config file: {system_prompt_path}"
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    assert SETTINGS_TOML_PATH.exists(), f"Missing config file: {SETTINGS_TOML_PATH}"
    with SETTINGS_TOML_PATH.open("rb") as f:
        toml_config = tomllib.load(f)
    toml_dict = toml_config.get("settings")
    preferred_model = toml_dict.get("preferred_model", config["preferred_model"])
    verbosity_str = toml_dict.get("verbosity", config["verbosity"].name)
    verbosity = Verbosity[verbosity_str.upper()]

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

    config.update(
        {
            "system_prompt": system_prompt,
            "preferred_model": preferred_model,
            "verbosity": verbosity,
        }
    )

    return Settings(**config)


# Singleton
settings = load_settings()
