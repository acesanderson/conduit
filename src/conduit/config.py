from pathlib import Path
from dataclasses import dataclass
from xdg_base_dirs import (
    xdg_config_home,
)

CONFIG_DIR = Path(xdg_config_home()) / "conduit"


@dataclass
class Settings:
    system_prompt: str
    preferred_model: str


def load_settings() -> Settings:
    # Defaults (lowest priority)
    config = {
        "system_prompt": "You are a helpful assistant.",
        "preferred_model": "gpt",
    }

    # System message
    system_prompt_path = CONFIG_DIR / "system_message.jinja2"
    assert system_prompt_path.exists(), f"Missing config file: {system_prompt_path}"
    system_prompt = system_prompt_path.read_text()

    # Preferred model
    preferred_model_path = CONFIG_DIR / "preferred_model"
    assert preferred_model_path.exists(), f"Missing config file: {preferred_model_path}"
    preferred_model = preferred_model_path.read_text().strip()

    config.update(
        {
            "system_prompt": system_prompt,
            "preferred_model": preferred_model,
        }
    )

    return Settings(**config)


# Singleton
settings = load_settings()
