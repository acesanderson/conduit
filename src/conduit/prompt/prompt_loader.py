"""
PromptLoader: Lazy-loading prompt registry.
This module provides a class to manage prompts, allowing for lazy loading of prompt files from a specified directory.

It supports auto-discovery of prompt files with extensions .jinja and .jinja2, and allows for custom keys to be specified.
It also ensures that prompt files are created if they do not exist, and caches loaded prompts for efficient access.

Production use case (auto-discovery)
    loader = PromptLoader('/path/to/prompts')
    prompt = loader['example_prompt']  # Loads example_prompt.jinja2 lazily

Development use case (scaffolding the directory + files):
    loader = PromptLoader('/path/to/prompts', keys=['example_prompt'])
    prompt = loader['example_prompt']  # Loads example_prompt.jinja2 lazily, creating the file if it doesn't exist
"""

from pathlib import Path


class PromptLoader:
    """Lazy-loading prompt registry for jinja/jinja2 template files."""

    def __init__(self, base_dir, keys=None):
        self.base_dir = Path(base_dir)

        # Ensure base_dir exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if keys is None:
            # Auto-discover jinja files
            self.file_map = {}
            for pattern in ["*.jinja", "*.jinja2"]:
                for file_path in self.base_dir.glob(pattern):
                    key = file_path.stem  # filename without extension
                    self.file_map[key] = file_path.name
        else:
            # Generate file_map from keys (key -> key.jinja2)
            self.file_map = {}
            for key in keys:
                filename = f"{key}.jinja2"
                self.file_map[key] = filename

                # Create file if it doesn't exist
                full_path = self.base_dir / filename
                if not full_path.exists():
                    full_path.touch()

        self._cache = {}

    @property
    def keys(self):
        return list(self.file_map.keys())

    def __getitem__(self, key):
        if key not in self._cache:
            if key not in self.file_map:
                raise KeyError(f"Unknown prompt: {key}")

            # Handle both strings and Path objects
            file_path = Path(self.file_map[key])
            if not file_path.is_absolute():
                file_path = self.base_dir / file_path

            from conduit.prompt.prompt import Prompt

            self._cache[key] = Prompt.from_file(file_path)

        return self._cache[key]

    def __contains__(self, key):
        return key in self.file_map
