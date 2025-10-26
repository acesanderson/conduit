"""
PromptLoader: Lazy-loading prompt registry.

This module provides a class to manage Jinja/Jinja2 prompt templates, supporting both
filesystem-based and package-embedded resources.

The loader automatically discovers prompt files with extensions `.jinja` and `.jinja2`,
caches loaded prompts for efficient access, and can optionally scaffold missing files
during development.

Use Cases
----------

**Production (read-only, packaged resources)**
    import my_package
    loader = PromptLoader(base_dir='ignored', package=my_package, subdir='prompts')
    prompt = loader['example_prompt']  # Loads example_prompt.jinja2 from package resources

**Production (auto-discovery from filesystem)**
    loader = PromptLoader('/path/to/prompts')
    prompt = loader['example_prompt']  # Loads example_prompt.jinja2 lazily

**Development (scaffolding mode)**
    loader = PromptLoader('/path/to/prompts', keys=['example_prompt'])
    prompt = loader['example_prompt']  # Creates example_prompt.jinja2 if missing
"""

from pathlib import Path
from importlib import resources
import types


class PromptLoader:
    """
    Lazy-loading prompt registry for jinja/jinja2 template files.
    """

    def __init__(
        self,
        base_dir: str | Path,
        keys=None,
        *,
        package: str | types.ModuleType | None = None,
        subdir: str = "prompts",
    ):  # NEW args
        self._pkg = package is not None  # NEW: flag
        self._pkg_base = None  # NEW: Traversable base when in pkg mode

        if self._pkg:
            # Package resources (read-only)
            self._pkg_base = resources.files(package).joinpath(subdir)  # Traversable
        else:
            # Original filesystem behavior
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)

        if keys is None:
            # Auto-discover jinja files
            self.file_map = {}
            if self._pkg:
                # Package mode discovery (read-only)
                for t in getattr(self._pkg_base, "iterdir", lambda: [])():
                    if getattr(t, "is_file", lambda: False)() and (
                        t.name.endswith(".jinja") or t.name.endswith(".jinja2")
                    ):
                        key = t.name.rsplit(".", 1)[0]
                        self.file_map[key] = t.name
            else:
                # Original filesystem discovery
                for pattern in ["*.jinja", "*.jinja2"]:
                    for file_path in self.base_dir.glob(pattern):
                        key = file_path.stem
                        self.file_map[key] = file_path.name
        else:
            # Generate file_map from keys (key -> key.jinja2)
            self.file_map = {}
            for key in keys:
                filename = f"{key}.jinja2"
                self.file_map[key] = filename

                # Create file if it doesn't exist (filesystem only)
                if not self._pkg:
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

            filename = self.file_map[key]

            if self._pkg:
                # Package mode: hand Prompt.from_file a real path
                traversable = self._pkg_base.joinpath(filename)
                with resources.as_file(traversable) as tmp_path:
                    from conduit.prompt.prompt import Prompt

                    self._cache[key] = Prompt.from_file(Path(tmp_path))
            else:
                # Original filesystem behavior
                file_path = Path(filename)
                if not file_path.is_absolute():
                    file_path = self.base_dir / file_path
                from conduit.prompt.prompt import Prompt

                self._cache[key] = Prompt.from_file(file_path)

        return self._cache[key]

    def __contains__(self, key):
        return key in self.file_map

    def __str__(self):
        base = self._pkg_base if self._pkg else self.base_dir
        return f"PromptLoader(base_dir={base}, keys={self.keys})"
