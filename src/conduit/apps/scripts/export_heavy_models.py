"""
Export the list of heavy Ollama models as YAML to stdout.

A model is "heavy" if heavy=True in its ModelSpec (>30B parameters or >24GB VRAM).

Usage:
    export_heavy_models            # writes YAML to stdout
    export_heavy_models > heavy.yaml

Output format:
    heavy_models:
      - qwq:latest
      - deepseek-r1:70b
"""

from __future__ import annotations

import sys

import yaml

from conduit.storage.modelspec_repository import ModelSpecRepository


def main() -> None:
    repo = ModelSpecRepository()
    all_specs = repo.get_all()
    heavy = sorted(
        spec.model for spec in all_specs if getattr(spec, "heavy", False)
    )
    yaml.dump({"heavy_models": heavy}, sys.stdout, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    main()
