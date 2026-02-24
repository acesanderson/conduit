from __future__ import annotations
import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from conduit.config import STATE_DIR
from conduit.core.model.model_sync import ModelSync

DEFAULT_MODEL = "gemini-2.5-flash-image"
IMAGEGEN_DIR = STATE_DIR / "imagegen"
INDEX_FILE = IMAGEGEN_DIR / "index.json"
MAX_HISTORY = 10

IMAGEGEN_MODELS = [
    ("imagegen", "dall-e-3 (alias)"),
    ("dall-e-3", "OpenAI DALL-E 3"),
    ("banana", "gemini-2.5-flash-image (alias)"),
    ("google-imagen", "gemini-2.5-flash-image (alias)"),
    ("gemini-2.5-flash-image", "Google Gemini — default"),
    ("gemini-3-pro-image-preview", "Google Gemini — higher quality"),
    ("imagen-4.0-generate-001", "Google Imagen 4 — Vertex AI credentials required"),
]


def _load_index() -> list[dict]:
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text())


def _save_to_history(prompt: str, model: str, b64_json: str) -> Path:
    IMAGEGEN_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = IMAGEGEN_DIR / f"{ts}.png"

    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(b64_json)))
    image.save(img_path)

    entries = _load_index()
    entries.append(
        {"timestamp": ts, "prompt": prompt, "model": model, "filename": img_path.name}
    )
    INDEX_FILE.write_text(json.dumps(entries[-MAX_HISTORY:], indent=2))

    return img_path


def _display_path(path: Path, label: str | None = None, display_image=True):
    if label:
        # Remove newlines and truncate if too long.
        label = label.replace("\n", " ")
        if len(label) > 80:
            label = label[:77] + "..."
        print(label)
    if display_image:
        subprocess.run(["viu", str(path)])


def _cmd_list_models():
    print("Image generation models:\n")
    for alias, note in IMAGEGEN_MODELS:
        print(f"  {alias:<30} {note}")


def _cmd_last(save: str | None):
    entries = _load_index()
    if not entries:
        print("No images in history.")
        return
    entry = entries[-1]
    img_path = IMAGEGEN_DIR / entry["filename"]
    if save == "-":
        sys.stdout.buffer.write(img_path.read_bytes())
    elif save:
        shutil.copy(img_path, save)
        print(f"Saved to {save}")
    else:
        _display_path(img_path, label=f"{entry['prompt'][:60]}  [{entry['model']}]")


def _cmd_history():
    from rich.console import Console
    from rich.text import Text

    entries = _load_index()
    if not entries:
        print("No images in history.")
        return
    entries = reversed(entries)

    console = Console()
    numbered = list(enumerate(entries, 1))
    for i, entry in numbered:
        img_path = IMAGEGEN_DIR / entry["filename"]
        if not img_path.exists():
            continue
        prompt = entry["prompt"].replace("\n", " ")[:62]
        line = Text()
        line.append(f"{i:>2}. ", style="bold cyan")
        line.append(f"{prompt:<62}", style="white")
        line.append(f"  {entry['model']}", style="dim green")
        console.print(line)


def _cmd_get(n: int, save: str | None):
    entries = _load_index()
    if not entries:
        print("No images in history.")
        return
    if n < 1 or n > len(entries):
        print(f"Invalid index {n}. History has {len(entries)} entries.")
        return
    entry = entries[n - 1]
    img_path = IMAGEGEN_DIR / entry["filename"]
    if save == "-":
        sys.stdout.buffer.write(img_path.read_bytes())
    elif save:
        shutil.copy(img_path, save)
        print(f"Saved to {save}")
    else:
        _display_path(img_path, label=f"{entry['prompt'][:60]}  [{entry['model']}]")


def _read_stdin() -> str | None:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip() or None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate an image from a prompt.",
        add_help=False,
    )
    parser.add_argument("--help", action="help", default=argparse.SUPPRESS)
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument(
        "--model",
        "-m",
        nargs="?",
        const="__list__",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help="Model to use. Bare flag lists available models.",
    )
    parser.add_argument(
        "--last", "-l", action="store_true", help="Display the last generated image."
    )
    parser.add_argument(
        "--history",
        "-h",
        action="store_true",
        help="Display the last 10 generated images.",
    )
    parser.add_argument(
        "--get",
        "-g",
        type=int,
        default=None,
        metavar="N",
        help="Display image number N from history.",
    )
    parser.add_argument(
        "--save",
        "-s",
        nargs="?",
        const="output.png",
        default=None,
        metavar="FILE",
        help="Save output to file (default: output.png).",
    )
    args = parser.parse_args()

    if args.model == "__list__":
        _cmd_list_models()
        return

    if args.last:
        _cmd_last(save=args.save)
        return

    if args.history:
        _cmd_history()
        return

    if args.get is not None:
        _cmd_get(args.get, save=args.save)
        return

    # Resolve prompt: stdin takes priority, positional arg is fallback.
    # If both present, concatenate (positional first — acts as style/prefix).
    stdin = _read_stdin()
    if args.prompt and stdin:
        prompt = args.prompt + " " + stdin
    elif stdin:
        prompt = stdin
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.error("a prompt is required (positional arg or stdin)")

    model = ModelSync(model=args.model)
    response = model.image.generate(prompt_str=prompt)

    if response.message.images:
        b64 = response.message.images[0].b64_json
        _save_to_history(prompt, args.model, b64)
        if args.save == "-":
            sys.stdout.buffer.write(base64.b64decode(b64))
            return
        if args.save:
            response.save_image(Path(args.save))

    response.display()


if __name__ == "__main__":
    main()
