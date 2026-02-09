from typing import Annotated
from conduit.domain.exceptions.exceptions import ToolError


async def glob_files(
    pattern: Annotated[
        str, "The glob pattern to match filenames (e.g., 'src/**/*.py' or '*.md')."
    ],
    root_dir: Annotated[str, "The root directory to start the search from."] = ".",
) -> dict[str, str]:
    """
    Finds file paths matching a glob pattern, respecting .gitignore. Useful for discovering project structure.
    """
    import asyncio
    from pathlib import Path
    import pathspec

    def _run_glob():
        p = Path(root_dir)

        # Load .gitignore if present
        spec = None
        gitignore_path = p / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)

        # Decide between recursive or flat glob
        files = (
            p.rglob(pattern.replace("**/", "")) if "**" in pattern else p.glob(pattern)
        )

        matches = []
        for file_path in files:
            if not file_path.is_file():
                continue

            # Get relative path
            rel_path = file_path.relative_to(p)

            # Check .gitignore
            if spec and spec.match_file(str(rel_path)):
                continue

            # Skip hidden files
            if file_path.name.startswith("."):
                continue

            matches.append(str(rel_path))

        if not matches:
            return "No files found matching that pattern."
        return "\n".join(matches)

    output = await asyncio.to_thread(_run_glob)
    return {"result": output}


async def grep_files(
    regex: Annotated[str, "The regular expression to search for in file contents."],
    glob_pattern: Annotated[
        str, "A glob pattern to limit the search scope (e.g., '**/*.py')."
    ] = "**/*",
    root_dir: Annotated[str, "The root directory to search within."] = ".",
) -> dict[str, str]:
    """
    Searches for a regex pattern inside file contents, respecting .gitignore.
    """
    import asyncio
    import re
    from pathlib import Path
    import pathspec

    def _run_grep():
        compiled_re = re.compile(regex)
        p = Path(root_dir)

        # Load .gitignore if present
        spec = None
        gitignore_path = p / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)

        results = []
        candidates = p.rglob(glob_pattern.replace("**/", ""))

        for file_path in candidates:
            if not file_path.is_file():
                continue

            rel_path = file_path.relative_to(p)

            # Check .gitignore
            if spec and spec.match_file(str(rel_path)):
                continue

            # Skip large/binary files
            if file_path.stat().st_size > 1_000_000:
                continue

            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if compiled_re.search(line):
                            results.append(f"{rel_path}:{i}: {line.strip()}")
            except Exception:
                continue

        if not results:
            return "No matches found."
        return "\n".join(results)

    output = await asyncio.to_thread(_run_grep)
    return {"result": output}


async def file_read(
    path: Annotated[str, "The path to the file to read."],
    start_line: Annotated[
        int, "The line number to start reading from (1-indexed)."
    ] = 1,
    end_line: Annotated[
        int | None, "The line number to stop reading at. If None, reads to end."
    ] = None,
    root_dir: Annotated[str, "The root directory to enforce security sandbox."] = ".",
) -> dict[str, str]:
    """
    Read a file's contents with line numbers, supporting pagination.
    """
    from pathlib import Path

    p = Path(path).expanduser().resolve()
    root = Path(root_dir).expanduser().resolve()

    # Security: Jail Check
    if not p.is_relative_to(root):
        raise ToolError(f"Access denied: {path} is outside the allowed directory.")

    if not p.is_file():
        raise ToolError(f"Path is not a file: {path}")

    if not p.exists():
        raise ToolError(f"The specified file does not exist: {path}")

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        raise ToolError("File is binary or not UTF-8 encoded.")

    total_lines = len(lines)

    # Handle end_line logic
    actual_end = end_line if end_line is not None else total_lines
    start_idx = max(0, start_line - 1)
    end_idx = min(total_lines, actual_end)

    if start_idx >= total_lines:
        return {
            "result": f"File has {total_lines} lines. Start line {start_line} is out of bounds."
        }

    # Add line numbers (1-indexed)
    numbered_lines = [
        f"{i + 1} | {line}"
        for i, line in enumerate(lines[start_idx:end_idx], start=start_idx)
    ]

    content = "\n".join(numbered_lines)

    return {
        "path": str(p),
        "total_lines": str(total_lines),
        "viewing_range": f"{start_idx + 1}-{end_idx}",
        "file_contents": content,
    }


async def ls(
    path: Annotated[str, "The directory path to list."] = ".",
    root_dir: Annotated[str, "The root directory to enforce security sandbox."] = ".",
) -> dict[str, str]:
    """
    Lists files in a directory with useful metadata (size, type).
    """
    from pathlib import Path

    target = Path(path).expanduser().resolve()
    root = Path(root_dir).expanduser().resolve()

    if not target.is_relative_to(root):
        raise ToolError(f"Access denied: {path} is outside the allowed directory.")

    if not target.is_dir():
        raise ToolError(f"Not a directory: {path}")

    results = []
    # Header
    results.append(f"{'Mode':<5} {'Size':<10} {'Name'}")
    results.append("-" * 30)

    # Sort directories first, then files
    try:
        entries = sorted(
            target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError:
        raise ToolError(f"Permission denied: Cannot list directory {path}")

    for item in entries:
        # Skip hidden files usually, or keep them if relevant to your dev work
        if item.name.startswith("."):
            continue

        try:
            stats = item.stat()
            size_str = f"{stats.st_size:,} B" if item.is_file() else "-"
            type_marker = "DIR" if item.is_dir() else "FILE"
            results.append(f"{type_marker:<5} {size_str:<10} {item.name}")
        except PermissionError:
            results.append(f"????  {'?':<10} {item.name} (Access Denied)")

    return {"result": "\n".join(results)}


__all__ = ["file_read", "glob_files", "grep_files", "ls"]
