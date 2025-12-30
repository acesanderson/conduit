def parse_skill(file_content: str) -> tuple[dict[str, str], str]:
    """
    Parses a string with YAML frontmatter delimited by '---'.
    Raises ValueError if format is invalid or YAML is malformed.
    """
    import yaml

    # Split by '---' max 2 times
    # Expected: [0] empty, [1] yaml, [2] body
    parts = file_content.split("---", 2)

    # 1. Validation: Must start with --- and have a closing ---
    if len(parts) < 3 or parts[0].strip() != "":
        raise ValueError(
            "Invalid Skill file format: Missing '---' frontmatter delimiters."
        )

    yaml_text = parts[1]
    body_text = parts[2].strip()

    # 2. Validation: YAML must be parseable
    try:
        metadata = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in Skill frontmatter: {e}")

    # 3. Validation: YAML must not be empty or non-dict
    if not isinstance(metadata, dict):
        raise ValueError("Skill frontmatter must contain valid key-value pairs.")

    return metadata, body_text
