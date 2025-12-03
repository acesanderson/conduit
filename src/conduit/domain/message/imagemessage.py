"""
Factory functions.
"""


def from_image_file(image_file: str | Path, text_content: str, role: str = "user"):
    raise NotImplementedError("Not implemented yet.")  # type: ignore
    """
    Create ImageMessage from image file.

    Args:
        image_file: Path to image file (any supported format)
        text_content: Text prompt/question about the image
        role: Message role (default: "user")

    Returns:
        ImageMessage with processed image data

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
    """
    image_path = Path(image_file)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file {image_path} does not exist")

    # Validate file format
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    # Convert image to optimized PNG base64
    try:
        image_content = convert_image_file(image_path)
    except Exception as e:
        raise ValueError(f"Failed to process image file {image_path}: {e}")

    # Validate the conversion worked
    if not is_base64_simple(image_content):
        raise ValueError("Image conversion produced invalid base64 data")

    return cls(
        role=role,
        content=[image_content, text_content],
        text_content=text_content,
        image_content=image_content,
        mime_type="image/png",  # Always PNG after convert_image_file()
    )


def from_base64(
    image_content: str,
    text_content: str,
    mime_type: str = "image/png",
    role: str = "user",
):
    raise NotImplementedError("Not implemented yet.")  # type: ignore
