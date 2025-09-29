from PIL import Image
import io, base64
from pathlib import Path


def convert_to_png_base64(base64_data: str) -> str:
    """
    Convert base64 image (any format) to PNG base64.
    
    Args:
        base64_data: Base64 string of image in any format
        
    Returns:
        Base64 string of PNG image
    """
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_data)
    
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Convert to appropriate mode for PNG
        if img.mode in ('P', 'LA'):
            img = img.convert('RGBA')
        elif img.mode in ('L', '1'):
            img = img.convert('RGB')
        
        # Save as PNG to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        
        # Convert back to base64
        png_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return png_base64


def resize_png_base64(png_base64: str, target_size: int = 1024) -> str:
    """
    Resize PNG base64 to optimal size for LLM vision models.
    
    Args:
        png_base64: Base64 string of PNG image
        target_size: Maximum dimension in pixels (default 1024 for OCR)
        
    Returns:
        Base64 string of resized PNG
    """
    # Decode base64 to bytes
    image_bytes = base64.b64decode(png_base64)
    
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Only resize if larger than target
        if max(img.size) > target_size:
            # Calculate new size maintaining aspect ratio
            ratio = target_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save resized PNG to buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        
        # Convert back to base64
        resized_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return resized_base64


def convert_image(base64_data: str, target_size: int = 1024) -> str:
    """
    Convert any base64 image to optimized PNG base64 for LLMs.
    
    Pipeline:
    1. Convert to PNG (if not already)
    2. Resize to LLM-friendly dimensions
    
    Args:
        base64_data: Base64 string of image in any format
        target_size: Maximum dimension in pixels (default 1024 for OCR)
        
    Returns:
        Base64 string of optimized PNG
    """
    # Convert to PNG format
    png_base64 = convert_to_png_base64(base64_data)
    
    # Resize to target dimensions
    final_base64 = resize_png_base64(png_base64, int(target_size))
    
    return final_base64


def convert_image_file(file_path: str | Path, target_size: int = 1024) -> str:
    """
    Convert image file to optimized PNG base64 for LLMs.
    
    Args:
        file_path: Path to image file (png, jpg, jpeg, gif, webp only)
        target_size: Maximum dimension in pixels (default 1024 for OCR)
        
    Returns:
        Base64 string of optimized PNG
        
    Raises:
        ValueError: If file format is not supported

    Usage:
        optimized_base64 = convert_image(original_base64)  # From base64
        optimized_base64 = convert_image_file("photo.jpg")  # From file path
    """
    # Validate file format
    if isinstance(file_path, str):
        file_path = Path(file_path)
    supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    
    if file_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported: {', '.join(supported_formats)}")
    
    # Read file to base64
    with open(file_path, "rb") as f:
        original_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Process through the pipeline
    return convert_image(original_base64, target_size)

