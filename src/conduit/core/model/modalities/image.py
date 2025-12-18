from __future__ import annotations
from conduit.core.model.model_async import ModelAsync
from conduit.domain.result.response import GenerationResponse
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import GenerationRequest
from conduit.domain.message.message import (
    UserMessage,
    TextContent,
    ImageContent,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from conduit.core.model.model_sync import ModelSync


ImageInput = str | bytes | ImageContent | Path


class ImageAsync:
    def __init__(self, parent: ModelAsync):
        self._parent: ModelAsync = parent

    def coerce_image_input(self, image: ImageInput) -> ImageContent:
        """
        Standardizes various input types into an ImageContent DTO.
        Early-exit strategy avoids redundant reads and complex nesting.
        """
        # 1. Direct Pass-through
        if isinstance(image, ImageContent):
            return image

        # 2. Path/File Detection
        # Path objects or strings that look like local file paths
        is_path = isinstance(image, (Path, str))
        if is_path:
            p = Path(image)
            # Check if it's a file before doing anything expensive
            if (
                p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
                and p.exists()
            ):
                # Use the existing DTO logic to handle the read/encode in one go
                return ImageContent.from_file(p)

        # 3. Raw Bytes
        if isinstance(image, bytes):
            import base64

            # Default to PNG if format is unknown; Client/Adapter can override later
            b64_str = base64.b64encode(image).decode("utf-8")
            return ImageContent(url=f"data:image/png;base64,{b64_str}")

        # 4. Fallback for Strings (URLs or already encoded Base64)
        if isinstance(image, str):
            # If it already looks like a Data URI or URL, pass it through
            if image.startswith(("http", "data:image")):
                return ImageContent(url=image)
            # Otherwise, assume it's a raw base64 string
            return ImageContent(url=f"data:image/png;base64,{image}")

        raise ValueError(f"Unsupported image input type: {type(image)}")

    async def generate(
        self,
        prompt_str: str,
        size: str = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        n: int = 1,
        response_format: Literal["url", "b64_json"] = "b64_json",
    ):
        """
        Models: dall-e-2, dall-e-3
        """
        from conduit.core.clients.openai.image_params import OpenAIImageParams

        # Construct params
        client_params = OpenAIImageParams(
            size=size,
            quality=quality,
            style=style,
            n=n,
            response_format=response_format,
        )
        params = GenerationParams(
            output_type="image",
            model=self._parent.model_name,
            client_params=client_params.model_dump(),
        )
        options = ConduitOptions(project_name="test")

        # Construct messages
        user_message = UserMessage(content=prompt_str)
        messages = [user_message]
        request = GenerationRequest(messages=messages, params=params, options=options)

        response: GenerationResponse = await self._parent.pipe(request)
        return response

    async def analyze(self, prompt_str: str, image: ImageInput) -> GenerationResponse:
        """
        Models: gpt-4.1
        """
        params = GenerationParams(
            output_type="text",
            model=self._parent.model_name,
        )
        options = ConduitOptions(project_name="test")
        # Coerce image input
        image_content: ImageContent = self.coerce_image_input(image)
        text_content = TextContent(text=prompt_str)
        user_message = UserMessage(content=[text_content, image_content])
        messages = [user_message]
        request = GenerationRequest(messages=messages, params=params, options=options)
        response: GenerationResponse = await self._parent.pipe(request)
        return response


class ImageSync:
    def __init__(self, parent: ModelSync):
        self._parent = parent
        self._impl = ImageAsync(parent._impl)

    def generate(
        self,
        prompt_str: str,
        size: str = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        n: int = 1,
        response_format: Literal["url", "b64_json"] = "b64_json",
    ) -> GenerationResponse:
        return self._parent._run_sync(
            self._impl.generate(
                prompt_str=prompt_str,
                size=size,
                quality=quality,
                style=style,
                n=n,
                response_format=response_format,
            )
        )

    def analyze(self, prompt_str: str, image: ImageInput) -> GenerationResponse:
        return self._parent._run_sync(
            self._impl.analyze(prompt_str=prompt_str, image=image)
        )


if __name__ == "__main__":
    from conduit.examples.sample_objects import sample_image_file
    import asyncio

    def test_image_generate():
        model = ModelAsync("dall-e-3")
        image = ImageAsync(model)

        async def _main():
            response = await image.generate(
                prompt_str="A los angeles beach babe smoking a joint, digital art",
                size="1024x1024",
                quality="hd",
                style="vivid",
                n=1,
                response_format="b64_json",
            )
            return response

        response = asyncio.run(_main())
        return response

    response = test_image_generate()
    response.display()

    def test_image_analyze():
        model = ModelAsync("gpt-4.1")
        image = ImageAsync(model)

        async def _main():
            response = await image.analyze(
                prompt_str="Describe the content of the image in detail.",
                image=sample_image_file,
            )
            return response

        result = asyncio.run(_main())
        print(result)
