from __future__ import annotations
from conduit.core.model.model_async import ModelAsync
from conduit.domain.result.response import GenerationResponse
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.request.request import GenerationRequest
from conduit.domain.message.message import (
    UserMessage,
    TextContent,
    AudioContent,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from conduit.core.model.model_sync import ModelSync
    from conduit.domain.result.response import GenerationResponse


AudioInput = str | bytes | AudioContent | Path


class AudioAsync:
    def __init__(self, parent: ModelAsync):
        self._parent: ModelAsync = parent

    def coerce_audio_input(self, audio: AudioInput) -> AudioContent:
        if isinstance(audio, AudioContent):
            return audio

        # Use Path object for cleaner checks
        path = Path(audio) if isinstance(audio, (str, Path)) else None

        if path and path.is_file():
            return AudioContent.from_file(path)

        if isinstance(audio, bytes):
            import base64

            return AudioContent(
                data=base64.b64encode(audio).decode("utf-8"), format="mp3"
            )

        if isinstance(audio, str):
            # Assume it's already base64 if it's not a file
            return AudioContent(data=audio, format="mp3")

        raise ValueError(f"Cannot coerce {type(audio)} to AudioContent")

    async def generate(
        self,
        prompt_str: str,
        voice: str = "alloy",
        format: str = "mp3",
        speed: float = 1.0,
    ) -> GenerationResponse:
        """
        Models: gpt-4o-mini-tts, gpt-4o-tts
        """
        # Construct params
        client_params = {
            "voice": voice,
            "response_format": format,
            "speed": speed,
        }
        params = GenerationParams(
            output_type="audio",
            model=self._parent.model_name,
            client_params=client_params,
        )
        options = ConduitOptions(project_name="test")

        # Construct messages
        user_message = UserMessage(content=prompt_str)
        messages = [user_message]
        request = GenerationRequest(messages=messages, params=params, options=options)

        response: GenerationResponse = await self._parent.pipe(request)
        return response

    async def analyze(self, prompt_str: str, audio: AudioInput) -> GenerationResponse:
        """
        Models: gpt-4o-audio-preview
        """
        params = GenerationParams(
            output_type="text",
            model=self._parent.model_name,
        )
        options = ConduitOptions(project_name="test")
        # Coerce audio input
        audio_content: AudioContent = self.coerce_audio_input(audio)
        text_content = TextContent(text=prompt_str)
        user_message = UserMessage(content=[text_content, audio_content])
        messages = [user_message]
        request = GenerationRequest(messages=messages, params=params, options=options)
        response: GenerationResponse = await self._parent.pipe(request)
        return response

    async def transcribe(
        self, audio: str | bytes | AudioContent | Path
    ) -> GenerationResponse:
        """
        Models: whisper-1
        """
        params = GenerationParams(
            output_type="transcription",
            model=self._parent.model_name,
        )
        options = ConduitOptions(project_name="test")
        # Coerce audio input
        audio_content: AudioContent = self.coerce_audio_input(audio)
        user_message = UserMessage(content=[audio_content])
        messages = [user_message]
        request = GenerationRequest(messages=messages, params=params, options=options)
        response: GenerationResponse = await self._parent.pipe(request)
        return response


class AudioSync:
    def __init__(self, parent: ModelSync):
        self._parent = parent
        self._impl = AudioAsync(parent._impl)

    def generate(
        self,
        prompt_str: str,
        voice: str = "alloy",
        format: str = "mp3",
        speed: float = 1.0,
    ) -> GenerationResponse:
        return self._parent._run_sync(
            self._impl.generate(
                prompt_str=prompt_str,
                voice=voice,
                format=format,
                speed=speed,
            )
        )

    def analyze(self, prompt_str: str, audio: AudioInput) -> GenerationResponse:
        return self._parent._run_sync(
            self._impl.analyze(prompt_str=prompt_str, audio=audio)
        )

    def transcribe(
        self, audio: str | bytes | AudioContent | Path
    ) -> GenerationResponse:
        return self._parent._run_sync(self._impl.transcribe(audio=audio))


if __name__ == "__main__":
    from conduit.examples.sample_objects import sample_audio_file
    import asyncio

    def test_audio_generate():
        model = ModelAsync("gpt-4o-mini-tts")
        audio = AudioAsync(model)

        async def _main():
            response = await audio.generate(
                prompt_str="Hello, this is a test of OpenAI's text to speech capabilities.",
                voice="alloy",
                format="mp3",
                speed=1.0,
            )
            return response

        result = asyncio.run(_main())
        print(result)

    def test_audio_analyze():
        model = ModelAsync("gpt-4o-audio-preview")
        audio = AudioAsync(model)

        async def _main():
            response = await audio.analyze(
                prompt_str="Please transcribe the following audio.",
                audio=sample_audio_file,
            )
            return response

        result = asyncio.run(_main())
        print(result)

    def test_audio_transcribe():
        model = ModelAsync("whisper-1")
        audio = AudioAsync(model)

        async def main():
            response = await audio.transcribe(
                audio=sample_audio_file,
            )
            return response

        response = asyncio.run(main())
        print(response)
