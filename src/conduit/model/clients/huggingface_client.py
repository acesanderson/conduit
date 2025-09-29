"""
Dummy implementation, purely for getting imagegen to work.
See OLD_huggingface_client.py for the previous ideas for a HuggingFace client.
Keep this simple; HuggingFace is a complex suite, it's possible we'll only have very custom configurations in here, for imagegen, audiogen, audio transcription, image analysis.
"""

from conduit.model.clients.client import Client, Usage
from conduit.request.request import Request
from conduit.logs.logging_config import get_logger


logger = get_logger(__name__)


class HuggingFaceClient(Client):
    """
    This is a base class; we have two subclasses: OpenAIClientSync and OpenAIClientAsync.
    Don't import this.
    """

    def __init__(self):
        self._client = self._initialize_client()

    def _get_api_key(self) -> str:
        pass

    def tokenize(self, model: str, text: str) -> int:
        """
        Return the token count for a string, per model's tokenization function.
        """
        pass


class HuggingFaceClientSync(HuggingFaceClient):
    def _initialize_client(self) -> object:
        pass

    def query(
        self,
        request: Request,
    ) -> tuple:
        """
        Return a tuple of result, usage.
        """
        print("HUGGINGFACE REQUEST DETECTED")
        match request.output_type:
            case "image":
                return self._generate_image(request)
            case "audio":
                from conduit.tests.fixtures.sample_objects import sample_audio_message

                result = (
                    sample_audio_message.audio_content
                )  # base64 data of a generated audio
                usage = Usage(
                    input_tokens=100, output_tokens=100
                )  # dummy usage data, for now
                return result, usage

    def _generate_image(self, request):
        import torch
        from diffusers import FluxPipeline

        # pipe = FluxPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        # )
        #

        pipe = FluxPipeline.from_pretrained(
            "Jlonge4/flux-dev-fp8", torch_dtype=torch.bfloat16
        )

        # pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        pipe.enable_sequential_cpu_offload()

        image = pipe(
            request.messages[-1].content,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        import io
        import base64

        # Convert PIL image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # or "JPEG"
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        result = img_str
        usage = Usage(input_tokens=0, output_tokens=0)
        return result, usage


"""
from pathlib import Path
from transformers import pipeline
import torch

# Import our centralized logger - no configuration needed here!
from Siphon.logs.logging_config import get_logger

# Get logger for this module - will inherit config from retrieve_audio.py
logger = get_logger(__name__)


# Transcript workflow
def transcribe(file_name: str | Path) -> str:
    ""
    Use Whisper to retrieve text content + timestamps.
    ""
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        # model="openai/whisper-large-v3",
        return_timestamps="sentence",
        device=0,
        torch_dtype=torch.float16,
    )
    logger.info(f"Transcribing file: {file_name}")
    result = transcriber(file_name)
    return result
"""
