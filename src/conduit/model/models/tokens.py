"""
Token estimation module for various LLM providers. NOTE: our odometer utilizes the token count provided in API response, which is more acccurate. This is for estimation only, primary usage is estimating window usage before making an API call.

Example usage:

    from tokens import OpenAITokenizer, AnthropicTokenizer, GeminiTokenizer, HuggingFaceTokenizer

    # OpenAI tokenizer
    openai_tokenizer = OpenAITokenizer()
    openai_tokens = openai_tokenizer.count_tokens("Hello, world!")

    # Anthropic tokenizer
    anthropic_tokenizer = AnthropicTokenizer()
    anthropic_tokens = anthropic_tokenizer.count_tokens("Hello, world!")

    # Gemini tokenizer
    gemini_tokenizer = GeminiTokenizer()
    gemini_tokens = gemini_tokenizer.count_tokens("Hello, world!")

    # Hugging Face tokenizer for a specific model
    hf_tokenizer = HuggingFaceTokenizer(model_hf_name="gpt2")
    hf_tokens = hf_tokenizer.count_tokens("Hello, world!")
"""

import abc
import os


class BaseTokenizer(abc.ABC):
    """
    Abstract base class for a tokenizer.
    Defines the common interface for encoding text and counting tokens.
    """

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes a text string into a list of token IDs.
        """
        pass

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in a given text string.
        """
        return len(self.encode(text))


class OpenAITokenizer(BaseTokenizer):
    """
    Tokenizer for OpenAI models using tiktoken for exact token counting.

    Provides accurate token counting for OpenAI models by lazy-loading the tiktoken library
    on first access. Uses the cl100k_base encoding by default, which is standard for most
    OpenAI models (GPT-3.5, GPT-4, etc.).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazy-loads the tiktoken tokenizer."""
        if self._tokenizer is None:
            try:
                import tiktoken

                print(f"[Lazy Load] Loading tiktoken for '{self.encoding_name}'...")
                self._tokenizer = tiktoken.get_encoding(self.encoding_name)
            except ImportError:
                raise ImportError(
                    "tiktoken is not installed. Please run: pip install tiktoken"
                )
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)


class AnthropicTokenizer(OpenAITokenizer):
    """
    Approximation tokenizer for Anthropic models using tiktoken encoding.

    Provides token counting for Anthropic Claude models by leveraging OpenAI's cl100k_base
    encoding. Note that actual server-side token counts from Anthropic often exceed reported
    values by 15-30%, making this suitable for estimates but not precise accounting.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        print("---")
        print("WARNING: AnthropicTokenizer is an APPROXIMATION.")
        print("The actual token count may be 15-30% higher than reported.")
        print("---")
        super().__init__(encoding_name)


class GeminiTokenizer(BaseTokenizer):
    """
    Tokenizer for Google Gemini models using Hugging Face transformers.

    Provides near-exact token counting for Gemini by leveraging the google/gemma-7b
    tokenizer from Hugging Face. Lazy-loads the transformers library on first use
    and supports HuggingFace gated model access via environment variable tokens.
    """

    def __init__(self, model_hf_name: str = "google/gemma-7b"):
        self.model_hf_name = model_hf_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """
        Lazy-loads the transformers tokenizer.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                # --- START MODIFICATION ---
                # Read token from env var
                token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
                if not token:
                    print(
                        "WARNING: HUGGINGFACEHUB_API_TOKEN or HF_TOKEN env var not set. "
                        "Gated models will fail."
                    )

                print(f"[Lazy Load] Loading transformers for '{self.model_hf_name}'...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_hf_name,
                    token=token,  # Pass the token here
                )
                # --- END MODIFICATION ---

            except ImportError:
                raise ImportError(
                    "transformers is not installed. "
                    "Please run: pip install transformers sentencepiece"
                )
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)


class HuggingFaceTokenizer(BaseTokenizer):
    """
    Generic tokenizer for any Hugging Face model using transformers library.

    Supports local models via Ollama (e.g., Llama 3, Mistral) by lazy-loading
    the transformers AutoTokenizer. Handles authentication via HuggingFace API tokens
    from environment variables for gated model access.
    """

    def __init__(self, model_hf_name: str):
        if not model_hf_name:
            raise ValueError("model_hf_name must be provided for HuggingFaceTokenizer")
        self.model_hf_name = model_hf_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        """
        Lazy-loads the transformers tokenizer.
        """
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                # --- START MODIFICATION ---
                # Read token from env var
                token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
                if not token:
                    print(
                        "WARNING: HUGGINGFACEHUB_API_TOKEN or HF_TOKEN env var not set. "
                        "Gated models will fail."
                    )

                print(f"[Lazy Load] Loading transformers for '{self.model_hf_name}'...")
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_hf_name,
                    token=token,  # Pass the token here
                )
                # --- END MODIFICATION ---

            except ImportError:
                raise ImportError(
                    "transformers is not installed. "
                    "Please run: pip install transformers"
                )
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
