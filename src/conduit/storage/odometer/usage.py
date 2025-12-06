from pydantic import BaseModel


class Usage(BaseModel):
    """
    Simple data class for usage statistics, standardized across providers.
    """

    input_tokens: int
    output_tokens: int
