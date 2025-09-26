from typing import Union
from Chain.result.response import Response
from Chain.result.error import ChainError

ChainResult = Union[Response, ChainError]
