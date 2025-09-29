from typing import Union
from conduit.result.response import Response
from conduit.result.error import ConduitError

ConduitResult = Union[Response, ConduitError]
