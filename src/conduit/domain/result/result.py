from conduit.domain.result.response import Response
from conduit.domain.result.error import ConduitError
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream

ConduitResult = Response | ConduitError | SyncStream | AsyncStream
