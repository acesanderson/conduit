from conduit.result.response import Response
from conduit.result.error import ConduitError
from conduit.parser.stream.protocol import SyncStream, AsyncStream

ConduitResult = Response | ConduitError | SyncStream | AsyncStream
