from conduit.domain.result.response import Response
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream

ConduitResult = Response | SyncStream | AsyncStream
