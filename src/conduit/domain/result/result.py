from conduit.domain.result.response import GenerationResponse
from conduit.core.parser.stream.protocol import SyncStream, AsyncStream

GenerationResult = GenerationResponse | SyncStream | AsyncStream
