from typing import Protocol, runtime_checkable
from conduit.storage.cache.protocol import ConduitCache
from conduit.storage.odometer.odometer_registry import OdometerRegistry
from conduit.utils.progress.verbosity import Verbosity


@runtime_checkable
class Instrumentable(Protocol):
    """
    Our middleware decorators attach instrumentation capabilities to classes that implement this protocol.
    """

    cache_engine: ConduitCache | None
    verbosity: Verbosity
    odometer_registry: OdometerRegistry | None
