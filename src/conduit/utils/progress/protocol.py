from typing import Protocol, runtime_checkable, Any
from conduit.utils.progress.verbosity import Verbosity


@runtime_checkable
class DisplayHandler(Protocol):
    """
    Protocol for handling UI feedback (spinners, logs, completion markers).
    Abstracts the difference between Rich (TUI) and Plain (CLI/Logs).
    """

    def show_spinner(
        self, model_name: str, query_preview: str, verbosity: Verbosity
    ) -> None:
        """
        Display the initial loading state (e.g., spinner or 'Starting...' log).
        """
        ...

    def show_complete(
        self,
        model_name: str,
        query_preview: str,
        duration: float,
        verbosity: Verbosity,
        response_obj: Any | None = None,
    ) -> None:
        """
        Display success state (e.g., Green Check or 'Completed').
        Should also handle detailed object printing if verbosity allows (SUMMARY/DETAILED/DEBUG).
        """
        ...

    def show_cached(
        self, model_name: str, query_preview: str, duration: float, verbosity: Verbosity
    ) -> None:
        """
        Display cache hit state (e.g., Lightning Bolt or 'Cached').
        """
        ...

    def show_failed(
        self,
        model_name: str,
        query_preview: str,
        error: str,
        verbosity: Verbosity,
        error_obj: Any | None = None,
    ) -> None:
        """
        Display failure state (e.g., Red X or 'Failed').
        Should also handle detailed error printing if verbosity allows.
        """
        ...
