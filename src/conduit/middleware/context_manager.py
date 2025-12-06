from contextlib import contextmanager, asynccontextmanager
from conduit.domain.result.result import ConduitResult

"""
What happens pre-execute:
- display progress status / set up spinner
- check cache for existing result (submit Request object)



"""


@contextmanager
def middleware_context_manager(self, request) -> dict:
    ctx = {}
    try:
        # Pre execute logic can go here
        yield ctx
    finally:
        result = ctx.get("result")
        # Post execute logic can go here
        pass
