import functools
import inspect
import time


def step(func):
    if not inspect.iscoroutinefunction(func):
        raise TypeError(
            f"@step requires async functions. "
            f"{func.__name__} must be defined with `async def`."
        )

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()

        current_meta = {}
        token_meta = _step_meta_ctx.set(current_meta)

        try:
            result = await func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = str(e)
            status = "error"
            raise
        finally:
            duration = time.time() - start

            trace_list = _trace_ctx.get()
            if trace_list is not None:
                trace_list.append(
                    {
                        "step": func.__name__,
                        "inputs": {"args": args, "kwargs": kwargs},
                        "output": result,
                        "duration": round(duration, 4),
                        "status": status,
                        "metadata": current_meta,
                    }
                )

            _step_meta_ctx.reset(token_meta)

        return result

    return wrapper
