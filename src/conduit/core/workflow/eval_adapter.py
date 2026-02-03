import importlib
from collections.abc import Callable
import asyncio
import inspect
import sys
import os

from conduit.core.workflow.harness import ConduitHarness

# AUTOMATICALLY FIX PATH:
# Add the current working directory to sys.path so we can import user scripts (like 'evals.py')
# regardless of where this adapter script actually lives.
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


def load_workflow_target(target_path: str, method_name: str = None) -> Callable:
    """
    Dynamically loads a workflow function or class method from a string path.

    Args:
        target_path: Dot-notation path (e.g. "my_pkg.flows.email.EmailWorkflow")
        method_name: Optional method to call on the class instance (e.g. "run")

    Returns:
        The callable entry point for the workflow.
    """
    try:
        # 1. Split module and object name
        module_path, obj_name = target_path.rsplit(".", 1)

        # 2. Import the module
        module = importlib.import_module(module_path)

        # 3. Get the object (Function or Class)
        target_obj = getattr(module, obj_name)

        # CASE A: It's a Function -> Return it directly
        if not isinstance(target_obj, type):
            return target_obj

        # CASE B: It's a Class -> Instantiate it
        # Note: We assume a no-arg constructor for generic composability.
        # If your workflows need init params, inject them via Harness config instead.
        instance = target_obj()

        # CASE C: Specific Method (e.g., instance.run_workflow)
        if method_name:
            return getattr(instance, method_name)

        # CASE D: Callable Class (instance.__call__)
        return instance

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load workflow target '{target_path}': {e}")


def sanitize_output(obj):
    """
    Recursively converts Pydantic models and Enums to JSON-serializable types.
    """
    # 1. Handle Pydantic Models (v1 and v2 support)
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return sanitize_output(obj.model_dump())
    if hasattr(obj, "dict"):  # Pydantic v1
        return sanitize_output(obj.dict())

    # 2. Handle Enums (convert to their value string)
    if hasattr(obj, "value") and hasattr(obj, "name"):
        return obj.value

    # 3. Handle Lists
    if isinstance(obj, list):
        return [sanitize_output(item) for item in obj]

    # 4. Handle Dictionaries
    if isinstance(obj, dict):
        return {k: sanitize_output(v) for k, v in obj.items()}

    # 5. Return basic types as-is
    return obj


def call_api(prompt, options, context):
    """
    Bridge that handles Async + Serialization automatically.
    """
    run_config = options.get("config", {})
    target_path = run_config.pop("workflow_target", None)
    method_name = run_config.pop("entry_point", None)

    if not target_path:
        return {"error": "Missing 'workflow_target' in config"}

    try:
        workflow_callable = load_workflow_target(target_path, method_name)
        harness = ConduitHarness(config=run_config)

        # Execute
        result = harness.run(workflow_callable, prompt)

        # Resolve Async
        if inspect.iscoroutine(result):
            result = asyncio.run(result)

        # === THE FIX: Sanitize before returning ===
        # This converts your EmailCategoryOutput to a plain dict
        safe_result = sanitize_output(result)

        return {
            "output": safe_result,
            "metadata": {
                "trace": harness.trace_log,
            },
        }
    except Exception as e:
        import traceback

        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}
