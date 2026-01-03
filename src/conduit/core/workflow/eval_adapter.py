"""
Adapter layer for integrating Conduit workflows with Promptfoo evaluation framework. This module dynamically loads workflow definitions from dot-notation paths and executes them within the ConduitHarness observability context, enabling tracing and configuration injection. The `load_workflow_target()` function performs runtime module/class resolution to support both standalone functions and callable class instances, while `call_api()` wraps the execution for Promptfoo's evaluation protocol by extracting workflow references from config, running them with context management, and returning results with trace metadata for cost analysis and debugging.

Usage:
```yaml
providers:
  - id: python:eval_adapter.py
    label: "Email Flow (GPT-4)"
    config:
      # THE DYNAMIC PARTS
      workflow_target: "library.workflows.email.EmailWorkflow"
      entry_point: "run"

      # THE TUNABLE PARTS (Passed to Harness)
      model: "gpt-4"
      verbosity: "high"
```
```python
# Promptfoo config specifies workflow target and entry point
result = call_api(
    prompt="Your input prompt",
    options={"config": {"workflow_target": "my_app.flows.EmailWorkflow", "entry_point": "run"}},
    context={}
)
# Returns {"output": result_value, "metadata": {"trace": [...]}}
```
"""

import importlib
from collections.abc import Callable

# Use your actual imports here
from conduit.core.workflow.workflow import ConduitHarness


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


def call_api(prompt, options, context):
    """
    Generic Promptfoo bridge.
    """
    # 1. Extract Config
    # We look for special keys 'workflow' and 'entry_point'
    run_config = options.get("config", {})

    target_path = run_config.pop("workflow_target", None)
    method_name = run_config.pop("entry_point", None)

    if not target_path:
        return {
            "error": "Missing 'workflow_target' in config (e.g. 'flows.email.EmailWorkflow')"
        }

    try:
        # 2. Dynamically Load the Logic
        workflow_callable = load_workflow_target(target_path, method_name)

        # 3. Initialize Harness (Trace/Config Context)
        harness = ConduitHarness(config=run_config)

        # 4. Execute
        # Note: We pass *prompt as the first arg. If your workflow takes multiple args,
        # you might need to parse 'prompt' as JSON or use run_config for others.
        output = harness.run(workflow_callable, prompt)

        # 5. Return to Promptfoo
        return {
            "output": output,
            "metadata": {
                "trace": harness.trace,
                # "cost": calculate_cost(harness.trace) # Add your helper here
            },
        }
    except Exception as e:
        # Return full traceback for easier debugging in Promptfoo UI
        import traceback

        return {"error": f"{str(e)}\n\n{traceback.format_exc()}"}
