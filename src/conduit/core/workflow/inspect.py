import ast
import inspect
from collections import deque


def scan_workflow(root_func) -> dict:
    """
    Recursively scans a workflow function and its @step children for configuration keys.
    """
    # 1. Setup Traversal
    schema = {}
    visited = set()
    # Handle the case where the root itself is decorated
    root = getattr(root_func, "__wrapped__", root_func)
    queue = deque([root])
    visited.add(root)

    while queue:
        current_func = queue.popleft()

        try:
            # Get the source code of the underlying function
            src = inspect.getsource(current_func)
            tree = ast.parse(src)

            # We need the globals of the function to resolve names (e.g. "summarize_requests")
            # to actual python objects
            func_globals = getattr(current_func, "__globals__", {})

        except (OSError, TypeError):
            # Cannot get source (dynamic code, builtins, etc.)
            continue

        # 2. Walk the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # --- A: CHECK FOR CONFIG (resolve_param) ---
                if getattr(node.func, "id", "") == "resolve_param":
                    if len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                        key = node.args[0].value

                        # Extract default
                        default_val = "dynamic"
                        if len(node.args) > 1:
                            if isinstance(node.args[1], ast.Constant):
                                default_val = node.args[1].value
                            elif isinstance(node.args[1], ast.List):
                                default_val = [
                                    getattr(e, "value", "unknown")
                                    for e in node.args[1].elts
                                ]

                        full_key = f"{current_func.__name__}.{key}"
                        schema[full_key] = default_val

                # --- B: CHECK FOR CHILDREN (Recursion) ---
                # We look for function calls like `await summarize_requests(...)`
                # The node.func might be an Attribute (obj.method) or Name (func)
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name and func_name in func_globals:
                    # Resolve the name to the object
                    child_obj = func_globals[func_name]

                    # Check if it is a @step (has __wrapped__)
                    # or just a regular function we haven't seen.
                    # We prefer scanning things that have been wrapped.
                    underlying = getattr(child_obj, "__wrapped__", None)

                    if underlying and underlying not in visited:
                        visited.add(underlying)
                        queue.append(underlying)

    return schema


def generate_mermaid(root_func) -> str:
    """
    Generates a high-level Call Hierarchy graph.
    Visualizes 'Parent calls Child' relationships without line numbers or logic control flow.
    """
    # Use a set to track processed functions to avoid infinite recursion
    visited_funcs = set()
    queue = deque([root_func])

    # We will collect edges as strings like "parent --> child"
    edges = []

    while queue:
        current_func = queue.popleft()
        real_func = getattr(current_func, "__wrapped__", current_func)

        # Deduplication: Don't scan the same function twice
        if real_func in visited_funcs:
            continue
        visited_funcs.add(real_func)

        parent_name = real_func.__name__

        try:
            src = inspect.getsource(real_func)
            src = inspect.cleandoc(src)
            tree = ast.parse(src)
            func_globals = getattr(real_func, "__globals__", {})
        except (OSError, TypeError):
            continue

        # Find unique dependencies in this function
        # We use a set here because if process_email calls extract_sender twice,
        # we only want to draw ONE structural arrow for the dependency.
        found_dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target_name = None

                # Check for: function_name(...)
                if isinstance(node.func, ast.Name):
                    target_name = node.func.id

                if target_name and target_name in func_globals:
                    child_obj = func_globals[target_name]

                    # CRITICAL FILTER: Only graph @steps
                    if getattr(child_obj, "__wrapped__", None):
                        if target_name not in found_dependencies:
                            found_dependencies.add(target_name)

                            # Add Edge
                            edges.append(f"  {parent_name} --> {target_name}")

                            # Add child to queue to find ITS children (indentation level 2+)
                            queue.append(child_obj)

    # Wrap in Mermaid Definition
    # graph LR (Left-Right) looks most like an indented tree
    return "graph LR\n" + "\n".join(edges)
