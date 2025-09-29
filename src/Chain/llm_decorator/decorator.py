from Chain.model.model import Model
from Chain.prompt.prompt import Prompt
from Chain.chain.chain import Chain
from Chain.logs.logging_config import get_logger
from typing import Callable
import inspect, re

logger = get_logger(__name__)

# Constants
param_string_pattern = re.compile("{{(.+)}}")


def _validate_template_params(func: Callable) -> None:
    """
    Validate that function parameters match template variables in docstring.

    Raises:
        ValueError: If parameters don't match template variables
    """
    # Get function signature parameters
    sig = inspect.signature(func)
    func_params = set()

    for param_name, param in sig.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        func_params.add(param_name)

    # Extract template variables from docstring
    docstring = func.__doc__
    if not docstring:
        raise ValueError(
            f"Function '{func.__name__}' must have a docstring containing the prompt template"
        )

    # Find all {{variable}} patterns
    template_vars = set(re.findall(r"\{\{(\w+)\}\}", docstring))

    # Compare the sets
    missing_in_template = func_params - template_vars
    missing_in_signature = template_vars - func_params

    errors = []

    if missing_in_template:
        errors.append(
            f"Parameters in function signature but not in template: {missing_in_template}"
        )

    if missing_in_signature:
        errors.append(
            f"Variables in template but not in function signature: {missing_in_signature}"
        )

    if errors:
        raise ValueError(
            f"Template validation failed for function '{func.__name__}':\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


def _wrap_params(prompt: str) -> str:
    """
    Generally speaking, wrapping context in xml tags is very helpful for LLMs.
    All prompts written as docstrings for an llm function get wrapped in XML.

    Example:

    {{param_name}} becomes:

    <param_name>
    {{param_name}}
    </param_name>
    """
    return (
        re.sub(
            param_string_pattern,
            lambda m: f"<{m.group(1)}>\n{{{{{m.group(1)}}}}}\n</{m.group(1)}> ",
            prompt,
        )
        + "\n"
    )


def llm(func: Callable = None, *, model="haiku") -> Callable:  # Note the *
    """
    Decorator to create a prompt function that can be used with an LLM.
    Adapts basic Chain syntax to the decorator pattern for easy composition.

    How to compose:

    @llm
    def my_prompt_function(input: str):
        \"""
        Your prompt template with {{input}}.
        \"""
    """

    def decorator(f: Callable) -> Callable:
        _validate_template_params(f)

        def wrapper(*args, **kwargs):  # Accept both positional and keyword
            # Get function signature to bind args properly

            sig = inspect.signature(f)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            model_object = Model(model)
            # Wrap any parameters in XML tags
            prompt_string = _wrap_params(f.__doc__)
            # Docstrings are tabbed / spaced, so we need to remove spaces at beginning of lines
            prompt_string = "\n".join(
                line.lstrip() for line in prompt_string.splitlines()
            )
            prompt = Prompt(prompt_string)
            chain = Chain(prompt=prompt, model=model_object)
            response = chain.run(input_variables=dict(bound_args.arguments))
            return response.content  # Note: .content to get just the text

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)
