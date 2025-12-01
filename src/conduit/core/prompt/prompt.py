"""
Prompt class -- coordinates templates, input variables, and rendering.
"""

from __future__ import annotations
from jinja2 import Environment, StrictUndefined, meta, Template
from typing import TYPE_CHECKING, override
import logging

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Define jinja2 environment that we will use across all prompts.
env = Environment(
    undefined=StrictUndefined
)  # set jinja2 to throw errors if a variable is undefined


class Prompt:
    """ "
    Takes a jinja2 ready string (note: not an actual Template object; that's created by the class).
    The three stages of prompt creation:
    - the prompt string, which is provided to the class
    - the jinja template created from the prompt string
    - the rendered prompt, which is returned by the class and submitted to the LLM model.
    """

    # Init methods
    def __init__(self, prompt_string: str):
        self.prompt_string: str = prompt_string
        self.template: Template = env.from_string(prompt_string)
        self.input_schema: set[str] = self._get_input_schema()

    def _get_input_schema(self) -> set[str]:
        """
        Returns a set of variable names from the template.
        This can be used to validate that the input variables match the template.
        """
        parsed_content = env.parse(self.prompt_string)
        return meta.find_undeclared_variables(parsed_content)

    # Main methods
    def render(self, input_variables: dict[str, str]) -> str:
        """
        takes a dictionary of variables
        """
        rendered_prompt = self.template.render(
            **input_variables
        )  # this takes all named variables from the dictionary we pass to this.
        return rendered_prompt

    # Factory functions
    @classmethod
    def from_file(cls, filename: str | Path) -> Prompt:
        """
        Creates a Prompt object from a file containing the prompt string.
        """
        from pathlib import Path

        # Coerce to Path object
        if not isinstance(filename, str):
            try:
                filename = Path(filename)
            except TypeError as e:
                logger.error(f"Invalid filename type: {type(filename)}")
                raise e

        assert isinstance(filename, Path), "filename must be a Path object"

        # Confirm file exists
        if not filename.exists():
            raise FileNotFoundError(f"Prompt file {filename} does not exist.")

        # Confirm file is .jinja2 or .jinja
        if filename.suffix not in {".jinja2", ".jinja"}:
            raise ValueError(
                f"Prompt file {filename} must be a .jinja2 or .jinja file."
            )

        # Read the file content
        with filename.open("r", encoding="utf-8") as file:
            prompt_string = file.read()

        # Create and return a Prompt object
        return cls(prompt_string)

    # Validation methods
    def validate_input_variables(self, input_variables: dict[str, str]) -> None:
        """
        Validates that the input variables match the template.
        """
        # Determine if prompt is expecting variables that are not provided
        missing_vars: set[str] = self.input_schema - input_variables.keys()
        if missing_vars:
            raise ValueError(
                f'Prompt is missing required input variable(s): "{'", "'.join(missing_vars)}"'
            )
        # Determine if extra variables are provided that the prompt does not expect
        extra_vars: set[str] = input_variables.keys() - self.input_schema
        if extra_vars:
            raise ValueError(
                f'Provided input variable(s) are not referenced in prompt: "{'", "'.join(extra_vars)}"'
            )

    def tokenize(self, model: str) -> int:
        """
        Tokenizes the prompt string using the specified model's tokenizer.
        """
        from conduit.core.model.model_sync import ModelSync

        model_obj = ModelSync(model)
        tokens: int = model_obj.tokenize(text=self.prompt_string)
        return tokens

    # Dunders
    @override
    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
