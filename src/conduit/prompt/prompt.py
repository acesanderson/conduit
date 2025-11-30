"""
Prompt class -- coordinates templates, input variables, and rendering.
"""

from jinja2 import Environment, StrictUndefined, meta
from typing import TYPE_CHECKING
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

    def __init__(self, prompt_string: str):
        self.prompt_string = prompt_string
        self.template = env.from_string(prompt_string)

    def render(self, input_variables: dict) -> str:
        """
        takes a dictionary of variables
        """
        rendered_prompt = self.template.render(
            **input_variables
        )  # this takes all named variables from the dictionary we pass to this.
        return rendered_prompt

    @property
    def input_schema(self) -> set[str]:
        """
        Returns a set of variable names from the template.
        This can be used to validate that the input variables match the template.
        """
        parsed_content = env.parse(self.prompt_string)
        return meta.find_undeclared_variables(parsed_content)

    @classmethod
    def from_file(cls, filename: "str | Path") -> "Prompt":
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

    def __repr__(self):
        attributes = ", ".join(
            [f"{k}={repr(v)[:50]}" for k, v in self.__dict__.items()]
        )
        return f"{self.__class__.__name__}({attributes})"
