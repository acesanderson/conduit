"""
Instructor library does most of the heavy lifting here -- great piece of software.
Parser wraps our Pydantic model as part of Conduit orchestration.
We don't want to serialize Parser or save it in any fashion; instead this is a bundle of class/static methods.
Serialization (for caching, messagestore, etc.) of pydantic models is entirely done through its json schema, so we can use the same model.

Three purposes for this class:
- as a store of Pydantic models, so we can validate them later
- as a wrapper for Pydantic models for use with Conduit
- as a library of static methods for handling Pydantic models (include special logic for Perplexity API, and serialization / hashing)
"""

from typing import Union, Type
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class Parser:
    _response_models: list[
        Type[BaseModel]
    ] = []  # Store all pydantic classes, important for de-serialization

    def __init__(self, pydantic_model: Union[Type[BaseModel], type]):
        """
        Initialize the Parser with a model specification.
        :param pydantic_model: A Pydantic BaseModel class or a list of BaseModel classes.
        """
        self.original_spec = pydantic_model
        self.pydantic_model = pydantic_model
        # Save response models to singleton for validation
        if isinstance(pydantic_model, type) and issubclass(pydantic_model, BaseModel):
            if pydantic_model not in self._response_models:
                self._response_models.append(pydantic_model)
        elif isinstance(pydantic_model, list):
            for model in pydantic_model:
                if isinstance(model, type) and issubclass(model, BaseModel):
                    if model not in self._response_models:
                        self._response_models.append(model)
                else:
                    raise ValueError(
                        "All items in the list must be Pydantic BaseModel subclasses."
                    )

    def __repr__(self):
        return f"Parser({self.original_spec})"

    # Static methods for handling Pydantic models
    @staticmethod
    def to_perplexity(pydantic_model: type) -> BaseModel | type:
        """
        Convert the Pydantic model to a type suitable for Perplexity API.
        For wrapper classes with single list fields, extract the inner type
        so Instructor can handle multiple tool calls properly.
        """
        # Handle wrapper models with a single list field
        if isinstance(pydantic_model, type) and issubclass(pydantic_model, BaseModel):
            # Pydantic v2
            if hasattr(pydantic_model, "model_fields"):
                fields = pydantic_model.model_fields
            # Pydantic v1 fallback
            else:
                fields = pydantic_model.__fields__

            # Check if it's a wrapper with a single list field
            if len(fields) == 1:
                field_name, field_info = next(iter(fields.items()))
                field_type = (
                    field_info.annotation
                    if hasattr(field_info, "annotation")
                    else field_info.type_
                )

                # Check if it's list[SomeBaseModel]
                field_origin = typing.get_origin(field_type)
                if field_origin is list:
                    args = typing.get_args(field_type)
                    if args and issubclass(args[0], BaseModel):
                        # Return List[InnerModel] for Instructor
                        return list[args[0]]

        # Handle direct list[BaseModel] type annotations
        origin = typing.get_origin(self.pydantic_model)
        if origin is list:
            args = typing.get_args(self.pydantic_model)
            if args and issubclass(args[0], BaseModel):
                return pydantic_model  # Return list[SomeModel] directly

        # For non-wrapper classes, return as-is
        return pydantic_model

    @staticmethod
    def as_string(pydantic_model: type[BaseModel]) -> str:
        """
        Convert the Pydantic model to a string representation.
        This is used for logging and debugging purposes (as well as constructing our cache hash for Params).
        """
        import json

        schema = pydantic_model.model_json_schema()
        # Deterministically serialize schema
        schema_str = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return schema_str
