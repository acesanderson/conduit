"""
CRUD functions for TinyDB; handles serializing / deserializing our ModelCapabilities objects.
"""

from conduit.model.models.modelspec import ModelSpec
from tinydb import TinyDB, Query
from pathlib import Path

dir_path = Path(__file__).parent
modelspecs_file = dir_path / "modelspecs.json"
db = TinyDB(modelspecs_file)


def create_modelspecs_from_scratch(modelspecs=list[ModelSpec]) -> None:
    """
    Create a new empty database for ModelSpecs.
    This will overwrite any existing data in modelspecs.json.
    """
    global db
    db.close()  # Close the existing database if it exists
    if modelspecs_file.exists():
        modelspecs_file.unlink()  # Delete the existing file
    db = TinyDB(modelspecs_file)  # Reopen the database
    print(f"Created new ModelSpecs database at {modelspecs_file}")
    # Now populate the database with any initial data if needed
    if modelspecs:
        for spec in modelspecs:
            add_modelspec(spec)


def add_modelspec(model_spec: ModelSpec) -> None:
    """
    Add a ModelSpec to the database.
    """
    db.insert(model_spec.model_dump())


def create_modelspec(model_spec: ModelSpec) -> None:
    t


def get_all_modelspecs() -> list[ModelSpec]:
    """
    Retrieve all ModelSpecs from the database.
    """
    return [ModelSpec(**item) for item in db.all()]


def get_modelspec_by_name(model: str) -> ModelSpec:
    """
    Retrieve a ModelSpec by its name.
    """
    ModelQuery = Query()
    item = db.search(ModelQuery.model == model)
    if item:
        return ModelSpec(**item[0])
    else:
        raise ValueError(f"ModelSpec with name '{model}' not found.")


def get_all_model_names() -> list[str]:
    """
    Retrieve all model names from the database.
    """
    ModelQuery = Query()
    return [item["model"] for item in db.all()]


def update_modelspec(model: str, updated_spec: ModelSpec) -> None:
    """
    Update a ModelSpec in the database.
    """
    ModelQuery = Query()
    db.update(updated_spec.model_dump(), ModelQuery.model == model)


def delete_modelspec(model: str) -> None:
    """
    Delete a ModelSpec from the database.
    """
    ModelQuery = Query()
    db.remove(ModelQuery.model == model)


def in_db(model_spec: ModelSpec) -> bool:
    """
    Check if a ModelSpec is already in the database.
    """
    ModelQuery = Query()
    item = db.search(ModelQuery.model == model_spec.model)
    return bool(item)
