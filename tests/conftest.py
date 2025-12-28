import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.request.generation_params import GenerationParams
from conduit.core.model.models.modelstore import ModelStore

# Add project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def mock_model_store_validation():
    """
    Mocks ModelStore.validate_model to always return True.
    This prevents validation errors in tests that instantiate GenerationParams.
    """
    with patch.object(ModelStore, "validate_model", side_effect=lambda model_name: model_name):
        yield


@pytest.fixture
def default_generation_params() -> GenerationParams:
    """
    Returns a default GenerationParams object.
    """
    return GenerationParams(model="gpt-4")


@pytest.fixture
def default_conduit_options() -> ConduitOptions:
    """
    Returns a default ConduitOptions object.
    """
    return ConduitOptions(project_name="test-project")