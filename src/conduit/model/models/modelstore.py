"""
Our list of API llm models is in models.json, in our repo, as is the list of aliases.
The list of ollama models is host-specific, and therefore in our state directory.
Ollama context sizes is user-configurable, and therefore in our config directory.
"""

from conduit.model.models.providerstore import ProviderStore
from conduit.model.models.provider import Provider
from conduit.model.models.modelspec import ModelSpec
from conduit.model.models.modelspecs_CRUD import (
    get_modelspec_by_name,
    get_all_modelspecs,
    delete_modelspec,
    get_all_model_names,
)
from conduit.model.models.research_models import create_modelspec
from xdg_base_dirs import xdg_state_home, xdg_config_home
from pathlib import Path
import json
import itertools
import logging
from rich.console import RenderableType

logger = logging.getLogger(__name__)

# Our data stores
DIR_PATH = Path(__file__).parent
MODELS_PATH = DIR_PATH / "models.json"
OLLAMA_MODELS_PATH = xdg_state_home() / "conduit" / "ollama_models.json"
OLLAMA_CONTEXT_SIZES_PATH = xdg_config_home() / "conduit" / "ollama_context_sizes.json"
SERVER_MODELS_PATH = xdg_state_home() / "conduit" / "server_models.json"
ALIASES_PATH = DIR_PATH / "aliases.json"


class ModelStore:
    """
    Class to manage model information for the Conduit library.
    Provides methods to retrieve supported models, aliases, and validate model names.
    """

    @classmethod
    def models(cls):
        """
        Definitive list of models supported by Conduit library, as well as the local list of ollama models.
        """
        with open(MODELS_PATH) as f:
            models_json = json.load(f)
        if OLLAMA_MODELS_PATH.exists():
            with open(OLLAMA_MODELS_PATH) as f:
                ollama_models = json.load(f)
            models_json["ollama"] = ollama_models["ollama"]
        if SERVER_MODELS_PATH.exists():
            with open(SERVER_MODELS_PATH) as f:
                server_models = json.load(f)
            models_json["ollama"] += server_models["ollama"]
        return models_json

    @classmethod
    def list_models(cls) -> list[str]:
        """List of all models supported by Conduit library."""
        models = cls.models()
        return list(itertools.chain.from_iterable(models.values()))

    @classmethod
    def list_model_types(cls) -> list[str]:
        """List of model types supported by Conduit library."""
        return [
            "image_analysis",
            "image_gen",
            "audio_analysis",
            "audio_gen",
            "video_analysis",
            "video_gen",
            "reasoning",
            "text_completion",
        ]

    @classmethod
    def list_providers(cls) -> list[str]:
        """Definitive list of providers supported by Conduit library."""
        providers = ProviderStore.get_all_providers()
        return [provider.provider for provider in providers]

    @classmethod
    def identify_provider(cls, model: str) -> Provider:
        """
        Identify the provider for a given model.
        Returns the Provider object if found, raises ValueError otherwise.
        """
        models = cls.models()
        for provider, model_list in models.items():
            if model in model_list:
                return provider
        raise ValueError(f"Provider not found for model: {model}")

    @classmethod
    def local_models(cls) -> list[str]:
        """List of all locally hosted models supported by Conduit library."""
        models = cls.models()
        local_models = []
        for provider, model_list in models.items():
            if provider in ["ollama", "local"]:
                local_models.extend(model_list)
        return local_models

    @classmethod
    def aliases(cls):
        """Definitive list of model aliases supported by Conduit library."""
        with open(ALIASES_PATH) as f:
            return json.load(f)

    @classmethod
    def is_supported(cls, model: str) -> bool:
        """
        Check if the model is supported by the Conduit library.
        Returns True if the model is supported, False otherwise.
        """
        in_aliases = model in cls.aliases().keys()
        in_models = model in list(itertools.chain.from_iterable(cls.models().values()))
        return in_aliases or in_models

    @classmethod
    def _validate_model(cls, model: str) -> str:
        """
        Validate the model name against the supported models and aliases.
        Converts aliases to their corresponding model names if necessary.
        """
        # Load aliases
        aliases = cls.aliases()
        # Assign models based on aliases
        if model in cls.aliases().keys():
            model = aliases[model]
        elif cls.is_supported(model):
            model = model
        else:
            ValueError(
                f"WARNING: Model not found locally: {model}. This may cause errors."
            )
        return model

    @classmethod
    def get_num_ctx(cls, ollama_model: str) -> int:
        """
        Get the preferred num_ctx for a given ollama model.
        """
        default_value = 32768
        if OLLAMA_CONTEXT_SIZES_PATH.exists():
            with open(OLLAMA_CONTEXT_SIZES_PATH) as f:
                context_sizes = json.load(f)
            if ollama_model in context_sizes:
                return context_sizes[ollama_model]
            else:
                logger.warning(
                    f"Model {ollama_model} not found in context sizes file. Using default value of {default_value}."
                )
                return default_value  # Default context size if not specified -- may throw an error for smaller models
        else:
            raise FileNotFoundError(
                f"Context sizes file not found: {ollama_context_sizes_path}"
            )

    @classmethod
    def _generate_renderable_model_list(cls) -> RenderableType:
        """
        Generate a Rich renderable object displaying the list of models in three columns.
        """
        from rich.console import Console
        from rich.columns import Columns
        from rich.text import Text

        models = cls.models()

        # Calculate total items and split points
        all_items = []
        for provider, model_list in models.items():
            all_items.append((provider, None))  # Provider header
            for model in model_list:
                all_items.append((provider, model))

        # Split into three roughly equal columns
        total_items = len(all_items)
        col1_end = total_items // 3
        col2_end = 2 * total_items // 3

        # Create three columns
        left_column = Text()
        middle_column = Text()
        right_column = Text()

        for i, (provider, model) in enumerate(all_items):
            if i < col1_end:
                target_column = left_column
            elif i < col2_end:
                target_column = middle_column
            else:
                target_column = right_column

            if model is None:  # Provider header
                target_column.append(f"{provider.upper()}\n", style="bold cyan")
            else:  # Model name
                target_column.append(f"  {model}\n", style="yellow")

        renderable_columns = Columns(
            [left_column, middle_column, right_column], equal=True, expand=True
        )
        return renderable_columns

    @classmethod
    def display(cls):
        """
        Display all available models in a formatted three-column layout to the console.
        """
        from rich.console import Console

        console = Console()
        renderable_columns = cls._generate_renderable_model_list()
        console.print(renderable_columns)

    # Consistency
    @classmethod
    def update(cls):
        """
        Compare model list from models.json, and make sure there are corresponding ModelSpec objects in database.
        Delete objects that don't have their model name in models.json; and create new ModelSpec objects if they are not in db.
        """
        if not cls._is_consistent():
            print(
                "Model specifications are not consistent with models.json. Updating..."
            )
            logger.info(
                "Model specifications are not consistent with models.json. Updating..."
            )
            cls._update_models()
        else:
            print(
                "Model specifications are consistent with models.json. No update needed."
            )
            logger.info(
                "Model specifications are consistent with models.json. No update needed."
            )

    @classmethod
    def _update_models(cls):
        # Get all ModelSpec objects from the database
        modelspec_db_names = set(get_all_model_names())
        # Create a set of model names from the models.json file
        models = cls.models()
        models_json_names = set(itertools.chain.from_iterable(models.values()))
        # Find models that are in models.json but not in the database
        models_not_in_modelspec_db = models_json_names - modelspec_db_names
        # Find models that are in the database but not in models.json
        models_not_in_model_list = modelspec_db_names - models_json_names
        logger.info(
            f"Found {len(models_not_in_modelspec_db)} models not in the database."
        )
        logger.info(f"Found {len(models_not_in_model_list)} models not in models.json.")
        # Delete all ModelSpec objects that are not in models.json
        logger.info(
            f"Deleting {len(models_not_in_model_list)} models not in models.json."
        )
        [delete_modelspec(model) for model in models_not_in_model_list]
        # Create all Modelspec objects that are in models.json but not in the database
        logger.info(
            f"Creating {len(models_not_in_modelspec_db)} models not in the database."
        )
        [create_modelspec(model) for model in models_not_in_modelspec_db]
        if cls._is_consistent():
            logger.info(
                "Model specifications are now consistent with models.json. Update complete."
            )
            return
        else:
            raise ValueError(
                "Model specifications are not consistent with models.json, after running .update()."
            )

    @classmethod
    def _is_consistent(cls) -> bool:
        """
        Check if the model specifications in the database are consistent with the models.json file.
        Returns True if consistent, False otherwise.
        """
        # Get list of models from models.json
        models = cls.models()

        # Get all ModelSpec objects from the database
        model_specs = get_all_modelspecs()

        # Create a set of model names from the models.json file
        model_names = set(itertools.chain.from_iterable(models.values()))

        consistent = True
        # Check if all ModelSpec names are in the models.json file
        for model_spec in model_specs:
            if model_spec.model not in model_names:
                consistent = False
        # Check if all models in models.json have a corresponding ModelSpec object
        for _, model_list in models.items():
            for model in model_list:
                if not any(model_spec.model == model for model_spec in model_specs):
                    consistent = False

        return consistent

    # Getters
    @classmethod
    def get_model(cls, model: str) -> ModelSpec:
        """
        Get the model name, validating against aliases and supported models.
        """
        model = cls._validate_model(model)
        try:
            return get_modelspec_by_name(model)
        except ValueError:
            raise ValueError(f"Model '{model}' not found in the database.")

    @classmethod
    def get_all_models(cls) -> list[ModelSpec]:
        """
        Get all models as ModelSpec objects.
        """
        return get_all_modelspecs()

    ## Get subsets of models by provider
    @classmethod
    def by_provider(cls, provider: Provider) -> list[ModelSpec]:
        """
        Get a list of models for a specific provider.
        """
        return [
            modelspec
            for modelspec in cls.get_all_models()
            if modelspec.provider == provider
        ]

    ## Get subsets of models by type
    @classmethod
    def by_type(cls, model_type: str) -> list[ModelSpec]:
        """
        Get a list of models by type.
        Raises ValueError if the model type is not valid.
        """
        if model_type not in cls.list_model_types():
            raise ValueError(
                f"Invalid model type: {model_type}. Must be one of: {', '.join(cls.list_model_types())}."
            )
        match model_type:
            case "image_analysis":
                return cls.image_analysis_models()
            case "image_gen":
                return cls.image_gen_models()
            case "audio_analysis":
                return cls.audio_analysis_models()
            case "audio_gen":
                return cls.audio_gen_models()
            case "video_analysis":
                return cls.video_analysis_models()
            case "video_gen":
                return cls.video_gen_models()
            case "reasoning":
                return cls.reasoning_models()
            case "text_completion":
                return cls.text_completion_models()
            case _:
                raise ValueError(
                    f"Invalid model type: {model_type}. Must be one of: {', '.join(cls.list_model_types())}."
                )

    ## Get lists of models by capability
    @classmethod
    def image_analysis_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support image analysis.
        """
        return [
            modelspec for modelspec in cls.get_all_models() if modelspec.image_analysis
        ]

    @classmethod
    def image_gen_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support image generation.
        """
        return [modelspec for modelspec in cls.get_all_models() if modelspec.image_gen]

    @classmethod
    def audio_analysis_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support audio analysis.
        """
        return [
            modelspec for modelspec in cls.get_all_models() if modelspec.audio_analysis
        ]

    @classmethod
    def audio_gen_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support audio generation.
        """
        return [modelspec for modelspec in cls.get_all_models() if modelspec.audio_gen]

    @classmethod
    def video_analysis_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support video analysis.
        """
        return [
            modelspec for modelspec in cls.get_all_models() if modelspec.video_analysis
        ]

    @classmethod
    def video_gen_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support video generation.
        """
        return [modelspec for modelspec in cls.get_all_models() if modelspec.video_gen]

    @classmethod
    def reasoning_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support reasoning.
        """
        return [modelspec for modelspec in cls.get_all_models() if modelspec.reasoning]

    @classmethod
    def text_completion_models(cls) -> list[ModelSpec]:
        """
        Get a list of models that support text completion.
        """
        return [
            modelspec for modelspec in cls.get_all_models() if modelspec.text_completion
        ]
