from conduit.model.model import Model
from conduit.request.request import Request
from conduit.progress.wrappers import progress_display
from conduit.progress.verbosity import Verbosity
from conduit.message.textmessage import TextMessage
from conduit.result.result import ConduitResult
from conduit.result.response import Response
from conduit.cache.cache import ConduitCache
from conduit.logs.logging_config import get_logger
from typing import Optional
from time import time
from pydantic import ValidationError, BaseModel
import importlib

logger = get_logger(__name__)


class ModelAsync(Model):
    _async_clients = {}  # Separate from Model._clients
    _conduit_cache: Optional[ConduitCache] = None

    def _get_client_type(self, model: str) -> tuple:
        """
        Overrides the parent method to return the async version of each client type.
        """
        from conduit.model.models.modelstore import ModelStore

        model_list = ModelStore.models()
        if model in model_list["openai"]:
            return "openai", "OpenAIClientAsync"
        elif model in model_list["anthropic"]:
            return "anthropic", "AnthropicClientAsync"
        elif model in model_list["ollama"]:
            return "ollama", "OllamaClientAsync"
        elif model in model_list["google"]:
            return "google", "GoogleClientAsync"
        elif model in model_list["perplexity"]:
            return "perplexity", "PerplexityClientAsync"
        else:
            raise ValueError(f"Model {model} not found in models")

    @classmethod
    def _get_client(cls, client_type: tuple):
        # print(f"client type: {client_type}")
        if client_type[0] not in cls._async_clients:
            try:
                module = importlib.import_module(
                    f"conduit.model.clients.{client_type[0].lower()}_client"
                )
                client_class = getattr(module, f"{client_type[1]}")
                cls._async_clients[client_type[0]] = client_class()
            except ImportError as e:
                raise ImportError(f"Failed to import {client_type} client: {str(e)}")
        client_object = cls._async_clients[client_type[0]]
        if not client_object:
            raise ValueError(f"Client {client_type} not found in clients")
        return client_object

    @progress_display
    async def query_async(
        self,
        # Standard params
        query_input: str | list,
        verbose: Verbosity = Verbosity.PROGRESS,
        response_model: type[BaseModel] | None = None,
        raw=False,
        cache=False,
        print_response=False,
        # If hand-rolling Request params, you can just pass the object directly.
        request: Optional[Request] = None,
        # For debug: return Request, or an example Error
        return_request: bool = False,
        return_error: bool = False,
    ) -> ConduitResult:
        try:
            if request == None:
                import inspect

                frame = inspect.currentframe()
                args, _, _, values = inspect.getargvalues(frame)

                query_args = {k: values[k] for k in args if k != "self"}
                query_args["model"] = self.model
                if query_input:
                    query_args.pop("query_input", None)
                    request = Request.from_query_input(
                        query_input=query_input, **query_args
                    )
                else:
                    request = Request(**query_args)

            assert request and isinstance(request, Request), (
                f"request should be a Request object, not {type(request)}"
            )

            # For debug, return Request if requested
            if return_request:
                return request
            # For debug, return error if requested
            if return_error:
                from conduit.tests.fixtures import sample_error

                return sample_error

            # Check cache first
            if cache and self._conduit_cache:
                cached_result = self._conduit_cache.check_for_model(request)
                if cached_result is not None:
                    return cached_result  # This should be a Response

            # Execute the query
            start_time = time()
            result, usage = await self._client.query(request)
            stop_time = time()

            # Create Response object
            if isinstance(result, Response):
                response = result
            elif isinstance(result, str):
                assistant_message = TextMessage(role="assistant", content=result)

                response = Response(
                    message=assistant_message,
                    request=request,
                    duration=stop_time - start_time,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
            else:
                # Handle other result types (BaseModel, etc.)
                assistant_message = TextMessage(role="assistant", content=result)

                response = Response(
                    message=assistant_message,
                    request=request,
                    duration=stop_time - start_time,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )

            # Update cache after successful query
            if cache and self._conduit_cache:
                self._conduit_cache.store_for_model(request, response)

            return response  # Always return Response (part of ConduitResult)

        except ValidationError as e:
            from conduit.result.error import ConduitError

            return ConduitError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
        except Exception as e:
            from conduit.result.error import ConduitError

            return ConduitError.from_exception(
                e,
                code="async_query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
