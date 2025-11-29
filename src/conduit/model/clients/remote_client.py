from conduit.model.clients.client import Client
from conduit.model.clients.client import Usage
from headwater_api.classes import StatusResponse
from conduit.request.request import Request
from headwater_client.client.headwater_client import HeadwaterClient
from typing import override


class RemoteClient(Client):
    def __init__(self):
        hc = HeadwaterClient()
        self._client = self._initialize_client().conduit
        self._validate_server_model()

    @override
    def _initialize_client(self) -> HeadwaterClient:
        """Initialize SiphonClient connection"""
        client = HeadwaterClient()

        # Test connection
        try:
            status = client.ping()
            if status == False:
                raise ConnectionError("Server is not healthy")
            return client
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Headwater server: {e}")

    def _validate_server_model(self):
        """Validate that the model is available on the server"""
        try:
            status: StatusResponse = self._client.get_status()
            available_models = getattr(status, "models_available", [])
            logger.info(f"Available models on server: {available_models}")
            # Update server models file
            with open(settings.paths["SERVER_MODELS_PATH"], "w") as f:
                json_dict = {"ollama": available_models}
                _ = f.write(json.dumps(json_dict, indent=4))
            logger.debug(
                f"Updated server models file at {settings.paths['SERVER_MODELS_PATH']}"
            )

            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' not available on server. "
                    f"Available models: {available_models}"
                )
        except Exception as e:
            raise ValueError(f"Failed to validate model on server: {e}")

    @override
    def _get_api_key(self) -> str: ...

    @override
    def query(self, request: Request) -> tuple:
        """
        Query the remote model via HeadwaterClient.
        We get a Response model, which we unpack and repack into a Response object + Usage.
        """
        response = self._client.query_sync(request)
        content = response.content
        usage = Usage(
            input_tokens=response.get("input_tokens"),
            output_tokens=response.get("output_tokens"),
        )
        return content, usage

    @override
    def tokenize(self, model: str, text: str) -> int:
        """
        Get the token count for a text, per a given model's tokenization function.
        """
        raise NotImplementedError("Tokenization not implemented in RemoteClient.")
