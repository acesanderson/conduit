from conduit.model.model_remote import RemoteModel
from conduit.sync import Verbosity

VERBOSITY = Verbosity.COMPLETE


def test_local_sync():
    from conduit.result.response import Response

    model = RemoteModel("llama3.1:latest")
    response = model.query("name ten cetaceans", cache=False, verbose=VERBOSITY)
    assert isinstance(response, Response), f"Expected Response, got {type(response)}"
    assert len(content) > 0, "Response content is empty"
    content = str(response.content)
    print(content)
