from Chain.model.model_client import ModelClient


def test_local_sync():
    from Chain.result.response import Response

    model = ModelClient("llama3.1:latest")
    response = model.query("name ten cetaceans", cache=False)
    content = str(response.content)
    assert isinstance(response, Response), f"Expected Response, got {type(response)}"
    assert len(content) > 0, "Response content is empty"


print("hi bitch")
