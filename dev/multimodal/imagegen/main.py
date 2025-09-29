from conduit.result.response import Response
from conduit.model.model import Model
from conduit.message.imagemessage import ImageMessage

m = Model("Jlonge4/flux-dev-fp8")
response = m.query(query_input="create an image of st. dymphna", output_type="image")

assert isinstance(response, Response)
assert isinstance(response.message, ImageMessage)
response.display()
