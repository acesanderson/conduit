from Chain.tests.fixtures.sample_objects import sample_audio_message
from Chain.result.response import Response
from Chain.model.model import Model
from Chain.message.audiomessage import AudioMessage

m = Model("openai/whisper-base")
response = m.query(
    query_input="I am the very model of a modern major general", output_type="audio"
)

assert isinstance(response, Response)
assert isinstance(response.message, AudioMessage)
response.message.play()
