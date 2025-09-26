def tts_openai(text: str) -> bytes:
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.audio.speech.create(
        model="tts-1",  # or "tts-1-hd"
        voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
        input="Today is a wonderful day to build something people love!",
    )
    return response.content
