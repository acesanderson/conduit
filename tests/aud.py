from conduit.sync import Model


def generate_audio(prompt_str: str):
    # 1. Initialize the Model with an OpenAI TTS model ID
    # Options: "tts-1", "tts-1-hd"
    model = Model("gpt-4o-mini-tts")

    print("Generating audio...")

    # 2. Use the .audio namespace to generate speech
    # This routes through Conduit -> ModelSync -> AudioSync -> OpenAIClient
    response = model.audio.generate(
        prompt_str=prompt_str,
        voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
        speed=1.0,
    )

    # 3. Play the audio directly
    # (Requires pydub and ffmpeg/ffplay installed on the system)
    try:
        response.play()
    except Exception as e:
        print(f"Could not play audio directly: {e}")

    # 4. Alternatively, save the audio to a file manually
    import base64

    # The raw base64 data is stored in the message content
    audio_b64 = response.message.content
    output_filename = "conduit_speech.mp3"

    with open(output_filename, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    print(f"Audio saved to {output_filename}")


if __name__ == "__main__":
    from pathlib import Path

    prompt = (
        Path("~/morphy/Summarization techniques from Arxiv survey.md").expanduser()
    ).read_text()

    generate_audio(prompt)
