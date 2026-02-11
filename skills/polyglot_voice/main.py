def speech_to_text(audio_path: str, language: str = "fr"):
    """
    Converts audio input to text using OpenAI Whisper.
    """
    from openai import OpenAI

    client = OpenAI()

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, language=language
        )
    return transcript.text


def text_to_speech(text: str, language: str = "fr"):
    """
    Converts informative text to speech for the user.
    """
    import uuid
    from openai import OpenAI

    client = OpenAI()

    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    # Use unique filename to prevent concurrent request conflicts
    output_path = f"/tmp/tts_{uuid.uuid4().hex}.mp3"
    response.stream_to_file(output_path)
    return output_path
