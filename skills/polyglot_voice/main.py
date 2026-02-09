import antigravity as ag

@ag.skill(name="polyglot_voice_stt")
def speech_to_text(audio_path: str, language: str = "fr"):
    """
    Converts audio input to text using OpenAI Whisper.
    """
    from openai import OpenAI
    client = OpenAI()
    
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            language=language
        )
    return transcript.text

@ag.skill(name="polyglot_voice_tts")
def text_to_speech(text: str, language: str = "fr"):
    """
    Converts informative text to speech for the user.
    """
    from openai import OpenAI
    client = OpenAI()
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    # Target path for the output audio
    output_path = "output_voice.mp3"
    response.stream_to_file(output_path)
    return output_path
