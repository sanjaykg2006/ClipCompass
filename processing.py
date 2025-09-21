import whisper

def transcribe(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result["text"], result["segments"]
