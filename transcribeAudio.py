from google.cloud import speech_v1p1beta1 as speech

def transcribe_audio(audio_file_path):
    """Transcribes the audio file using Cloud Speech-to-Text (local dev)."""

    client = speech.SpeechClient()  # No explicit credentials needed!

    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    # 1. Determine encoding and sample rate (crucial!)
    encoding = None
    sample_rate_hertz = None

    if audio_file_path.lower().endswith(".mp3"):
        encoding = speech.RecognitionConfig.AudioEncoding.MP3
        sample_rate_hertz = 44100  # Or 48000, check your MP3 file
    elif audio_file_path.lower().endswith(".wav"):  # Add WAV support
        encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
        sample_rate_hertz = 16000  # Or another rate, check your WAV
    elif audio_file_path.lower().endswith(".flac"): # Add FLAC support
        encoding = speech.RecognitionConfig.AudioEncoding.FLAC
        sample_rate_hertz = 16000 # Or other rate, check your FLAC
    else:
        raise ValueError("Unsupported audio format. Use MP3, WAV, or FLAC.")

    config = speech.RecognitionConfig(
        encoding=encoding,  # Adjust if needed
        sample_rate_hertz=sample_rate_hertz,      # Adjust if needed
        language_code="en-US",       # Set the correct language
        enable_automatic_punctuation=True,  # Optional
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        for alternative in result.alternatives:
            print(f"Transcript: {alternative.transcript}")
            print(f"Confidence: {alternative.confidence}")

# Example usage:
audio_file_path = "/Users/goutham.reddy/Downloads/audioshort.mp3"  # Replace with your audio file
transcribe_audio(audio_file_path)