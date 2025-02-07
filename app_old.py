from flask import Flask, request, jsonify
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v3beta1 as translate
from google.cloud import texttospeech
from google.cloud import storage
import base64
import os
import traceback

app = Flask(__name__)

# Initialize Google Cloud clients (ADC will be used automatically)
speech_client = speech.SpeechClient()
translate_client = translate.TranslationServiceClient()
tts_client = texttospeech.TextToSpeechClient()
storage_client = storage.Client()

# Configuration (replace with your values)
PROJECT_ID = "jistar-hack25bom-201"  # Replace with your project ID
BUCKET_NAME = "bhasha-verse-bucket" # Replace with your bucket name

# 1. Audio to English Transcript
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    data = request.get_json()
    audio_uri = data.get('audio_uri')

    if not audio_uri:
        return jsonify({'error': 'Missing audio URI'}), 400

    try:
        audio = speech.RecognitionAudio(uri=audio_uri)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Adjust if needed
            sample_rate_hertz=16000,  # Adjust if needed
            language_code="en-US",
        )

        response = speech_client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            for alternative in result.alternatives:
                transcript += alternative.transcript

        return jsonify({'transcript': transcript})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2. English to Other Languages
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    print(f"Received data: {data}")
    english_text = data.get('text')
    target_language = data.get('target_language')
    text: str = "YOUR_TEXT_TO_TRANSLATE",
    language_code: str = "fr",

    if not english_text or not target_language:
        return jsonify({'error': 'Missing text or target language'}), 400

    try:
        parent = f"projects/{PROJECT_ID}/locations/global"
        
        text = "Hello, how are you?"
        translation = {
            "content": text,
            "mime_type": "text/plain",
        }

        response = translate_client.translate_text(
            contents=[translation],
            target_language_codes=["es"],
            parent=parent,
        )

        translated_text = ""
        if response.translations:
            translated_text = response.translations[0].translated_text

        return jsonify({'translated_text': translated_text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# 3. Text to Speech
@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    data = request.get_json()
    text = data.get('text')
    language_code = data.get('language_code')
    voice_name = data.get('voice_name')

    if not text or not language_code or not voice_name:
        return jsonify({'error': 'Missing text, language code, or voice name'}), 400

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_content = response.audio_content
        base64_audio = base64.b64encode(audio_content).decode('utf-8')

        return jsonify({'audio_content': base64_audio})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 4. Sign Language (Placeholder - needs implementation)
@app.route('/sign_language', methods=['POST'])
def generate_sign_language():
    data = request.get_json()
    text = data.get('text')

    # TODO: Implement sign language generation
    return jsonify({'sign_language_data': 'Sign language data (placeholder)'})

# 5. Social Media (Placeholder - needs implementation)
@app.route('/social_media', methods=['POST'])
def get_social_media():
    # TODO: Implement social media post retrieval
    return jsonify({'social_media_posts': 'Social media posts (placeholder)'})

# 6. Low Bandwidth Mode (Handled on the client-side)

# 7. Upload to Cloud Storage (Separate endpoint for Android)
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(audio_file.filename)  # Or generate a unique filename
        blob.upload_from_file(audio_file)

        audio_uri = f"gs://{BUCKET_NAME}/{blob.name}"
        return jsonify({'audio_uri': audio_uri})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # debug=True for development