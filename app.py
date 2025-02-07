from flask import Flask, request, jsonify
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v2 as translate  # Import Cloud Translation

import os
import spacy

app = Flask(__name__)

# Load the trained spaCy model (do this *once* when the app starts)
try:
    trained_model_path = "./models/cricket_ner_model"  # Path to your trained model
    trained_ner_model = spacy.load(trained_model_path)
except OSError as e:
    print(f"Error loading spaCy model: {e}")
    trained_ner_model = None  # Handle the case where the model can't be loaded

# Initialize the translation client (do this *once* when the app starts)
try:
    translate_client = translate.Client()  # No explicit credentials needed if set up correctly
except Exception as e:
    print(f"Error initializing translation client: {e}")
    translate_client = None


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes the audio file using Cloud Speech-to-Text (local dev)."""
    data = request.get_json()
    audio_file_path = data.get('audio_uri')
    try: 
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
            encoding=encoding,
            sample_rate_hertz=sample_rate_hertz,
            language_code="en-UK",
            enable_automatic_punctuation=True,
            model="default",  # Try different models
            #enhanced_model=True,  # Try enabling enhanced models
            # speech_contexts=[speech.SpeechContext(phrases=["common phrase 1", "common phrase 2"])] #Add common phrases
        )
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            for alternative in result.alternatives:
                transcript += alternative.transcript + " " # Concatenate alternatives

        return jsonify({'transcript': transcript}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error details (dev only)

def extract_cricket_events(transcript, ner_model):
    """Extracts cricket events from a transcript using the trained NER model."""
    events = []
    doc = ner_model(transcript)  # Use the trained model

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            events.append({"event": "PLAYER", "entity": ent.text})
        elif ent.label_ == "GPE":
            events.append({"event": "TEAM", "entity": ent.text})  # Or LOCATION if needed
        elif ent.label_ == "LOC":
            events.append({"event": "LOCATION", "entity": ent.text})

    # Add more logic to extract other events (FOUR, SIX, WICKET, etc.)
    # based on the entities and context in the commentary.  This is where
    # you will need to add more sophisticated rules and potentially
    # use dependency parsing.

    # Example (very basic - needs improvement):
    for token in doc:
        if token.text.lower() == "four":
            events.append({"event": "FOUR"})  # Needs player context
        elif token.text.lower() == "six":
            events.append({"event": "SIX"})  # Needs player context
        elif token.text.lower() == "out":
            events.append({"event": "DISMISSAL"})  # Needs player context
        # ... more rules

    return events

@app.route('/extract_events', methods=['POST'])
def extract_events_endpoint():
    """Endpoint to extract cricket events from a transcript."""
    data = request.get_json()
    transcript = data.get('transcript')

    if not transcript:
        return jsonify({'error': 'Missing "transcript" in request'}), 400

    if trained_ner_model is None:  # Check if the model loaded successfully
        return jsonify({'error': 'spaCy model not loaded'}), 500

    try:
        events = extract_cricket_events(transcript, trained_ner_model)
        return jsonify({'events': events}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate_text():
    """Endpoint to translate text to multiple languages."""
    data = request.get_json()
    text = data.get('text')
    target_languages = data.get('target_languages')  # List of target language codes

    if not text:
        return jsonify({'error': 'Missing "text" in request'}), 400

    if not target_languages:
        return jsonify({'error': 'Missing "target_languages" in request'}), 400

    if not isinstance(target_languages, list):  # Ensure it's a list
      return jsonify({'error': '"target_languages" must be a list'}), 400

    if translate_client is None:
        return jsonify({'error': 'Translation client not initialized'}), 500

    translations = {}  # Dictionary to store translations for each language

    try:
        for lang in target_languages:
            try:  # Handle individual translation errors
                result = translate_client.translate(text, target_language=lang)
                translations[lang] = result['translatedText']
            except Exception as e:
                translations[lang] = f"Error: {str(e)}"  # Store error message
        return jsonify({'translations': translations}), 200 #Returns a dictionary of translations

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Combines transcription, event extraction, and translation."""
    data = request.get_json()
    audio_file_path = data.get('audio_uri')
    target_languages = data.get('target_languages', ['en']) #Default to English
    translate_transcript = data.get('translate_transcript', True) #Translate the entire transcript by default
    translate_events = data.get('translate_events', False) #Translate the extracted events by default

    try:
        # 1. Transcription
        client = speech.SpeechClient()
        with open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        # ... (speech recognition config - same as in /transcribe)

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
            encoding=encoding,
            sample_rate_hertz=sample_rate_hertz,
            language_code="en-UK",
            enable_automatic_punctuation=True,
            model="default",  # Try different models
            #enhanced_model=True,  # Try enabling enhanced models
            # speech_contexts=[speech.SpeechContext(phrases=["common phrase 1", "common phrase 2"])] #Add common phrases
        )
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            for alternative in result.alternatives:
                transcript += alternative.transcript + " "

        # 2. Event Extraction
        if trained_ner_model is None:
            return jsonify({'error': 'spaCy model not loaded'}), 500

        events = extract_cricket_events(transcript, trained_ner_model)

        # 3. Translation
        translations = {}

        if translate_transcript:
            transcript_translations = {}
            for lang in target_languages:
              try:
                result = translate_client.translate(transcript, target_language=lang)
                transcript_translations[lang] = result['translatedText']
              except Exception as e:
                transcript_translations[lang] = f"Error: {str(e)}"
            translations['transcript'] = transcript_translations

        if translate_events:
            event_translations = {}
            for event in events:
                event_type = event.get('event')
                entity = event.get('entity')
                if entity: #Translate the entity
                    entity_translations = {}
                    for lang in target_languages:
                      try:
                        result = translate_client.translate(entity, target_language=lang)
                        entity_translations[lang] = result['translatedText']
                      except Exception as e:
                        entity_translations[lang] = f"Error: {str(e)}"
                    if event_type not in event_translations:
                        event_translations[event_type] = {}
                    event_translations[event_type][entity] = entity_translations

            translations['events'] = event_translations

        return jsonify({'transcript': transcript, 'events': events, 'translations':translations}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)  # debug=True for local development only