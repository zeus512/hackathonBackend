from flask import Flask, request, jsonify, make_response, send_from_directory
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v2 as translate  # Import Cloud Translation
from google.cloud import texttospeech
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import os
import spacy
import uuid 
import json
import threading
from flask import Flask, request, jsonify, Response
from flask_sse import sse  # For SSE
from flask_cors import CORS
import time
import urllib.parse

app = Flask(__name__)
CORS(app, resources={r"/stream/*": {"origins": "*"}})
#CORS(app)  # Enable CORS for all routes
app.config["REDIS_URL"] = "redis://localhost:6379/0" #Configure redis url
app.register_blueprint(sse, url_prefix='/stream') #Register SSE blueprint

active_clients = set()
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

# Initialize the TTS client (do this *once* when the app starts)
try:
    tts_client = texttospeech.TextToSpeechClient()
except Exception as e:
    print(f"Error initializing TTS client: {e}")
    tts_client = None


AUDIO_DIRECTORY = "./data"  # Directory where your audio files are stored. Create this directory.
# Ensure the directory exists
os.makedirs(AUDIO_DIRECTORY, exist_ok=True)


AUDIO_GENERATED_DIRECTORY = "./data/generated_audio_files"  # Directory where your audio files are stored. Create this directory.
# Ensure the directory exists
os.makedirs(AUDIO_GENERATED_DIRECTORY, exist_ok=True)

manifests = {}  # Store manifests in memory (for simplicity)

vertexai.init(
    project="555187634505",
    location="us-central1",
    api_endpoint="us-central1-aiplatform.googleapis.com"
)
model = GenerativeModel(
    "projects/555187634505/locations/us-central1/endpoints/8278945424664952832",
)


def generate_manifest_from_files(audio_id):
    manifest = {"audio_chunks": []}
    i = 1
    while True:
        chunk_file = f"{audio_id}_chunk{i}.mp3"
        original_file = f"{audio_id}_chunk{i}.mp3"  # Assuming original and chunk filenames are the same initially
        chunk_path = os.path.join(AUDIO_GENERATED_DIRECTORY, chunk_file)

        if not os.path.exists(chunk_path):
            break

        manifest["audio_chunks"].append({
            "original_audio_filename": original_file,  # Add original filename HERE
            "translated_audio_filename": None,  # Placeholder
            "translated_audio_url": None,  # Placeholder
            "start_time": 0.0,  # Or get actual start/end times if available
            "end_time": 10.0
        })
        i += 1
    return manifest

# def generate_manifest_from_files(audio_id):
#     manifest = {"audio_chunks": []}
#     i = 1
#     while True:
#         chunk_file = f"{audio_id}_chunk{i}.mp3"
#         chunk_path = os.path.join(AUDIO_GENERATED_DIRECTORY, chunk_file)

#         if not os.path.exists(chunk_path):
#             break

#         manifest["audio_chunks"].append({
#             "translated_audio_url": None  # Placeholder
#         })
#         i += 1
#     return manifest

# def process_chunk(audio_id, chunk_index):
#     try:
#         print(f"Processing chunk {chunk_index} for {audio_id}")  # Add this line
#         chunk_file = f"{audio_id}_chunk{chunk_index + 1}.mp3"  # Adjust indexing if needed
#         translated_audio_url = f"/audio/mobile?filename={chunk_file}&generated=true"

#         manifests[audio_id]["audio_chunks"][chunk_index]["translated_audio_url"] = translated_audio_url

#         # Send SSE event to connected clients
#         print(f"translated_audio_url for chunk {chunk_index}: {translated_audio_url}") # Add this
#         sse.publish(json.dumps({'chunk_index': chunk_index, 'audio_url': translated_audio_url}), type=f'chunk_update_{audio_id}') #Use type to filter messages
#         print(f"Published SSE event for chunk {chunk_index} of {audio_id}")  # Add this

#     except Exception as e:
#         print(f"Error processing chunk {chunk_index}: {e}")
#         # ... (Handle error)

@app.route('/manifest/<audio_id>', methods=['GET'])
def get_manifest(audio_id):
    client_id = request.args.get('client_id', str(time.time()))  # You can use any unique client ID
    active_clients.add(client_id)  # Add client to active clients list
    if audio_id not in manifests:
        manifest = generate_manifest_from_files(audio_id)
        if manifest:
            manifests[audio_id] = manifest
        else:
            return jsonify({'error': 'Error generating manifest'}), 500
        print(f"Manifest for {audio_id}: {manifest}")
    return jsonify(manifests[audio_id])

# @app.route('/process_audio/<audio_id>', methods=['POST'])
# def process_audio_for_manifest(audio_id):
#     num_chunks = len(manifests[audio_id]["audio_chunks"])
#     for i in range(num_chunks):
#         print(f"I'm working till here")
#         process_chunk(audio_id, i) #Process the chunk and update manifest
#     return jsonify({"message": "Processing started"}), 200

processing_complete = {}  # Dictionary to track processing completion for each audio_id


@app.route('/process_audio/<audio_id>', methods=['POST'])
def process_audio_for_manifest(audio_id):
    if audio_id not in manifests:
        return jsonify({'error': 'Manifest not found'}), 404

    processing_complete[audio_id] = False  # Initialize the flag

    # Start processing in the background
    thread = threading.Thread(target=process_chunks_background, args=(audio_id,))
    thread.start()

    # Keep the request alive until processing is complete
    while not processing_complete[audio_id]:
        time.sleep(1)  # Check every second (adjust as needed)

    del processing_complete[audio_id] #Remove the flag once processing is complete
    return jsonify({"message": "Processing complete"}), 200 # Return a success message

def process_chunks_background(audio_id):
    manifest = manifests.get(audio_id)
    if manifest:
        for i, chunk in enumerate(manifest["audio_chunks"]):
            with app.app_context():
                original_file = chunk["original_audio_filename"]  # Get original filename
                translated_file = original_file.replace(".mp3", "_te.mp3")  # Create translated filename
                translated_filepath = os.path.join(AUDIO_GENERATED_DIRECTORY, translated_file)

                # Check if the translated file already exists (for resuming)
                if not os.path.exists(translated_filepath):
                    try:
                        # 1. Transcription (Pipeline Step 1)
                        transcript_data = transcribe_audio_local(os.path.join(AUDIO_DIRECTORY, original_file), "en-UK")
                        if isinstance(transcript_data, tuple):
                            print(f"Transcription error for {original_file}: {transcript_data[1]}")
                            continue  # Skip to the next file

                        transcript = transcript_data.get('transcript')

                        # 2. Translation (Pipeline Step 2)
                        translated_transcript = translate_cricket_commentary(transcript, "te")

                        # 3. TTS (Pipeline Step 3)
                        tts_data = generate_speech_local(translated_transcript, "te-IN")
                        if isinstance(tts_data, tuple):
                            print(f"TTS error for {original_file}: {tts_data[1]}")
                            continue  # Skip to the next file

                        audio_url = tts_data.get('audio_url')
                         # Extract the unique filename from the audio_url (if it's from your server)
                        parsed_url = urllib.parse.urlparse(audio_url)
                        query_params = urllib.parse.parse_qs(parsed_url.query)
                        unique_filename = query_params.get('filename', [None])[0]

                        if unique_filename:
                            # Rename the file (if necessary - depends on your file storage)
                            source_path = os.path.join(AUDIO_GENERATED_DIRECTORY, unique_filename) #Use unique name to find audio
                            if os.path.exists(source_path): #If file exists then rename
                                os.rename(source_path, translated_filepath)

                            # ... (rest of the processing logic)
                        else:
                            print("Filename not found in audio_url")
                            continue # Skip to the next file


                        # ... (Your download code here if needed.  If TTS saves directly, skip this)

                    except Exception as e:
                        print(f"Error processing {original_file}: {e}")
                        continue  # Skip to the next file

                process_chunk(audio_id, i, translated_file)  # Pass translated_file to process_chunk

def process_chunk(audio_id, chunk_index, translated_file):  # Add translated_file parameter
    with app.app_context():
        try:
            print(f"Processing chunk {chunk_index} for {audio_id}")
            translated_audio_url = f"/audio/mobile?filename={translated_file}&generated=true"  # Use translated_file

            manifests[audio_id]["audio_chunks"][chunk_index]["translated_audio_url"] = translated_audio_url

            print(f"translated_audio_url for chunk {chunk_index}: {translated_audio_url}")
            sse.publish(json.dumps({'chunk_index': chunk_index, 'audio_url': translated_audio_url}), type=f'chunk_update_{audio_id}')
            print(f"Published SSE event for chunk {chunk_index} of {audio_id}")

        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {e}")
            # ... (Handle error)


def translate_cricket_commentary(english_text, target_language):
    """Translates cricket commentary using a chat-based approach.

    Args:
        english_text: The English cricket commentary text.
        target_language: The target language code (e.g., "te", "hi", "ta").

    Returns:
        The translated commentary, or an error message.
    """

    chat = model.start_chat()  # Start a chat session

    prompt = f"""You are an expert in translating cricket commentary from English to {target_language}. You should use terminology commonly used in {target_language} cricket commentary. You must preserve the meaning and provide accurate translations. Output ONLY the {target_language} translation. Do not include any other languages or explanations."""

    full_prompt = f"{prompt}\n\n{english_text}"

    generation_config = {
        "max_output_tokens": 8192,  # Adjust as needed
        "temperature": 1,         # Adjust as needed
        "top_p": 0.95,          # Adjust as needed
    }

    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF
        ),
    ]

    try:
        response = chat.send_message(  # Use chat.send_message()
            full_prompt,  # Send the combined prompt and text
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        translation = response.text.strip()
        return translation
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/manifest', methods=['POST'])
def generate_manifest():
    data = request.get_json()
    original_audio_files = data.get('audio_files')

    if not original_audio_files or not isinstance(original_audio_files, list):
        return jsonify({'error': 'Missing or invalid "audio_files" in request'}), 400

    manifest = {"audio_chunks": []}

    for original_file in original_audio_files:
        try:
            translated_file = original_file.replace(".mp3", "_te.mp3")  # Example naming (improve if needed)
            translated_filepath = os.path.join(AUDIO_GENERATED_DIRECTORY, translated_file)

            if not os.path.exists(translated_filepath):  # Only generate if it doesn't exist
                # 1. Transcription (if needed - depends on your workflow)
                transcript_data = transcribe_audio_local(os.path.join(AUDIO_DIRECTORY, original_file), "en-UK") # Transcribe to English first
                if isinstance(transcript_data, tuple): #Check for error
                    return jsonify({'error': transcript_data[1]}), transcript_data[0] #Return error
                transcript = transcript_data.get('transcript')

                # 2. Translation
                translated_transcript = translate_cricket_commentary(transcript, "te")  # Translate to Telugu

                # 3. TTS
                tts_data = generate_speech_local(translated_transcript, "te-IN") #Generate speech in Telugu
                if isinstance(tts_data, tuple): #Check for error
                    return jsonify({'error': tts_data[1]}), tts_data[0] #Return error
                audio_url = tts_data.get('audio_url')

                # # Download the audio from the URL and save it locally
                # import requests
                # response = requests.get(audio_url)
                # if response.status_code == 200:
                #     with open(translated_filepath, "wb") as f:
                #         f.write(response.content)
                # else:
                #     return jsonify({'error': f"Error downloading translated audio: {response.status_code}"}), 500

            # 4. Add chunk info to the manifest
            manifest["audio_chunks"].append({
                "original_audio_filename": original_file,
                "translated_audio_filename": translated_file,
                "start_time": 0.0,  # Or get the actual start time if available
                "end_time": 10.0   # Or get the actual end time if available
            })

        except Exception as e:
            return jsonify({'error': f'Error processing {original_file}: {str(e)}'}), 500

    return jsonify(manifest)

# Helper functions for local transcription and TTS
def transcribe_audio_local(audio_file_path, language_code):
    """Transcribes audio using Cloud Speech-to-Text (local file path)."""
    try:
        client = speech.SpeechClient()

        with open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        # ... (rest of your transcription code - same as in /transcribe)
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
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="default",  # Try different models
            #enhanced_model=True,  # Try enabling enhanced models
            # speech_contexts=[speech.SpeechContext(phrases=["common phrase 1", "common phrase 2"])] #Add common phrases
        )
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        transcript_with_timestamps = []
        for result in response.results:
             for alternative in result.alternatives:
                start_time = result.result_end_time.total_seconds() - sum([x.end_time.total_seconds() for x in result.alternatives if x != alternative]) #Start time of the sentence
                end_time = result.result_end_time.total_seconds() #End time of the sentence
                transcript_with_timestamps.append({"transcript": alternative.transcript, "start_time": start_time, "end_time": end_time})
        for result in response.results:
            for alternative in result.alternatives:
                transcript += alternative.transcript + " " # Concatenate alternatives

        return {'transcript': transcript, 'transcript_with_timestamps': transcript_with_timestamps}

    except Exception as e:
        return 500, str(e)  # Return error code and message

def generate_speech_local(text, language_code):
    """Generates speech from text using Cloud Text-to-Speech (local save)."""
    try:
        tts_client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)

        available_voices = list_voices(language_code)
        if available_voices:
            # Choose a voice (e.g., the first one)
            voice_name = list(available_voices.keys())[0]  # Get the first voice name
            actual_language_code = available_voices[voice_name]["language_codes"][0] #Get New language code
            print(f"Using default voice: {voice_name} for language {language_code}")

        voice = texttospeech.VoiceSelectionParams(
            language_code=actual_language_code,
            name=voice_name  # Optional: specify voice
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,  # Or LINEAR16, WAV, etc.
            speaking_rate = 1.0
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_content = response.audio_content

        unique_filename = str(uuid.uuid4()) + ".mp3"
        local_file_path = os.path.join("./data/generated_audio_files", unique_filename)

        with open(local_file_path, "wb") as f:
            f.write(audio_content)

        audio_url = f"/audio/mobile?filename={unique_filename}&generated=true"

        return {'audio_url': audio_url}

    except Exception as e:
        return 500, str(e)


@app.route('/audio/mobile', methods=['GET'])
def player_audio():
    """Serves the requested audio file."""
    filename = request.args.get('filename')
    generated = request.args.get('generated', 'false').lower() == 'true'

    if not filename:
        return jsonify({'error': 'Missing "filename" query parameter'}), 400

    path = AUDIO_GENERATED_DIRECTORY if generated else AUDIO_DIRECTORY  # Correct path selection

    filepath = os.path.join(path, filename)  # Path join for security

    try:  # Include file existence check in try block
        if not os.path.exists(filepath):  # Check if file exists *before* sending
            return jsonify({'error': 'Audio file not found'}), 404
        return send_from_directory(path, filename)  # Use the selected path

    except Exception as e:
        return jsonify({'error': f'Error serving audio file: {str(e)}'}), 500



@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribes the audio file using Cloud Speech-to-Text (local dev)."""
    data = request.get_json()
    audio_file_path = data.get('audio_uri')
    language_code = data.get('language_code')
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
            language_code=language_code,
            enable_automatic_punctuation=True,
            model="default",  # Try different models
            #enhanced_model=True,  # Try enabling enhanced models
            # speech_contexts=[speech.SpeechContext(phrases=["common phrase 1", "common phrase 2"])] #Add common phrases
        )
        response = client.recognize(config=config, audio=audio)
        transcript = ""
        transcript_with_timestamps = []
        for result in response.results:
             for alternative in result.alternatives:
                start_time = result.result_end_time.total_seconds() - sum([x.end_time.total_seconds() for x in result.alternatives if x != alternative]) #Start time of the sentence
                end_time = result.result_end_time.total_seconds() #End time of the sentence
                transcript_with_timestamps.append({"transcript": alternative.transcript, "start_time": start_time, "end_time": end_time})
        for result in response.results:
            for alternative in result.alternatives:
                transcript += alternative.transcript + " " # Concatenate alternatives

        return jsonify({'transcript': transcript, 'transcript_with_timestamps': transcript_with_timestamps}), 200

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
    target_languages = data.get('target_languages')

    if not text:
        return jsonify({'error': 'Missing "text" in request'}), 400

    if not target_languages:
        return jsonify({'error': 'Missing "target_languages" in request'}), 400

    if not isinstance(target_languages, list):
        return jsonify({'error': '"target_languages" must be a list'}), 400

    translations = {}

    for lang in target_languages:
        translations[lang] = translate_cricket_commentary(text, lang)  # Use the new function

    return jsonify({'translations': translations}), 200

@app.route('/translate_old', methods=['POST'])
def translate_text_old():
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


def list_voices(language_code=None):
    """Lists the available voices for Cloud Text-to-Speech."""

    client = texttospeech.TextToSpeechClient()

    request = texttospeech.ListVoicesRequest(language_code=language_code)

    response = client.list_voices(request=request)

    voices = {}
    for voice in response.voices:
        languages = voice.language_codes
        name = voice.name
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        natural_sample_rate_hertz = voice.natural_sample_rate_hertz

        voices[name] = {
            "language_codes": languages,
            "gender": gender,
            "natural_sample_rate_hertz": natural_sample_rate_hertz,
        }

    return voices

@app.route('/tts', methods=['POST'])
def generate_speech():
    """Endpoint to generate speech from text."""
    data = request.get_json()
    text = data.get('text')
    language_code = data.get('language_code', 'en-US')  # Default to US English
    voice_name = data.get('voice_name')  # Optional: specify voice
    speaking_rate = data.get('speaking_rate', 1.0) #Optional: speaking rate
    actual_language_code = language_code
    if not text:
        return jsonify({'error': 'Missing "text" in request'}), 400

    if tts_client is None:
        return jsonify({'error': 'TTS client not initialized'}), 500

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
    
        if not voice_name:  # If voice_name is not provided, try to find a default
            available_voices = list_voices(language_code)
            if available_voices:
                # Choose a voice (e.g., the first one)
                voice_name = list(available_voices.keys())[0]  # Get the first voice name
                actual_language_code = available_voices[voice_name]["language_codes"][0] #Get New language code
                print(f"Using default voice: {voice_name} for language {language_code}")


        voice = texttospeech.VoiceSelectionParams(
            language_code=actual_language_code,
            name=voice_name  # Optional: specify voice
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,  # Or LINEAR16, WAV, etc.
            speaking_rate = speaking_rate
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        audio_content = response.audio_content
        # You can now save audio_content to a file or stream it directly
         # Generate a unique filename (to avoid collisions)
        unique_filename = str(uuid.uuid4()) + ".mp3"  # Or appropriate extension
        local_file_path = os.path.join("./data/generated_audio_files", unique_filename)  # Path to save locally - make sure this directory exists

        # Save the audio content to a local file
        with open(local_file_path, "wb") as f:
            f.write(audio_content)

        # Construct the URL (for local development, this will be a file path)
        audio_url = f"/audio/mobile?filename={unique_filename}&generated=true"  # Adjust path as needed

        # uncomment for cloud storage usage
        # # Create a Cloud Storage blob (file)
        # blob = bucket.blob(unique_filename)

        # # Upload the audio content to Cloud Storage
        # blob.upload_from_string(audio_content)

        # # Get the public URL of the blob
        # audio_url = blob.public_url

        return jsonify({'audio_url': audio_url}), 200
        #return jsonify({'audio_content': audio_content.decode('ISO-8859-1')}), 200 #Returns base64 encoded string

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process_text', methods=['POST'])
def process_text():
    """Processes the transcript (event extraction, translation)."""
    data = request.get_json()
    transcript = data.get('transcript')
    target_languages = data.get('target_languages', ['en'])
    translate_transcript = data.get('translate_transcript', True)
    translate_events = data.get('translate_events', False)

    if not transcript:
        return jsonify({'error': 'Missing "transcript" in request'}), 400

    try:
        # 2. Event Extraction
        if trained_ner_model is None:
            return jsonify({'error': 'spaCy model not loaded'}), 500

        events = extract_cricket_events(transcript, trained_ner_model)

        # 3. Translation (same as before)
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
        response = make_response(jsonify({'events': events, 'translations': translations}), 200)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

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
        response = make_response(jsonify({'transcript': transcript, 'events': events, 'translations': translations}), 200)
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
        #return jsonify({'transcript': transcript, 'events': events, 'translations':translations}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/clear_all_sse_sessions', methods=['POST'])
def clear_all_sse_sessions():
    # Clear all active SSE sessions by disconnecting all clients
    for client_id in list(active_clients):
        print(f"Disconnecting client {client_id}")
        # You can use a custom method to notify or disconnect the client if needed.
        # For now, we are simply removing from the active_clients set.
        active_clients.remove(client_id)

    return jsonify({"message": "All SSE client sessions cleared."})

with app.app_context():
    clear_all_sse_sessions()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='3000', debug=True) # debug=True for local development only