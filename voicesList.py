from google.cloud import texttospeech

def list_voices(language_code=None):
    """Lists the available voices for Cloud Text-to-Speech."""

    client = texttospeech.TextToSpeechClient()

    request = texttospeech.ListVoicesRequest(language_code=language_code)  # Optional: filter by language

    response = client.list_voices(request=request)

    voices = {}
    for voice in response.voices:
        languages = voice.language_codes
        name = voice.name
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        natural_sample_rate_hertz = voice.natural_sample_rate_hertz
        # supported_encodings = []  # Initialize an empty list
        # for encoding in voice.supported_encodings: # Iterate through the repeated field
        #     supported_encodings.append(texttospeech.AudioEncoding(encoding).name)

        voices[name] = {
            "language_codes": languages,
            "gender": gender,
            "natural_sample_rate_hertz": natural_sample_rate_hertz,
            #"supported_encodings": supported_encodings,
        }

    return voices

# ... (rest of the code - example usage - remains the same)
# Example usage (all languages):
all_voices = list_voices()
for voice_name, details in all_voices.items():
  print(f"Voice Name: {voice_name}")
  print(f"  Languages: {details['language_codes']}")
  print(f"  Gender: {details['gender']}")
  print(f"  Sample Rate: {details['natural_sample_rate_hertz']}")
  #print(f"  Encodings: {details['supported_encodings']}")
  print("-" * 20)

# Example usage (for a specific language):
english_voices = list_voices("en-US")
for voice_name, details in english_voices.items():
    print(f"Voice Name: {voice_name}")
    # ... (print other details)