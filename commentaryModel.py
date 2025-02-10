import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part

import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting

def translate_cricket_commentary(english_text, target_language):
    """Translates cricket commentary using a chat-based approach.

    Args:
        english_text: The English cricket commentary text.
        target_language: The target language code (e.g., "te", "hi", "ta").

    Returns:
        The translated commentary, or an error message.
    """

    vertexai.init(
        project="555187634505",
        location="us-central1",
        api_endpoint="us-central1-aiplatform.googleapis.com"
    )
    model = GenerativeModel(
        "projects/555187634505/locations/us-central1/endpoints/8278945424664952832",
    )

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

# Example Usage:
english_text = """set for play to start and here's Ashley  fade away some swing  down the next side and he's off and away and the typical Steve Smith shot hit on the up.  Let's hit hard very hard.  second boundary for Smith another one  that's the best of the Lord that West away total control.  That's pulled away.  mother shot delivery  Put away by shot.  It's trash that through the offside.  He's used the pace here to good affect one bounce over the room.  when I start to start off driving through the covers and it's about boundary for Josh English and face on  and it goes to the  boundary  again in the gap again missing his language in his line.  And many balls- won outside the has been plattered over point for a 6.  Goes for the slower one.  It's very nicely in other.  languages  22 runs coming  Up, it's a feeling of the deep. That's well placed again from Joss English  Lot better it's got a rational stryker's egg. Oh my god.  This is it completely.  a little bit wicked number two for India  oh  that is  zinc piece of timing  the today good shot"""

target_language = "telugu"  # Or "hi", "ta", "bn", etc.

translated_commentary = translate_cricket_commentary(english_text, target_language)
print(translated_commentary)
