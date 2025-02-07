import spacy
from spacy.training import Example
import json
import pandas as pd
import re
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm") #Load a pre-trained model

def train_spacy_ner(training_data_file, output_model_path, n_iter=10, dropout=0.2):
    """Trains a custom spaCy NER model."""
    try:
        nlp = spacy.load("en_core_web_sm")

        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner")
        else:
            ner = nlp.get_pipe("ner")

        for label in ["PERSON", "GPE", "LOC"]:  # Add your labels
            ner.add_label(label)

        with open(training_data_file, "r", encoding="utf-8") as f:
            training_data = json.load(f)

        examples = []
        for item in training_data:
            text = item["text"]
            annotations = item.get("entities", [])
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, {"entities": annotations})
            examples.append(example)

        optimizer = nlp.begin_training()
        for i in range(n_iter):
            losses = {}
            for example in examples:
                nlp.update([example], sgd=optimizer, drop=dropout, losses=losses)
            print(f"Losses at iteration {i}: {losses}")

        nlp.to_disk(output_model_path)
        print(f"Trained model saved to {output_model_path}")

    except FileNotFoundError:
        print(f"Error: Training data file '{training_data_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{training_data_file}'.")
    except Exception as e:
        print(f"An error occurred during training: {e}")


def extract_cricket_events(transcript, ner_model):
    """Extracts cricket events from a transcript using the trained NER model."""
    events = []
    doc = ner_model(transcript) #Use the trained model

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            events.append({"event": "PLAYER", "entity": ent.text})
        elif ent.label_ == "GPE":
            events.append({"event": "LOCATION", "entity": ent.text})
        elif ent.label_ == "LOC":
            events.append({"event": "LOCATION", "entity": ent.text})
        # Add more logic to extract other events (FOUR, SIX, WICKET, etc.)
        # based on the entities and context in the commentary.
    return events


# Example Usage
csv_file = "data/ball_by_ball_data_commentary.csv"  # Replace with your CSV file
spacy_training_data_file = "data/spacy_training_data_ipl.json" #File to save the spacy format data
trained_model_path = "./models/cricket_ner_model"  # Path to save the trained model


# 2. Train the custom NER model
train_spacy_ner(spacy_training_data_file, trained_model_path, n_iter=15, dropout=0.3) #Adjust n_iter and dropout

# 3. Load the trained model and extract events
trained_ner_model = spacy.load(trained_model_path)
transcript = "Kohli scored a century at MCG. Bumrah bowled well. India won the match against Australia."
events = extract_cricket_events(transcript, trained_ner_model)
print(events)