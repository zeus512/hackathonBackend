import pandas as pd
import re
import json

# Define file paths
input_file_path = ".data/ball_by_ball_data_commentary.csv"
output_file_path = "spacy_training_data.json"

# Load the CSV file
df = pd.read_csv(input_file_path)

# Extract relevant columns
commentary_texts = df["commentary"].astype(str).tolist()
bowler_batsman_texts = df["bowler_to_batsman, runs"].astype(str).tolist()

# Function to extract player names from "bowler_to_batsman" text
def extract_players(text):
    match = re.match(r"([\w\s]+) to ([\w\s]+),", text)
    if match:
        bowler, batsman = match.groups()
        return bowler.strip(), batsman.strip()
    return None, None

# Prepare training data in spaCy format
training_data = []
for commentary, bowler_batsman in zip(commentary_texts, bowler_batsman_texts):
    bowler, batsman = extract_players(bowler_batsman)
    entities = []
    
    # Find player mentions in commentary and mark them as PERSON entities
    if bowler and bowler in commentary:
        start_idx = commentary.find(bowler)
        end_idx = start_idx + len(bowler)
        entities.append((start_idx, end_idx, "PERSON"))
    
    if batsman and batsman in commentary:
        start_idx = commentary.find(batsman)
        end_idx = start_idx + len(batsman)
        entities.append((start_idx, end_idx, "PERSON"))
    
    if entities:
        training_data.append({"text": commentary, "entities": entities})

# Save the formatted training data as a JSON file
with open(output_file_path, "w") as f:
    json.dump(training_data, f, indent=4)

print(f"Training data saved to {output_file_path}")
