from google.cloud import language_v1

# Initialize the Google NLP client
client = language_v1.LanguageServiceClient()

def analyze_text(text):
    # Prepare the document
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Detect entities
    entities = client.analyze_entities(document=document).entities
    events = []

    # Extract entities (such as players and actions)
    for entity in entities:
        if entity.type == language_v1.Entity.Type.PERSON:
            events.append({"player": entity.name, "type": "PERSON"})
        elif entity.type == language_v1.Entity.Type.LOCATION:
            events.append({"location": entity.name, "type": "LOCATION"})
        elif entity.type == language_v1.Entity.Type.EVENT:
            events.append({"event": entity.name, "type": "EVENT"})
    
    # Analyze syntax to identify key actions and relationships
    syntax = client.analyze_syntax(document=document)
    for token in syntax.tokens:
        print(f"Token: {token.text.content}, Lemma: {token.lemma}, Dep: {token.dependency_edge.dep_}, POS: {token.part_of_speech.tag_}")
    
    return events

# Example usage
transcript = """
Kohli scored 50 runs. Dhawan was dismissed by Cummins. It's a great day at MCG!
India is leading the match. Kohli hit a six, and now the team is building a solid partnership.
"""
events = analyze_text(transcript)
print(events)
