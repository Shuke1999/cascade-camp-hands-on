# Re-process using raw line parsing to avoid pandas parsing issues

from collections import defaultdict
import json

# Load lines directly
with open("/Users/shuke/Desktop/cascade-camp-hands-on/data/topres19th/HIPE-2022-v2.1-topres19th-train-en.tsv", "r", encoding="utf-8") as f:
    lines = f.readlines()

documents = []
current_doc_id = None
tokens_with_space = []
bio_tokens = []
bio_labels = []

# Helper to extract entities
def extract_entities(tokens, labels):
    entities = defaultdict(list)
    entity_tokens = []
    current_label = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if entity_tokens:
                entities[current_label].append(" ".join(entity_tokens))
                entity_tokens = []
            current_label = label[2:]
            entity_tokens = [token]
        elif label.startswith("I-") and current_label == label[2:]:
            entity_tokens.append(token)
        else:
            if entity_tokens:
                entities[current_label].append(" ".join(entity_tokens))
                entity_tokens = []
            current_label = None
    if entity_tokens:
        entities[current_label].append(" ".join(entity_tokens))
    return entities

for line in lines:
    line = line.strip()

    if line.startswith("# hipe2022:document_id = "):
        if current_doc_id is not None:
            full_text = "".join(tokens_with_space)
            entities = extract_entities(bio_tokens, bio_labels)
            documents.append({
                "document_id": current_doc_id,
                "text": full_text.strip(),
                "entities": dict(entities)
            })
        current_doc_id = line.split("= ")[-1]
        tokens_with_space = []
        bio_tokens = []
        bio_labels = []
        continue

    if line.startswith("#") or line.strip() == "":
        continue

    parts = line.split("\t")
    if len(parts) < 2:
        continue  # skip malformed

    token = parts[0]
    label = parts[1] if len(parts) > 1 else "O"
    misc = parts[-1] if len(parts) == 10 else ""

    if "NoSpaceAfter" in misc:
        tokens_with_space.append(token)
    else:
        tokens_with_space.append(token + " ")

    bio_tokens.append(token)
    bio_labels.append(label)

# Save last document
if current_doc_id is not None:
    full_text = "".join(tokens_with_space)
    entities = extract_entities(bio_tokens, bio_labels)
    documents.append({
        "document_id": current_doc_id,
        "text": full_text.strip(),
        "entities": dict(entities)
    })

# Save to JSON
final_output_path = "/Users/shuke/Desktop/cascade-camp-hands-on/data/topres19th/HIPE-prep.json"
with open(final_output_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)
