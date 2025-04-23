import json
import re
from pathlib import Path

# File paths
input_file = Path(r"C:\Users\Sydney Parker\h2o-llmstudio\dataset_a_updated_fixed.jsonl")
output_file = Path(r"C:\Users\Sydney Parker\h2o-llmstudio\dataset_a_transformed.jsonl")

# Check if the input file exists
if not input_file.exists():
    raise FileNotFoundError(f"Input file {input_file} does not exist.")

# Function to extract the question from the text field (for deduplication)
def extract_question(text):
    match = re.search(r"### Instruction: Answer the following question.\n(.*?)\n### Response:", text, re.DOTALL)
    return match.group(1).strip() if match else text

# Read the entire file content (since it's a single JSON object)
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read().strip()

# Parse the single JSON object
try:
    data_obj = json.loads(content)
except json.JSONDecodeError as e:
    print(f"Invalid JSON in file: {e}")
    raise

# Extract entries from the "data" field
entries = []
seen_questions = set()

if "data" in data_obj:
    for item in data_obj["data"]:
        if "text" in item:
            # Already in the desired format
            text = item["text"]
            question = extract_question(text)
            if question not in seen_questions:
                seen_questions.add(question)
                entries.append({"text": text})
        elif "input" in item and "output" in item:
            # Transform input/output format into text format
            input_text = item["input"]
            output_text = item["output"]
            text = f"### Instruction: Answer the following question.\n{input_text}\n### Response: {output_text}"
            question = input_text
            if question not in seen_questions:
                seen_questions.add(question)
                entries.append({"text": text})
        else:
            print(f"Skipping invalid entry: {item}")
else:
    print("No 'data' field found in the JSON object")
    raise ValueError("Expected a 'data' field in the JSON object")

print(f"Number of unique entries: {len(entries)}")

# Save the transformed dataset in .jsonl format
with open(output_file, "w", encoding="utf-8") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")

print(f"Transformed dataset saved to {output_file}")