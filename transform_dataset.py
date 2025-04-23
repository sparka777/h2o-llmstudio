import json
from pathlib import Path

# Load the original dataset
input_file = Path(r"C:\Users\Sydney Parker\h2o-llmstudio\dataset_a_updated_fixed.jsonl")
output_file = Path(r"C:\Users\Sydney Parker\h2o-llmstudio\dataset_a_transformed.jsonl")

# Read and deduplicate entries
entries = []
seen_inputs = set()
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line.strip())
        input_text = entry["input"]
        if input_text not in seen_inputs:
            seen_inputs.add(input_text)
            entries.append(entry)

print(f"Number of unique entries: {len(entries)}")

# Transform entries into prompt-response format
transformed_entries = []
for entry in entries:
    input_text = entry["input"]
    output_text = entry["output"]
    # Format as a single text field with instruction and response
    text = f"### Instruction: Answer the following question.\n{input_text}\n### Response: {output_text}"
    transformed_entries.append({"text": text})

# Save the transformed dataset
with open(output_file, "w", encoding="utf-8") as f:
    for entry in transformed_entries:
        f.write(json.dumps(entry) + "\n")

print(f"Transformed dataset saved to {output_file}")