from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, concatenate_datasets
from pathlib import Path
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and tokenizer
model_path = Path(r"C:\Users\Sydney Parker\Phi-3-mini-4k-instruct")
try:
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    logging.info("Tokenizer loaded successfully!")

    logging.info("Loading model with eager attention implementation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    )
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Load the Intel/orca_dpo_pairs dataset and transform it
logging.info("Loading Intel/orca_dpo_pairs dataset...")
orca_dataset = load_dataset("Intel/orca_dpo_pairs", split="train[:1000]")
logging.info("Intel/orca_dpo_pairs dataset loaded successfully!")

# Transform orca_dpo_pairs into prompt-response format
logging.info("Transforming Intel/orca_dpo_pairs dataset...")
def transform_orca(example):
    return {
        "text": f"### Instruction: Answer the following question.\n{example['question']}\n### Response: {example['chosen']}"
    }
orca_dataset = orca_dataset.map(transform_orca, remove_columns=["system", "question", "chosen", "rejected"])
logging.info("Intel/orca_dpo_pairs dataset transformed successfully!")

# Load the transformed custom dataset
logging.info("Loading transformed custom dataset...")
custom_dataset = load_dataset("json", data_files="C:/Users/Sydney Parker/h2o-llmstudio/dataset_a_transformed.jsonl", split="train")
logging.info("Transformed custom dataset loaded successfully!")

# Combine the datasets
logging.info("Combining datasets...")
combined_dataset = concatenate_datasets([orca_dataset, custom_dataset])
logging.info(f"Combined dataset size: {len(combined_dataset)} examples")

# Inspect the combined dataset structure
logging.info("Inspecting combined dataset structure...")
print("First example from combined dataset:", combined_dataset[0])
logging.info("Dataset inspection complete!")

# Configure LoRA
logging.info("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.train()
for name, param in model.named_parameters():
    if param.requires_grad:
        logging.info(f"Trainable parameter: {name}")
logging.info("LoRA configured successfully!")

# Tokenize the combined dataset
logging.info("Tokenizing combined dataset...")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256, return_tensors="pt")

tokenized_dataset = combined_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
# Add labels for causal language modeling (shifted input_ids)
tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True)
logging.info("Combined dataset tokenized successfully!")

# Training arguments
logging.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="C:/Users/Sydney Parker/output_ethical",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    gradient_checkpointing=False,
    report_to="none"
)

# Train
logging.info("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)
trainer.train()
logging.info("Training completed successfully!")

# Save the fine-tuned model
logging.info("Saving fine-tuned model...")
model.save_pretrained("C:/Users/Sydney Parker/Phi-3-mini-Ethical")
tokenizer.save_pretrained("C:/Users/Sydney Parker/Phi-3-mini-Ethical")
logging.info("Model saved successfully!")