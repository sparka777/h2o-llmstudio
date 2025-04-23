import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the fine-tuned model and tokenizer
model_path = "C:/Users/Sydney Parker/Phi-3-mini-Ethical"
try:
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Set pad_token_id to eos_token_id to avoid attention mask warning
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Tokenizer loaded successfully!")

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise Exception("Failed to load model")

class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: GenerateRequest):
    logger.info(f"Received request with prompt: {request.prompt}")
    try:
        # Tokenize the input prompt
        inputs = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]  # Length of the input prompt in tokens

        # Generate response
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=150,  # Increased to allow longer responses
                do_sample=False,  # Use greedy decoding to favor exact matches
                temperature=None,  # Not used with do_sample=False
                top_p=None,  # Not used with do_sample=False
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode the response, excluding the input prompt
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        logger.info("Response generated successfully")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")