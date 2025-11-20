import os
import json
import glob
import torch
import re
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Or "mistralai/Mistral-7B-Instruct-v0.3"
SOURCE_DIR = "./raw_compliance_docs"
OUTPUT_FILE = "compliance_finetune.jsonl"
CHUNK_SIZE = 2000  # Characters per chunk
OVERLAP = 200

# --- 1. SETUP MODEL (Local & Quantized) ---
print(f"Loading {MODEL_ID}...")

# 4-bit config to save memory (fits on ~6GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Ensure pad token is set (common issue with Llama)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# --- 2. HELPER FUNCTIONS (Replacements for LangChain) ---

def simple_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into chunks without external libraries.
    Preserves sentence boundaries roughly by finding nearest whitespace.
    """
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # If we are not at the end, try to break at a space/newline
        if end < text_len:
            # Look for the last space within the window to avoid cutting words
            lookback = text[start:end].rfind(' ')
            if lookback != -1 and lookback > (chunk_size * 0.8):
                end = start + lookback

        chunks.append(text[start:end])
        start = end - overlap  # Move pointer back for overlap

    return chunks


def clean_json_output(response_text: str):
    """
    LLMs often chat ("Here is your JSON:"). We need to extract just the JSON block.
    """
    # Try to find content between ```json and ```
    match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1)

    # If no markdown, try to find the first { and last }
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0)

    return response_text


# --- 3. THE PROMPT STRATEGY ---

def generate_synthetic_data(text_chunk: str):
    """
    Uses the local model to generate Q&A pairs from the chunk.
    """

    system_prompt = """You are a Financial Crime Compliance Specialist. 
    Your goal is to create a dataset for training an AI on AML (Anti-Money Laundering) and KYC rules.

    Based on the TEXT provided below, generate 2 training examples.
    Output MUST be a raw JSON object with a key "examples" containing a list.

    The structure of each example:
    {
        "instruction": "A specific question or task based on the text",
        "input": "Context details (if a scenario) or empty string (if a definition)",
        "output": "The correct compliance answer or verdict"
    }

    Ensure one example is a DEFINITION and one is a SCENARIO (detecting red flags).
    Do not output any text other than the JSON."""

    user_prompt = f"TEXT TO ANALYZE:\n{text_chunk}"

    # Apply the specific chat template for Llama/Mistral
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and skip the input tokens
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return response


# --- 4. MAIN EXECUTION LOOP ---

def main():
    files = glob.glob(os.path.join(SOURCE_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {SOURCE_DIR}")
        return

    valid_entries = []

    for filepath in files:
        print(f"Processing {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        chunks = simple_chunk_text(raw_text, CHUNK_SIZE, OVERLAP)
        print(f"--> Split into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            if len(chunk) < 300: continue  # Skip tiny chunks

            print(f"  --> Generating for chunk {i + 1}...")

            try:
                raw_response = generate_synthetic_data(chunk)
                json_str = clean_json_output(raw_response)

                data = json.loads(json_str)

                if "examples" in data:
                    for ex in data["examples"]:
                        # Validate structure
                        if "instruction" in ex and "output" in ex:
                            valid_entries.append(ex)
                            print(f"      [+] Added: {ex['instruction'][:50]}...")

            except json.JSONDecodeError:
                print("      [!] Failed to parse JSON from model response.")
            except Exception as e:
                print(f"      [!] Error: {e}")

    # --- 5. SAVE TO JSONL ---
    print(f"\nSaving {len(valid_entries)} pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()