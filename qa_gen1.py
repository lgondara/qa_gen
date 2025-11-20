import os
import json
import glob
from typing import List, Dict
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
# Point this to your documents folder
SOURCE_DOCS_DIR = "./raw_compliance_docs"
OUTPUT_FILE = "compliance_finetune_dataset.jsonl"

# Initialize OpenAI (or point to a local server like vLLM/Ollama)
client = OpenAI(api_key="YOUR_API_KEY_HERE")

# --- THE PROMPT STRATEGY ---
# This is the most critical part. We force the LLM to act as a
# Senior Compliance Officer creating training materials for juniors.
GENERATION_PROMPT = """
You are an expert Financial Crimes Compliance Officer. I will provide you with a section of a regulatory document or enforcement action.
Your task is to generate 3 distinct training examples based ONLY on this text.

The output must be a JSON object with a key "pairs" containing a list of 3 objects.
Each object must have: "instruction", "input" (optional context), and "output".

Create one example for each of these categories:
1. **Concept Definition:** Direct question about a rule or definition (e.g., "What is the Travel Rule?").
2. **Scenario Analysis:** Create a short hypothetical scenario based on the text and ask for a verdict (e.g., "A customer did X. Is this suspicious?").
3. **Fact Extraction:** Ask to extract specific red flags or penalties mentioned in the text.

**Input Text:**
{text_chunk}

**JSON Output Format:**
{{
  "pairs": [
    {{
      "instruction": "...",
      "input": "...",
      "output": "..."
    }}
  ]
}}
"""


def load_documents(directory: str) -> List[str]:
    """Reads all .txt files from the directory."""
    texts = []
    files = glob.glob(os.path.join(directory, "*.txt"))
    print(f"Found {len(files)} documents.")

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def chunk_text(text: str, chunk_size=2000) -> List[str]:
    """
    Splits text into manageable chunks for the API.
    Overlapping ensures context isn't lost at the edges.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


def generate_instruction_pairs(text_chunk: str) -> List[Dict]:
    """Sends the chunk to the Teacher LLM to generate training pairs."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Or "gpt-3.5-turbo" for lower cost
            response_format={"type": "json_object"},  # Enforces JSON structure
            messages=[
                {"role": "system", "content": "You are a data generation assistant for AML/KYC compliance."},
                {"role": "user", "content": GENERATION_PROMPT.format(text_chunk=text_chunk)}
            ],
            temperature=0.7  # Slight creativity for scenarios
        )

        # Parse the JSON response
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("pairs", [])

    except Exception as e:
        print(f"Error generating pairs: {e}")
        return []


def main():
    # 1. Load raw text
    raw_texts = load_documents(SOURCE_DOCS_DIR)

    all_pairs = []

    # 2. Process each document
    for i, doc_text in enumerate(raw_texts):
        print(f"Processing document {i + 1}...")

        # 3. Chunk the document
        chunks = chunk_text(doc_text)
        print(f"  - Split into {len(chunks)} chunks.")

        # 4. Generate pairs for each chunk
        for j, chunk in enumerate(chunks):
            # Skip very small chunks (likely footer junk)
            if len(chunk) < 300:
                continue

            print(f"  - Generating data from chunk {j + 1}...")
            pairs = generate_instruction_pairs(chunk)

            # Add source metadata (optional, helps debugging)
            for pair in pairs:
                pair['source_doc_id'] = i

            all_pairs.extend(pairs)

    # 5. Save to JSONL (Standard format for fine-tuning)
    print(f"Saving {len(all_pairs)} training examples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in all_pairs:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()