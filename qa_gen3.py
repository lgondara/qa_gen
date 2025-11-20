"""
Compliance QA Pair Generator - Hugging Face Edition
Single-file program to generate question-answer pairs from financial compliance data

Usage:
    python compliance_qa_huggingface.py --input data/combined/compliance_dataset_train.jsonl --output compliance_qa_pairs.jsonl

    # Or process specific files
    python compliance_qa_huggingface.py --input data/vanguard/vanguard_raw_content.json --output vanguard_qa.jsonl

    # Specify model
    python compliance_qa_huggingface.py --input data.json --output qa.jsonl --model mistralai/Mistral-7B-Instruct-v0.2

Requirements:
    pip install transformers torch accelerate sentencepiece

Recommended Models (choose based on your hardware):
    - mistralai/Mistral-7B-Instruct-v0.2 (7B, needs ~16GB GPU)
    - meta-llama/Llama-2-7b-chat-hf (7B, needs ~16GB GPU)
    - microsoft/Phi-3-mini-4k-instruct (3.8B, needs ~8GB GPU)
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, can run on CPU)
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re
from dataclasses import dataclass, asdict
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComplianceQAPair:
    """Represents a compliance-focused QA pair"""
    question: str
    answer: str
    context: str
    compliance_topic: str  # e.g., "disclosure", "suitability", "conflicts"
    difficulty: str  # easy, medium, hard
    source_document: str

    def to_training_format(self) -> Dict:
        """Convert to instruction-tuning format"""
        return {
            "instruction": "You are a financial compliance expert. Answer the following question based on the provided regulatory context.",
            "input": f"Context: {self.context}\n\nQuestion: {self.question}",
            "output": self.answer,
            "metadata": {
                "compliance_topic": self.compliance_topic,
                "difficulty": self.difficulty,
                "source": self.source_document
            }
        }


class ComplianceDataLoader:
    """Load compliance data from various formats"""

    @staticmethod
    def load_data(filepath: str) -> List[Dict]:
        """Load compliance data from JSON or JSONL file"""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Handle JSONL
        if path.suffix == '.jsonl':
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        # Handle JSON
        elif path.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)

                # Handle different JSON structures
                if isinstance(content, list):
                    return content
                elif isinstance(content, dict):
                    # If it's wrapped in a key
                    if 'pages' in content:
                        return content['pages']
                    elif 'data' in content:
                        return content['data']
                    else:
                        return [content]

        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def extract_text(item: Dict) -> str:
        """Extract text content from data item"""
        # Try different possible keys
        for key in ['content', 'full_content', 'text', 'input', 'context']:
            if key in item and item[key]:
                return str(item[key])

        # If no key found, return stringified item
        return str(item)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()

        chunks = []
        words = text.split()

        chunk_words = chunk_size // 5  # Approximate words per chunk
        overlap_words = overlap // 5

        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + chunk_words])
            if chunk:
                chunks.append(chunk)
            i += chunk_words - overlap_words

        return chunks


class HuggingFaceQAGenerator:
    """Generate QA pairs using Hugging Face models"""

    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 device: str = "auto",
                 max_new_tokens: int = 512):
        """
        Initialize generator with Hugging Face model

        Args:
            model_name: HF model identifier
            device: 'auto', 'cuda', 'cpu'
            max_new_tokens: Max tokens to generate
        """

        logger.info(f"Loading model: {model_name}")
        logger.info("This may take a few minutes on first run...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            if device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )

            self.device = device
            self.max_new_tokens = max_new_tokens

            logger.info(f"Model loaded successfully on {device}")

        except ImportError:
            raise ImportError("Please install: pip install transformers torch accelerate")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def generate_compliance_qa_pairs(self,
                                     text: str,
                                     num_questions: int = 5) -> List[ComplianceQAPair]:
        """Generate compliance-specific QA pairs from text"""

        prompt = self._build_compliance_prompt(text, num_questions)

        try:
            response = self._generate_text(prompt)
            qa_pairs = self._parse_qa_response(response, text)

            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            return qa_pairs

        except Exception as e:
            logger.error(f"Error generating QA pairs: {e}")
            return []

    def _build_compliance_prompt(self, text: str, num_questions: int) -> str:
        """Build prompt for compliance QA generation"""

        prompt = f"""You are an expert in financial compliance and regulations. Given the following compliance-related text, generate {num_questions} question-answer pairs that would help train an AI system to understand financial compliance requirements.

COMPLIANCE TEXT:
{text[:1500]}

REQUIREMENTS:
1. Questions should focus on:
   - Specific compliance requirements
   - Regulatory obligations
   - When rules apply
   - What constitutes violations
   - Proper procedures
   - Risk factors

2. Answers should:
   - Be accurate and complete
   - Cite specific requirements when possible
   - Explain the reasoning
   - Include practical implications

3. Mix of difficulties:
   - Easy: {num_questions // 3} questions (basic factual)
   - Medium: {num_questions // 3} questions (requires understanding)
   - Hard: {num_questions // 3} questions (complex analysis)

4. Compliance topics to cover:
   - Disclosure requirements
   - Suitability and best interest
   - Conflicts of interest
   - Risk management
   - Recordkeeping
   - Supervisory obligations

OUTPUT FORMAT (JSON):
{{
  "qa_pairs": [
    {{
      "question": "Clear compliance-focused question",
      "answer": "Detailed accurate answer with regulatory reasoning",
      "compliance_topic": "disclosure|suitability|conflicts|risk|recordkeeping|supervision",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Generate exactly {num_questions} high-quality compliance QA pairs. Output ONLY valid JSON, no other text."""

        return prompt

    def _generate_text(self, prompt: str) -> str:
        """Generate text using the model"""

        # Format for instruction-following models
        formatted_prompt = self._format_for_model(prompt)

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to device
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the new generated part
        response = generated_text[len(formatted_prompt):].strip()

        return response

    def _format_for_model(self, prompt: str) -> str:
        """Format prompt for specific model architectures"""

        model_name = self.model.config._name_or_path.lower()

        # Mistral/Mixtral format
        if "mistral" in model_name or "mixtral" in model_name:
            return f"<s>[INST] {prompt} [/INST]"

        # Llama-2 chat format
        elif "llama-2" in model_name and "chat" in model_name:
            return f"<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n{prompt} [/INST]"

        # Phi format
        elif "phi" in model_name:
            return f"<|user|>\n{prompt}<|end|>\n<|assistant|>"

        # Generic format
        else:
            return prompt

    def _parse_qa_response(self, response: str, original_text: str) -> List[ComplianceQAPair]:
        """Parse model response into QAPair objects"""

        qa_pairs = []

        try:
            # Clean response
            response = response.strip()

            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)

            # Parse JSON
            data = json.loads(response)

            # Extract QA pairs
            pairs = data.get('qa_pairs', [])

            for item in pairs:
                if 'question' in item and 'answer' in item:
                    qa_pair = ComplianceQAPair(
                        question=item['question'],
                        answer=item['answer'],
                        context=original_text[:1000],  # Store context
                        compliance_topic=item.get('compliance_topic', 'general'),
                        difficulty=item.get('difficulty', 'medium'),
                        source_document="compliance_data"
                    )
                    qa_pairs.append(qa_pair)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:200]}")

            # Fallback: try to extract QA pairs with regex
            qa_pairs = self._fallback_extraction(response, original_text)

        except Exception as e:
            logger.error(f"Error parsing response: {e}")

        return qa_pairs

    def _fallback_extraction(self, response: str, original_text: str) -> List[ComplianceQAPair]:
        """Fallback method to extract QA pairs using regex"""

        qa_pairs = []

        # Try to find Q: and A: patterns
        qa_pattern = r'(?:Q|Question):\s*(.+?)\s*(?:A|Answer):\s*(.+?)(?=(?:Q|Question):|$)'
        matches = re.findall(qa_pattern, response, re.DOTALL | re.IGNORECASE)

        for question, answer in matches:
            qa_pair = ComplianceQAPair(
                question=question.strip(),
                answer=answer.strip(),
                context=original_text[:1000],
                compliance_topic="general",
                difficulty="medium",
                source_document="compliance_data"
            )
            qa_pairs.append(qa_pair)

        return qa_pairs


class ComplianceQADatasetBuilder:
    """Build complete QA dataset from compliance data"""

    def __init__(self,
                 model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 device: str = "auto"):

        self.loader = ComplianceDataLoader()
        self.generator = HuggingFaceQAGenerator(model_name=model_name, device=device)
        self.all_qa_pairs = []

    def build_dataset(self,
                      input_file: str,
                      num_questions_per_chunk: int = 5,
                      max_chunks: int = None) -> List[ComplianceQAPair]:
        """
        Build QA dataset from compliance data file

        Args:
            input_file: Path to compliance data (JSON/JSONL)
            num_questions_per_chunk: Questions to generate per chunk
            max_chunks: Maximum chunks to process (None = all)

        Returns:
            List of ComplianceQAPair objects
        """

        logger.info(f"Loading data from: {input_file}")

        # Load data
        data_items = self.loader.load_data(input_file)
        logger.info(f"Loaded {len(data_items)} data items")

        # Process each item
        chunks_processed = 0

        for i, item in enumerate(data_items):

            # Extract text
            text = self.loader.extract_text(item)

            # Skip if too short
            if len(text) < 200:
                continue

            # Chunk text
            chunks = self.loader.chunk_text(text, chunk_size=1200, overlap=200)

            logger.info(f"Processing item {i + 1}/{len(data_items)}: {len(chunks)} chunks")

            # Generate QA for each chunk
            for j, chunk in enumerate(chunks):

                if max_chunks and chunks_processed >= max_chunks:
                    logger.info(f"Reached max_chunks limit ({max_chunks})")
                    break

                logger.info(f"  Generating QA for chunk {j + 1}/{len(chunks)}")

                qa_pairs = self.generator.generate_compliance_qa_pairs(
                    text=chunk,
                    num_questions=num_questions_per_chunk
                )

                # Add source document info
                for qa in qa_pairs:
                    qa.source_document = item.get('url', item.get('source', f"item_{i}"))

                self.all_qa_pairs.extend(qa_pairs)
                chunks_processed += 1

            if max_chunks and chunks_processed >= max_chunks:
                break

        logger.info(f"Generated {len(self.all_qa_pairs)} total QA pairs")
        return self.all_qa_pairs

    def filter_qa_pairs(self) -> List[ComplianceQAPair]:
        """Filter QA pairs for quality"""

        filtered = []

        for qa in self.all_qa_pairs:
            # Basic quality checks
            if len(qa.question) < 15:
                continue
            if len(qa.answer) < 30:
                continue
            if not qa.question.strip().endswith('?'):
                continue
            if qa.question.lower() == qa.answer.lower():
                continue

            filtered.append(qa)

        logger.info(f"Filtered to {len(filtered)}/{len(self.all_qa_pairs)} QA pairs")
        self.all_qa_pairs = filtered
        return filtered

    def deduplicate(self) -> List[ComplianceQAPair]:
        """Remove duplicate QA pairs"""

        seen = set()
        unique = []

        for qa in self.all_qa_pairs:
            fingerprint = (qa.question.lower().strip(), qa.answer.lower()[:100])

            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(qa)

        logger.info(f"Removed {len(self.all_qa_pairs) - len(unique)} duplicates")
        self.all_qa_pairs = unique
        return unique

    def save_dataset(self,
                     output_file: str,
                     train_split: float = 0.8) -> Dict[str, int]:
        """
        Save QA dataset to file

        Args:
            output_file: Output file path (.jsonl)
            train_split: Proportion for training set

        Returns:
            Statistics
        """

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to training format
        formatted_data = [qa.to_training_format() for qa in self.all_qa_pairs]

        # Shuffle
        import random
        random.shuffle(formatted_data)

        # Split
        split_idx = int(len(formatted_data) * train_split)
        train_data = formatted_data[:split_idx]
        val_data = formatted_data[split_idx:]

        # Save train
        train_path = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')

        # Save val
        val_path = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item) + '\n')

        # Save stats
        stats = self._compute_statistics()
        stats_path = output_path.parent / f"{output_path.stem}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved {len(train_data)} training examples to {train_path}")
        logger.info(f"Saved {len(val_data)} validation examples to {val_path}")
        logger.info(f"Saved statistics to {stats_path}")

        return {
            'total': len(formatted_data),
            'train': len(train_data),
            'val': len(val_data)
        }

    def _compute_statistics(self) -> Dict:
        """Compute dataset statistics"""

        stats = {
            'total_qa_pairs': len(self.all_qa_pairs),
            'compliance_topics': {},
            'difficulty_distribution': {},
            'avg_question_length': 0,
            'avg_answer_length': 0,
            'generation_date': datetime.now().isoformat()
        }

        total_q_len = 0
        total_a_len = 0

        for qa in self.all_qa_pairs:
            # Compliance topics
            stats['compliance_topics'][qa.compliance_topic] = \
                stats['compliance_topics'].get(qa.compliance_topic, 0) + 1

            # Difficulty
            stats['difficulty_distribution'][qa.difficulty] = \
                stats['difficulty_distribution'].get(qa.difficulty, 0) + 1

            # Lengths
            total_q_len += len(qa.question)
            total_a_len += len(qa.answer)

        if self.all_qa_pairs:
            stats['avg_question_length'] = total_q_len / len(self.all_qa_pairs)
            stats['avg_answer_length'] = total_a_len / len(self.all_qa_pairs)

        return stats


def main():
    """Main execution"""

    parser = argparse.ArgumentParser(
        description='Generate compliance QA pairs using Hugging Face models'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input compliance data file (JSON/JSONL)')
    parser.add_argument('--output', '-o', default='compliance_qa_pairs.jsonl',
                        help='Output file path')
    parser.add_argument('--model', '-m',
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Hugging Face model name')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                        default='auto',
                        help='Device to use')
    parser.add_argument('--questions-per-chunk', type=int, default=5,
                        help='Questions to generate per chunk')
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Maximum chunks to process (for testing)')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPLIANCE QA PAIR GENERATOR - HUGGING FACE EDITION")
    print("=" * 80)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    # Initialize builder
    logger.info("Initializing QA generator...")
    builder = ComplianceQADatasetBuilder(
        model_name=args.model,
        device=args.device
    )

    # Build dataset
    logger.info("Building QA dataset...")
    builder.build_dataset(
        input_file=args.input,
        num_questions_per_chunk=args.questions_per_chunk,
        max_chunks=args.max_chunks
    )

    # Filter and deduplicate
    logger.info("Filtering and deduplicating...")
    builder.filter_qa_pairs()
    builder.deduplicate()

    # Save
    logger.info("Saving dataset...")
    stats = builder.save_dataset(args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("QA GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal QA pairs: {stats['total']}")
    print(f"Training examples: {stats['train']}")
    print(f"Validation examples: {stats['val']}")
    print(f"\nOutput files:")
    print(f"  - {args.output.replace('.jsonl', '_train.jsonl')}")
    print(f"  - {args.output.replace('.jsonl', '_val.jsonl')}")
    print(f"  - {args.output.replace('.jsonl', '_stats.json')}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()