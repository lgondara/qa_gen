#!/usr/bin/env python3
"""
Compliance Document Fine-tuning Data Generator

Generates instruction-response pairs from SEC/FINRA scraped documents
for fine-tuning LLMs on compliance-related tasks.

Supports multiple pair generation strategies:
1. Question-Answer pairs
2. Summarization pairs
3. Violation classification pairs
4. Entity extraction pairs
5. Analysis/reasoning pairs

Author: Compliance AI Training Pipeline
"""

import json
import re
import random
import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Generator
from datetime import datetime
import hashlib

# HuggingFace imports
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline,
        BitsAndBytesConfig
    )
    from datasets import Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace libraries not installed. Install with:")
    print("  pip install transformers datasets torch accelerate bitsandbytes")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComplianceDocument:
    """Represents a scraped compliance document."""
    unique_id: str
    source: str
    action_type: str
    title: str
    date: Optional[str]
    url: str
    summary: Optional[str]
    violations: list
    penalties: Optional[str]
    raw_text: Optional[str]
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ComplianceDocument':
        violations = d.get('violations', [])
        if isinstance(violations, str):
            try:
                violations = json.loads(violations)
            except:
                violations = []
        return cls(
            unique_id=d.get('unique_id', ''),
            source=d.get('source', ''),
            action_type=d.get('action_type', ''),
            title=d.get('title', ''),
            date=d.get('date'),
            url=d.get('url', ''),
            summary=d.get('summary'),
            violations=violations,
            penalties=d.get('penalties'),
            raw_text=d.get('raw_text')
        )


@dataclass
class TrainingPair:
    """Represents a single training pair for fine-tuning."""
    instruction: str
    input_text: str
    output: str
    task_type: str
    source_doc_id: str
    metadata: dict = field(default_factory=dict)
    
    @property
    def unique_id(self) -> str:
        content = f"{self.instruction}:{self.input_text}:{self.output}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_alpaca_format(self) -> dict:
        """Convert to Alpaca training format."""
        return {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output
        }
    
    def to_sharegpt_format(self) -> dict:
        """Convert to ShareGPT conversation format."""
        messages = []
        if self.input_text:
            messages.append({
                "from": "human",
                "value": f"{self.instruction}\n\nContext:\n{self.input_text}"
            })
        else:
            messages.append({
                "from": "human", 
                "value": self.instruction
            })
        messages.append({
            "from": "gpt",
            "value": self.output
        })
        return {"conversations": messages}
    
    def to_chat_format(self) -> dict:
        """Convert to chat/messages format for newer models."""
        user_content = self.instruction
        if self.input_text:
            user_content += f"\n\nDocument:\n{self.input_text}"
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": self.output}
            ]
        }


# =============================================================================
# Document Loader
# =============================================================================

class DocumentLoader:
    """Load compliance documents from various sources."""
    
    @staticmethod
    def from_json(filepath: str) -> list[ComplianceDocument]:
        """Load documents from JSON export."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        actions = data.get('actions', data) if isinstance(data, dict) else data
        return [ComplianceDocument.from_dict(a) for a in actions]
    
    @staticmethod
    def from_database(db_path: str, limit: int = None) -> list[ComplianceDocument]:
        """Load documents from SQLite database."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM enforcement_actions"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        docs = [ComplianceDocument.from_dict(dict(row)) for row in cursor.fetchall()]
        conn.close()
        return docs
    
    @staticmethod
    def from_csv(filepath: str) -> list[ComplianceDocument]:
        """Load documents from CSV export."""
        import csv
        docs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                docs.append(ComplianceDocument.from_dict(row))
        return docs


# =============================================================================
# Prompt Templates
# =============================================================================

class PromptTemplates:
    """Templates for generating different types of training pairs."""
    
    # Question-Answer templates
    QA_TEMPLATES = [
        {
            "instruction": "What compliance violations are described in this enforcement action?",
            "task": "violation_identification"
        },
        {
            "instruction": "Summarize the key facts of this SEC/FINRA enforcement action.",
            "task": "summarization"
        },
        {
            "instruction": "What penalties or sanctions were imposed in this case?",
            "task": "penalty_extraction"
        },
        {
            "instruction": "Who are the respondents or defendants in this enforcement action?",
            "task": "entity_extraction"
        },
        {
            "instruction": "What regulatory rules or laws were violated according to this document?",
            "task": "rule_identification"
        },
        {
            "instruction": "What lessons can compliance officers learn from this enforcement action?",
            "task": "analysis"
        },
        {
            "instruction": "Classify the type of compliance violation described in this document.",
            "task": "classification"
        },
        {
            "instruction": "What supervisory failures contributed to this violation?",
            "task": "root_cause"
        },
        {
            "instruction": "Extract the timeline of events from this enforcement action.",
            "task": "timeline_extraction"
        },
        {
            "instruction": "What remediation steps were required as part of this settlement?",
            "task": "remediation"
        },
    ]
    
    # Instruction variants for diversity
    INSTRUCTION_VARIANTS = {
        "violation_identification": [
            "Identify all compliance violations mentioned in the following document.",
            "What securities law violations are alleged in this enforcement action?",
            "List the regulatory violations found in this case.",
            "Based on the document, what compliance failures occurred?",
        ],
        "summarization": [
            "Provide a brief summary of this enforcement action.",
            "Summarize the main points of this compliance case.",
            "Give an executive summary of this regulatory action.",
            "What is the essence of this SEC/FINRA enforcement?",
        ],
        "penalty_extraction": [
            "What financial penalties were assessed in this case?",
            "Extract all monetary sanctions from this document.",
            "What fines or disgorgement amounts are mentioned?",
            "Identify the civil penalties imposed.",
        ],
        "classification": [
            "What category of compliance violation does this represent?",
            "Classify this enforcement action by violation type.",
            "What type of regulatory breach is described here?",
            "Categorize the nature of this compliance failure.",
        ],
        "analysis": [
            "What are the key takeaways for compliance professionals?",
            "How could this violation have been prevented?",
            "What compliance controls failed in this case?",
            "Analyze the root causes of this enforcement action.",
        ],
    }
    
    # System prompts for different tasks
    SYSTEM_PROMPTS = {
        "compliance_expert": """You are an expert compliance analyst specializing in SEC and FINRA regulations. 
You provide accurate, detailed analysis of enforcement actions and compliance violations. 
Your responses are professional, factual, and actionable for compliance officers.""",
        
        "summarizer": """You are a compliance document summarizer. 
You extract key information from regulatory enforcement actions and present it clearly and concisely.
Focus on violations, penalties, and lessons learned.""",
        
        "classifier": """You are a compliance violation classifier.
You categorize enforcement actions by violation type, severity, and regulatory framework.
Use standard compliance terminology and be precise in your classifications.""",
    }


# =============================================================================
# Pair Generation Strategies
# =============================================================================

class PairGenerationStrategy:
    """Base class for pair generation strategies."""
    
    def generate(self, doc: ComplianceDocument) -> list[TrainingPair]:
        raise NotImplementedError


class RuleBasedGenerator(PairGenerationStrategy):
    """Generate pairs using rule-based templates."""
    
    def __init__(self, include_variants: bool = True):
        self.include_variants = include_variants
        self.templates = PromptTemplates()
    
    def generate(self, doc: ComplianceDocument) -> list[TrainingPair]:
        pairs = []
        
        # Skip documents without meaningful content
        content = doc.raw_text or doc.summary or doc.title
        if not content or len(content) < 50:
            return pairs
        
        # Truncate very long documents
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        # Generate summarization pair
        pairs.append(self._create_summary_pair(doc, content))
        
        # Generate violation identification pair
        if doc.violations:
            pairs.append(self._create_violation_pair(doc, content))
        
        # Generate penalty extraction pair
        if doc.penalties:
            pairs.append(self._create_penalty_pair(doc, content))
        
        # Generate classification pair
        pairs.append(self._create_classification_pair(doc, content))
        
        # Generate analysis pair
        pairs.append(self._create_analysis_pair(doc, content))
        
        # Add instruction variants for diversity
        if self.include_variants:
            pairs.extend(self._create_variant_pairs(doc, content))
        
        return [p for p in pairs if p is not None]
    
    def _create_summary_pair(self, doc: ComplianceDocument, content: str) -> TrainingPair:
        instruction = random.choice(self.templates.INSTRUCTION_VARIANTS.get(
            "summarization", 
            ["Summarize this enforcement action."]
        ))
        
        # Create a good summary output
        summary_parts = []
        summary_parts.append(f"This {doc.source} {doc.action_type.replace('_', ' ')} involves {doc.title}.")
        
        if doc.violations:
            summary_parts.append(f"The key violations include: {', '.join(doc.violations[:5])}.")
        
        if doc.penalties:
            summary_parts.append(f"Penalties imposed: {doc.penalties}.")
        
        if doc.date:
            summary_parts.append(f"Date: {doc.date}.")
        
        return TrainingPair(
            instruction=instruction,
            input_text=content[:2000],
            output=" ".join(summary_parts),
            task_type="summarization",
            source_doc_id=doc.unique_id,
            metadata={"source": doc.source, "url": doc.url}
        )
    
    def _create_violation_pair(self, doc: ComplianceDocument, content: str) -> TrainingPair:
        instruction = random.choice(self.templates.INSTRUCTION_VARIANTS.get(
            "violation_identification",
            ["What violations are described?"]
        ))
        
        violations_text = "The following compliance violations are identified in this enforcement action:\n\n"
        for i, v in enumerate(doc.violations, 1):
            violations_text += f"{i}. {v.replace('_', ' ').title()}\n"
        
        return TrainingPair(
            instruction=instruction,
            input_text=content[:2000],
            output=violations_text.strip(),
            task_type="violation_identification",
            source_doc_id=doc.unique_id
        )
    
    def _create_penalty_pair(self, doc: ComplianceDocument, content: str) -> TrainingPair:
        instruction = random.choice(self.templates.INSTRUCTION_VARIANTS.get(
            "penalty_extraction",
            ["What penalties were imposed?"]
        ))
        
        output = f"The penalties imposed in this case include: {doc.penalties}"
        
        return TrainingPair(
            instruction=instruction,
            input_text=content[:2000],
            output=output,
            task_type="penalty_extraction",
            source_doc_id=doc.unique_id
        )
    
    def _create_classification_pair(self, doc: ComplianceDocument, content: str) -> TrainingPair:
        instruction = random.choice(self.templates.INSTRUCTION_VARIANTS.get(
            "classification",
            ["Classify this violation."]
        ))
        
        # Determine primary classification
        violation_categories = {
            "recordkeeping": ["recordkeeping", "books and records", "off-channel", "communications"],
            "supervision": ["supervision", "supervisory", "failure to supervise", "oversight"],
            "disclosure": ["disclosure", "misleading", "material misstatement", "omission"],
            "fraud": ["fraud", "fraudulent", "scheme", "deception"],
            "aml_kyc": ["aml", "anti-money laundering", "kyc", "bsa", "suspicious activity"],
            "suitability": ["suitability", "best interest", "fiduciary", "churning"],
            "market_manipulation": ["insider trading", "manipulation", "spoofing", "front running"],
            "custody": ["custody", "safeguarding", "customer assets"],
            "advertising": ["advertising", "marketing", "testimonial"],
            "cybersecurity": ["cybersecurity", "data breach", "cyber"],
        }
        
        found_categories = []
        for cat, keywords in violation_categories.items():
            if any(kw in content.lower() for kw in keywords):
                found_categories.append(cat)
        
        if not found_categories:
            found_categories = ["general_compliance"]
        
        output = f"This enforcement action is classified as: {', '.join(found_categories).replace('_', ' ').title()}\n\n"
        output += f"Source: {doc.source}\n"
        output += f"Action Type: {doc.action_type.replace('_', ' ').title()}"
        
        return TrainingPair(
            instruction=instruction,
            input_text=content[:2000],
            output=output,
            task_type="classification",
            source_doc_id=doc.unique_id
        )
    
    def _create_analysis_pair(self, doc: ComplianceDocument, content: str) -> TrainingPair:
        instruction = random.choice(self.templates.INSTRUCTION_VARIANTS.get(
            "analysis",
            ["What are the compliance lessons from this case?"]
        ))
        
        # Generate analytical output
        analysis_parts = ["Key compliance lessons from this enforcement action:\n"]
        
        if "recordkeeping" in str(doc.violations).lower() or "record" in content.lower():
            analysis_parts.append("1. Maintain comprehensive records of all business communications, including electronic messages.")
        
        if "supervision" in str(doc.violations).lower() or "supervis" in content.lower():
            analysis_parts.append("2. Implement robust supervisory systems with clear escalation procedures.")
        
        if "disclosure" in str(doc.violations).lower() or "disclos" in content.lower():
            analysis_parts.append("3. Ensure all material information is accurately disclosed to clients and regulators.")
        
        if doc.penalties:
            analysis_parts.append(f"4. Financial impact: {doc.penalties} - emphasizing the cost of non-compliance.")
        
        analysis_parts.append("5. Regular compliance training and policy reviews are essential to prevent similar violations.")
        
        return TrainingPair(
            instruction=instruction,
            input_text=content[:2000],
            output="\n".join(analysis_parts),
            task_type="analysis",
            source_doc_id=doc.unique_id
        )
    
    def _create_variant_pairs(self, doc: ComplianceDocument, content: str) -> list[TrainingPair]:
        """Create additional variant pairs for training diversity."""
        pairs = []
        
        # Yes/No question variants
        if doc.violations:
            pairs.append(TrainingPair(
                instruction=f"Does this document describe {random.choice(doc.violations)} violations?",
                input_text=content[:1500],
                output="Yes, this enforcement action involves " + random.choice(doc.violations) + " violations as described in the document.",
                task_type="yes_no_question",
                source_doc_id=doc.unique_id
            ))
        
        # Source identification
        pairs.append(TrainingPair(
            instruction="Is this an SEC or FINRA enforcement action?",
            input_text=content[:1500],
            output=f"This is a {doc.source} enforcement action ({doc.action_type.replace('_', ' ')}).",
            task_type="source_identification",
            source_doc_id=doc.unique_id
        ))
        
        return pairs


class LLMAssistedGenerator(PairGenerationStrategy):
    """Generate pairs using a local LLM for more diverse outputs."""
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        device: str = "auto",
        use_quantization: bool = True
    ):
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace libraries required for LLM-assisted generation")
        
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        self._model = None
        self._tokenizer = None
        self._pipeline = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._pipeline is not None:
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Determine model type and load accordingly
        if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                device_map=self.device if self.device != "auto" else "auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self._pipeline = pipeline(
                "text2text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=256
            )
        else:
            # For causal LM models (Mistral, Llama, etc.)
            quantization_config = None
            if self.use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device != "auto" else "auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=256
            )
        
        logger.info("Model loaded successfully")
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the loaded model."""
        self._load_model()
        
        try:
            result = self._pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    generated = result[0]["generated_text"]
                    # For causal LMs, remove the input prompt from output
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    return generated
            return ""
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    def generate(self, doc: ComplianceDocument) -> list[TrainingPair]:
        pairs = []
        
        content = doc.raw_text or doc.summary or doc.title
        if not content or len(content) < 100:
            return pairs
        
        # Truncate for model context
        content_truncated = content[:1500]
        
        # Generate Q&A pair
        qa_prompt = f"""Given this compliance enforcement document, generate a relevant question and detailed answer.

Document: {content_truncated}

Generate a compliance-focused question and answer pair in this format:
Question: [Your question]
Answer: [Your detailed answer]"""
        
        qa_response = self._generate_text(qa_prompt)
        if qa_response and "Question:" in qa_response and "Answer:" in qa_response:
            try:
                q_start = qa_response.index("Question:") + 9
                a_start = qa_response.index("Answer:")
                question = qa_response[q_start:a_start].strip()
                answer = qa_response[a_start + 7:].strip()
                
                if question and answer:
                    pairs.append(TrainingPair(
                        instruction=question,
                        input_text=content_truncated,
                        output=answer,
                        task_type="llm_generated_qa",
                        source_doc_id=doc.unique_id
                    ))
            except ValueError:
                pass
        
        # Generate summary using LLM
        summary_prompt = f"""Summarize this SEC/FINRA enforcement action in 2-3 sentences, focusing on violations and penalties:

{content_truncated}

Summary:"""
        
        summary = self._generate_text(summary_prompt)
        if summary and len(summary) > 20:
            pairs.append(TrainingPair(
                instruction="Provide a concise summary of this enforcement action.",
                input_text=content_truncated,
                output=summary,
                task_type="llm_generated_summary",
                source_doc_id=doc.unique_id
            ))
        
        return pairs


# =============================================================================
# Main Generator Class
# =============================================================================

class ComplianceFinetuneDataGenerator:
    """Main class for generating fine-tuning data from compliance documents."""
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_model: str = "google/flan-t5-base",
        include_variants: bool = True
    ):
        self.generators: list[PairGenerationStrategy] = []
        
        # Always include rule-based generator
        self.generators.append(RuleBasedGenerator(include_variants=include_variants))
        
        # Optionally add LLM-assisted generator
        if use_llm and HF_AVAILABLE:
            try:
                self.generators.append(LLMAssistedGenerator(model_name=llm_model))
            except Exception as e:
                logger.warning(f"Could not initialize LLM generator: {e}")
        
        self.documents: list[ComplianceDocument] = []
        self.pairs: list[TrainingPair] = []
    
    def load_documents(
        self, 
        source: str, 
        source_type: str = "json",
        limit: Optional[int] = None
    ):
        """Load documents from specified source."""
        logger.info(f"Loading documents from {source} ({source_type})")
        
        if source_type == "json":
            self.documents = DocumentLoader.from_json(source)
        elif source_type == "database":
            self.documents = DocumentLoader.from_database(source, limit)
        elif source_type == "csv":
            self.documents = DocumentLoader.from_csv(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        if limit:
            self.documents = self.documents[:limit]
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return self
    
    def generate_pairs(self, max_pairs_per_doc: int = 10) -> 'ComplianceFinetuneDataGenerator':
        """Generate training pairs from loaded documents."""
        logger.info("Generating training pairs...")
        
        self.pairs = []
        
        for i, doc in enumerate(self.documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(self.documents)}")
            
            doc_pairs = []
            for generator in self.generators:
                doc_pairs.extend(generator.generate(doc))
            
            # Limit pairs per document
            if len(doc_pairs) > max_pairs_per_doc:
                doc_pairs = random.sample(doc_pairs, max_pairs_per_doc)
            
            self.pairs.extend(doc_pairs)
        
        # Deduplicate
        seen_ids = set()
        unique_pairs = []
        for pair in self.pairs:
            if pair.unique_id not in seen_ids:
                seen_ids.add(pair.unique_id)
                unique_pairs.append(pair)
        
        self.pairs = unique_pairs
        logger.info(f"Generated {len(self.pairs)} unique training pairs")
        return self
    
    def get_statistics(self) -> dict:
        """Get statistics about generated pairs."""
        stats = {
            "total_documents": len(self.documents),
            "total_pairs": len(self.pairs),
            "pairs_by_task": {},
            "pairs_by_source": {},
            "avg_instruction_length": 0,
            "avg_output_length": 0,
        }
        
        if not self.pairs:
            return stats
        
        for pair in self.pairs:
            stats["pairs_by_task"][pair.task_type] = stats["pairs_by_task"].get(pair.task_type, 0) + 1
            source = pair.metadata.get("source", "unknown")
            stats["pairs_by_source"][source] = stats["pairs_by_source"].get(source, 0) + 1
        
        stats["avg_instruction_length"] = sum(len(p.instruction) for p in self.pairs) / len(self.pairs)
        stats["avg_output_length"] = sum(len(p.output) for p in self.pairs) / len(self.pairs)
        
        return stats
    
    def export_jsonl(self, filepath: str, format_type: str = "alpaca") -> Path:
        """Export pairs to JSONL format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in self.pairs:
                if format_type == "alpaca":
                    data = pair.to_alpaca_format()
                elif format_type == "sharegpt":
                    data = pair.to_sharegpt_format()
                elif format_type == "chat":
                    data = pair.to_chat_format()
                else:
                    data = asdict(pair)
                
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        
        logger.info(f"Exported {len(self.pairs)} pairs to {filepath}")
        return filepath
    
    def export_json(self, filepath: str) -> Path:
        """Export pairs to JSON format."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_pairs": len(self.pairs),
                "statistics": self.get_statistics()
            },
            "training_data": [asdict(p) for p in self.pairs]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported to {filepath}")
        return filepath
    
    def export_huggingface_dataset(self, output_dir: str) -> Path:
        """Export as HuggingFace Dataset."""
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets library required")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict format
        data_dict = {
            "instruction": [p.instruction for p in self.pairs],
            "input": [p.input_text for p in self.pairs],
            "output": [p.output for p in self.pairs],
            "task_type": [p.task_type for p in self.pairs],
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Save
        split_dataset.save_to_disk(str(output_dir))
        
        logger.info(f"Saved HuggingFace dataset to {output_dir}")
        logger.info(f"  Train: {len(split_dataset['train'])} examples")
        logger.info(f"  Validation: {len(split_dataset['test'])} examples")
        
        return output_dir
    
    def create_sample_data(self, n_samples: int = 5) -> list[dict]:
        """Return sample training pairs for inspection."""
        samples = random.sample(self.pairs, min(n_samples, len(self.pairs)))
        return [asdict(s) for s in samples]


# =============================================================================
# CLI and Main
# =============================================================================

def main():
    """Main entry point with CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate fine-tuning data from SEC/FINRA compliance documents"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Input file (JSON/CSV) or database path"
    )
    parser.add_argument(
        "--input-type", "-t",
        choices=["json", "csv", "database"],
        default="json",
        help="Input type (default: json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="compliance_finetune_data",
        help="Output directory/file base name"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["alpaca", "sharegpt", "chat", "all"],
        default="all",
        help="Output format (default: all)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for additional pair generation"
    )
    parser.add_argument(
        "--llm-model",
        default="google/flan-t5-base",
        help="LLM model for generation (default: google/flan-t5-base)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents to process"
    )
    parser.add_argument(
        "--max-pairs-per-doc",
        type=int,
        default=8,
        help="Maximum pairs per document (default: 8)"
    )
    parser.add_argument(
        "--export-hf",
        action="store_true",
        help="Also export as HuggingFace Dataset"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Compliance Fine-tuning Data Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = ComplianceFinetuneDataGenerator(
        use_llm=args.use_llm,
        llm_model=args.llm_model
    )
    
    # Load and process
    generator.load_documents(args.input, args.input_type, args.limit)
    generator.generate_pairs(max_pairs_per_doc=args.max_pairs_per_doc)
    
    # Print statistics
    stats = generator.get_statistics()
    print(f"\nGeneration Statistics:")
    print(f"  Documents processed: {stats['total_documents']}")
    print(f"  Total pairs generated: {stats['total_pairs']}")
    print(f"\n  Pairs by task type:")
    for task, count in sorted(stats['pairs_by_task'].items(), key=lambda x: -x[1]):
        print(f"    - {task}: {count}")
    
    # Export
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    if args.format in ["alpaca", "all"]:
        generator.export_jsonl(output_base / "train_alpaca.jsonl", "alpaca")
    
    if args.format in ["sharegpt", "all"]:
        generator.export_jsonl(output_base / "train_sharegpt.jsonl", "sharegpt")
    
    if args.format in ["chat", "all"]:
        generator.export_jsonl(output_base / "train_chat.jsonl", "chat")
    
    # Full JSON export
    generator.export_json(output_base / "training_data_full.json")
    
    # HuggingFace dataset
    if args.export_hf and HF_AVAILABLE:
        generator.export_huggingface_dataset(output_base / "hf_dataset")
    
    # Show samples
    print("\n" + "=" * 60)
    print("Sample Training Pairs:")
    print("=" * 60)
    
    for i, sample in enumerate(generator.create_sample_data(3), 1):
        print(f"\n--- Sample {i} ({sample['task_type']}) ---")
        print(f"Instruction: {sample['instruction'][:100]}...")
        print(f"Input: {sample['input_text'][:100]}..." if sample['input_text'] else "Input: (none)")
        print(f"Output: {sample['output'][:150]}...")
    
    print(f"\n{'=' * 60}")
    print(f"Output saved to: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
