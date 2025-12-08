Excellent! Let me walk you through the complete process of formatting your data and fine-tuning Mistral 7B Instruct for compliance detection.

---

## **Part 1: Data Formatting**

### **Step 1: Consolidate Your Datasets**

Assuming you have:
- SEC synthetic conversations (compliant + non-compliant pairs)
- FINRA synthetic conversations (compliant + non-compliant pairs)
- Internal mutated conversations (if ready)

```python
import json
import pandas as pd
from datasets import Dataset, DatasetDict

# Load your data
sec_data = []
with open('sec_synthetic_conversations.jsonl', 'r') as f:
    for line in f:
        sec_data.append(json.loads(line))

finra_data = []
with open('finra_synthetic_conversations.jsonl', 'r') as f:
    for line in f:
        finra_data.append(json.loads(line))

# Combine
all_data = sec_data + finra_data
print(f"Total examples: {len(all_data)}")
```

### **Step 2: Extract and Structure Training Examples**

Your synthetic data has both compliant and non-compliant versions. Extract both:

```python
def extract_training_examples(synthetic_data):
    """
    Extract both compliant and non-compliant examples from synthetic data
    """
    training_examples = []
    
    for item in synthetic_data:
        # Extract compliant version
        if 'compliant_version' in item:
            compliant = item['compliant_version']
            training_examples.append({
                'conversation': compliant['scenario'],
                'final_verdict': compliant['final_verdict'],
                'executive_summary': compliant['executive_summary'],
                'rule_violated': compliant['rule_violated'],
                'compliance_topic': compliant['compliance_topic'],
                'severity': compliant['severity'],
                'evidence_list': compliant['evidence_list'],
                'reasoning_trace': compliant['reasoning_trace'],
                'source': item.get('source', 'synthetic'),
                'format_type': item.get('chunk_analysis', {}).get('format_identified', 'unknown')
            })
        
        # Extract non-compliant version
        if 'non_compliant_version' in item:
            non_compliant = item['non_compliant_version']
            training_examples.append({
                'conversation': non_compliant['scenario'],
                'final_verdict': non_compliant['final_verdict'],
                'executive_summary': non_compliant['executive_summary'],
                'rule_violated': non_compliant['rule_violated'],
                'compliance_topic': non_compliant['compliance_topic'],
                'severity': non_compliant['severity'],
                'evidence_list': non_compliant['evidence_list'],
                'reasoning_trace': non_compliant['reasoning_trace'],
                'source': item.get('source', 'synthetic'),
                'format_type': item.get('chunk_analysis', {}).get('format_identified', 'unknown')
            })
    
    return training_examples

training_examples = extract_training_examples(all_data)
print(f"Training examples after extraction: {len(training_examples)}")

# Should be roughly 2x your original data (compliant + non-compliant)
```

### **Step 3: Create Instruction-Tuning Format**

Mistral 7B Instruct expects a specific format with instruction, input, and output:

```python
def create_instruction_format(example):
    """
    Convert to instruction-following format for Mistral
    """
    
    # Create the instruction prompt
    instruction = """You are an expert AI Forensic Compliance Auditor analyzing financial advisor-client communications for regulatory violations.

Analyze the following communication and provide a comprehensive compliance assessment.

Your response must be a valid JSON object with these fields:
- final_verdict: "COMPLIANT" or "NON_COMPLIANT"
- executive_summary: One sentence summary (max 50 words)
- rule_violated: Specific rule name/number or "None"
- compliance_topic: Primary violation category or "None"
- severity: "HIGH", "MEDIUM", "LOW", or "NONE"
- evidence_list: Array of direct quotes from the communication
- reasoning_trace: Detailed analysis following the forensic protocol"""

    # The communication to analyze
    user_input = f"COMMUNICATION:\n{example['conversation']}"
    
    # Expected output (ground truth)
    output = {
        "final_verdict": example['final_verdict'],
        "executive_summary": example['executive_summary'],
        "rule_violated": example['rule_violated'],
        "compliance_topic": example['compliance_topic'],
        "severity": example['severity'],
        "evidence_list": example['evidence_list'],
        "reasoning_trace": example['reasoning_trace']
    }
    
    return {
        'instruction': instruction,
        'input': user_input,
        'output': json.dumps(output, indent=2)
    }

# Apply to all examples
formatted_data = [create_instruction_format(ex) for ex in training_examples]
```

### **Step 4: Convert to Mistral Chat Format**

Mistral 7B Instruct uses special tokens. Here's the format:

```python
def format_for_mistral(example):
    """
    Format using Mistral's chat template with special tokens
    Mistral format: <s>[INST] {instruction} [/INST] {response}</s>
    """
    
    # Combine instruction and input
    prompt = f"{example['instruction']}\n\n{example['input']}"
    
    # Full formatted text with Mistral special tokens
    formatted_text = f"<s>[INST] {prompt} [/INST]\n{example['output']}</s>"
    
    return {
        'text': formatted_text,
        'prompt': prompt,
        'completion': example['output']
    }

# Apply Mistral formatting
mistral_formatted = [format_for_mistral(ex) for ex in formatted_data]

# Convert to HuggingFace Dataset
df = pd.DataFrame(mistral_formatted)
dataset = Dataset.from_pandas(df)

print(f"Dataset size: {len(dataset)}")
print(f"\nExample formatted text (first 500 chars):\n{dataset[0]['text'][:500]}")
```

### **Step 5: Train/Val/Test Split**

```python
from sklearn.model_selection import train_test_split

# Split stratified by compliance_topic to ensure balanced representation
labels = [ex['compliance_topic'] for ex in training_examples]

train_idx, temp_idx = train_test_split(
    range(len(training_examples)), 
    test_size=0.2, 
    random_state=42,
    stratify=labels
)

val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.5, 
    random_state=42,
    stratify=[labels[i] for i in temp_idx]
)

# Create splits
train_data = [mistral_formatted[i] for i in train_idx]
val_data = [mistral_formatted[i] for i in val_idx]
test_data = [mistral_formatted[i] for i in test_idx]

print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")

# Convert to HuggingFace DatasetDict
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(pd.DataFrame(train_data)),
    'validation': Dataset.from_pandas(pd.DataFrame(val_data)),
    'test': Dataset.from_pandas(pd.DataFrame(test_data))
})

# Save to disk
dataset_dict.save_to_disk("./compliance_dataset")

# Or push to HuggingFace Hub (optional)
# dataset_dict.push_to_hub("your-username/compliance-detection-dataset")
```

---

## **Part 2: Fine-Tuning Mistral 7B Instruct**

### **Option A: Using Unsloth (RECOMMENDED - Fastest & Most Memory Efficient)**

Unsloth is optimized for fast, memory-efficient fine-tuning:

```python
# Install unsloth
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps xformers trl peft accelerate bitsandbytes

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("./compliance_dataset")

# Model configuration
max_seq_length = 2048  # Adjust based on your conversation lengths
dtype = None  # Auto-detect
load_in_4bit = True  # Use 4-bit quantization to save memory

# Load Mistral 7B Instruct with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Add LoRA adapters for parameter-efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    random_state=42,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-7b-compliance-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",  # or "wandb"
    run_name="mistral-7b-compliance-v1",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field="text",  # The field containing formatted text
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,  # Can enable for efficiency if examples are short
)

# Train
trainer.train()

# Save final model
model.save_pretrained("./mistral-7b-compliance-final")
tokenizer.save_pretrained("./mistral-7b-compliance-final")

# Save with full 16-bit precision (optional, for deployment)
model.save_pretrained_merged(
    "./mistral-7b-compliance-merged",
    tokenizer,
    save_method="merged_16bit"
)
```

---

### **Option B: Using Standard HuggingFace Transformers + PEFT**

If you prefer the standard approach:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("./compliance_dataset")

# Model name
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

# Quantization config for 4-bit training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=2048,
        padding=False,
    )

# Tokenize datasets
tokenized_train = dataset['train'].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)

tokenized_val = dataset['validation'].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['validation'].column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral-7b-compliance",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
)

# Data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("./mistral-7b-compliance-lora")
tokenizer.save_pretrained("./mistral-7b-compliance-lora")
```

---

### **Option C: Using Axolotl (Most Flexible)**

Axolotl provides a config-driven approach:

```bash
# Install axolotl
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install -e .
```

Create config file `mistral_compliance_config.yml`:

```yaml
base_model: mistralai/Mistral-7B-Instruct-v0.3
model_type: MistralForCausalLM

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: ./compliance_dataset
    type: completion
    field: text

dataset_prepared_path: ./prepared_data
val_set_size: 0.1
output_dir: ./mistral-7b-compliance-axolotl

adapter: lora
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

sequence_len: 2048
sample_packing: false
pad_to_sequence_len: true

wandb_project: mistral-compliance
wandb_watch: false

gradient_accumulation_steps: 4
micro_batch_size: 4
num_epochs: 3
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: true

gradient_checkpointing: true
early_stopping_patience: 3
logging_steps: 10
eval_steps: 500
save_steps: 500
save_total_limit: 2

warmup_steps: 100
evals_per_epoch: 4
```

Run training:

```bash
accelerate launch -m axolotl.cli.train mistral_compliance_config.yml
```

---

## **Part 3: Hyperparameter Recommendations**

### **For Your Use Case (Compliance Detection):**

```python
RECOMMENDED_HYPERPARAMETERS = {
    # Model
    'base_model': 'mistralai/Mistral-7B-Instruct-v0.3',
    'max_seq_length': 2048,  # Increase to 4096 if conversations are long
    
    # LoRA
    'lora_r': 16,  # Can try 32 for more capacity
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                       'gate_proj', 'up_proj', 'down_proj'],
    
    # Training
    'num_epochs': 3,  # Start with 3, may need 5-10 for convergence
    'batch_size': 4,  # Per device
    'gradient_accumulation_steps': 4,  # Effective batch size = 16
    'learning_rate': 2e-4,  # Conservative for LoRA
    'warmup_steps': 100,
    'lr_scheduler': 'cosine',
    'weight_decay': 0.01,
    
    # Optimization
    'optimizer': 'adamw_8bit',  # or 'paged_adamw_8bit'
    'fp16': True,  # or bf16 if supported
    'gradient_checkpointing': True,
    
    # Regularization
    'max_grad_norm': 1.0,
    
    # Evaluation
    'eval_steps': 500,
    'save_steps': 500,
    'logging_steps': 10,
}
```

### **Adjust Based on Dataset Size:**

```python
# Small dataset (<10k examples)
learning_rate = 5e-4
num_epochs = 5-10
early_stopping_patience = 3

# Medium dataset (10k-50k examples)  
learning_rate = 2e-4
num_epochs = 3-5
early_stopping_patience = 2

# Large dataset (>50k examples)
learning_rate = 1e-4
num_epochs = 2-3
early_stopping_patience = 1
```

---

## **Part 4: Monitoring Training**

### **TensorBoard:**

```python
# During training, in another terminal:
tensorboard --logdir=./mistral-7b-compliance/runs
```

### **Weights & Biases (Better):**

```python
# Install
pip install wandb

# In training code
import wandb
wandb.login()

# In TrainingArguments
training_args = TrainingArguments(
    # ... other args
    report_to="wandb",
    run_name="mistral-compliance-v1",
)
```

### **Key Metrics to Watch:**

```python
METRICS_TO_MONITOR = {
    'train_loss': 'Should decrease steadily',
    'eval_loss': 'Should decrease, watch for overfitting if diverges from train_loss',
    'learning_rate': 'Should follow scheduler (cosine decay)',
    'gradient_norm': 'Should be stable, spikes indicate instability',
    'perplexity': 'Lower is better',
}

# Custom evaluation metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Add custom compliance-specific metrics
    # e.g., accuracy on verdict prediction, F1 on violation categories
    pass
```

---

## **Part 5: Evaluation & Testing**

### **Generate Predictions on Test Set:**

```python
from transformers import pipeline
import json

# Load fine-tuned model
model_path = "./mistral-7b-compliance-final"
pipe = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    device_map="auto",
)

# Load test data
test_data = dataset_dict['test']

def evaluate_model(test_examples, num_samples=100):
    results = []
    
    for i, example in enumerate(test_examples.select(range(num_samples))):
        # Extract prompt (instruction + input)
        prompt = example['prompt']
        ground_truth = json.loads(example['completion'])
        
        # Generate prediction
        formatted_prompt = f"<s>[INST] {prompt} [/INST]\n"
        outputs = pipe(
            formatted_prompt,
            max_new_tokens=1024,
            do_sample=False,  # Deterministic for evaluation
            temperature=0.1,
            top_p=0.95,
        )
        
        predicted_text = outputs[0]['generated_text']
        # Extract JSON from response
        try:
            # Find JSON in response
            json_start = predicted_text.find('{')
            json_end = predicted_text.rfind('}') + 1
            predicted_json = json.loads(predicted_text[json_start:json_end])
        except:
            predicted_json = None
        
        results.append({
            'prompt': prompt,
            'ground_truth': ground_truth,
            'prediction': predicted_json,
            'raw_output': predicted_text
        })
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{num_samples}")
    
    return results

# Run evaluation
eval_results = evaluate_model(test_data, num_samples=500)

# Save results
with open('evaluation_results.json', 'w') as f:
    json.dump(eval_results, f, indent=2)
```

### **Calculate Metrics:**

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def calculate_metrics(results):
    # Extract verdicts
    true_verdicts = []
    pred_verdicts = []
    
    true_topics = []
    pred_topics = []
    
    true_severity = []
    pred_severity = []
    
    for r in results:
        if r['prediction'] is None:
            continue
        
        true_verdicts.append(r['ground_truth']['final_verdict'])
        pred_verdicts.append(r['prediction'].get('final_verdict', 'UNKNOWN'))
        
        true_topics.append(r['ground_truth']['compliance_topic'])
        pred_topics.append(r['prediction'].get('compliance_topic', 'None'))
        
        true_severity.append(r['ground_truth']['severity'])
        pred_severity.append(r['prediction'].get('severity', 'NONE'))
    
    # Verdict metrics (binary: COMPLIANT vs NON_COMPLIANT)
    verdict_accuracy = accuracy_score(true_verdicts, pred_verdicts)
    verdict_precision, verdict_recall, verdict_f1, _ = precision_recall_fscore_support(
        true_verdicts, pred_verdicts, average='binary', pos_label='NON_COMPLIANT'
    )
    
    # Compliance topic metrics (multi-class)
    topic_accuracy = accuracy_score(true_topics, pred_topics)
    topic_precision, topic_recall, topic_f1, _ = precision_recall_fscore_support(
        true_topics, pred_topics, average='weighted'
    )
    
    # Severity metrics
    severity_accuracy = accuracy_score(true_severity, pred_severity)
    
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nVERDICT CLASSIFICATION:")
    print(f"  Accuracy:  {verdict_accuracy:.3f}")
    print(f"  Precision: {verdict_precision:.3f}")
    print(f"  Recall:    {verdict_recall:.3f}")
    print(f"  F1 Score:  {verdict_f1:.3f}")
    
    print(f"\nCOMPLIANCE TOPIC CLASSIFICATION:")
    print(f"  Accuracy:  {topic_accuracy:.3f}")
    print(f"  Precision: {topic_precision:.3f}")
    print(f"  Recall:    {topic_recall:.3f}")
    print(f"  F1 Score:  {topic_f1:.3f}")
    
    print(f"\nSEVERITY CLASSIFICATION:")
    print(f"  Accuracy:  {severity_accuracy:.3f}")
    
    # Confusion matrices
    print(f"\nVERDICT CONFUSION MATRIX:")
    print(confusion_matrix(true_verdicts, pred_verdicts))
    
    return {
        'verdict': {
            'accuracy': verdict_accuracy,
            'precision': verdict_precision,
            'recall': verdict_recall,
            'f1': verdict_f1
        },
        'topic': {
            'accuracy': topic_accuracy,
            'precision': topic_precision,
            'recall': topic_recall,
            'f1': topic_f1
        },
        'severity': {
            'accuracy': severity_accuracy
        }
    }

metrics = calculate_metrics(eval_results)
```

---

## **Part 6: Inference & Deployment**

### **Simple Inference Script:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

class ComplianceDetector:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    
    def analyze(self, conversation):
        instruction = """You are an expert AI Forensic Compliance Auditor analyzing financial advisor-client communications for regulatory violations.

Analyze the following communication and provide a comprehensive compliance assessment.

Your response must be a valid JSON object with these fields:
- final_verdict: "COMPLIANT" or "NON_COMPLIANT"
- executive_summary: One sentence summary (max 50 words)
- rule_violated: Specific rule name/number or "None"
- compliance_topic: Primary violation category or "None"
- severity: "HIGH", "MEDIUM", "LOW", or "NONE"
- evidence_list: Array of direct quotes from the communication
- reasoning_trace: Detailed analysis following the forensic protocol"""
        
        user_input = f"COMMUNICATION:\n{conversation}"
        prompt = f"{instruction}\n\n{user_input}"
        
        formatted_prompt = f"<s>[INST] {prompt} [/INST]\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            result = json.loads(response[json_start:json_end])
        except Exception as e:
            result = {
                'error': str(e),
                'raw_response': response
            }
        
        return result

# Usage
detector = ComplianceDetector("./mistral-7b-compliance-final")

conversation = """
Client: What kind of returns can I expect?

Advisor: This fund will definitely deliver 10% annually. It's a sure thing, 
you can't lose money with this investment.
"""

result = detector.analyze(conversation)
print(json.dumps(result, indent=2))
```

---

## **Part 7: Best Practices & Tips**

### **Memory Optimization:**

```python
# If running out of memory:
MEMORY_SAVING_TECHNIQUES = {
    '4-bit quantization': 'load_in_4bit=True',
    'Gradient checkpointing': 'gradient_checkpointing=True',
    'Smaller batch size': 'per_device_train_batch_size=2',
    'Larger grad accumulation': 'gradient_accumulation_steps=8',
    'Shorter sequences': 'max_seq_length=1024',
    'LoRA instead of full fine-tune': 'Always recommended',
    'Flash Attention 2': 'attn_implementation="flash_attention_2"'
}
```

### **Training Time Estimates:**

```python
# Rough estimates on A100 40GB:
TRAINING_TIME_ESTIMATES = {
    '10k examples, 3 epochs': '2-3 hours',
    '50k examples, 3 epochs': '8-12 hours',
    '100k examples, 3 epochs': '16-24 hours',
}

# On consumer GPU (RTX 4090):
# Multiply by 1.5-2x
```

### **When to Stop Training:**

```python
STOPPING_CRITERIA = {
    'eval_loss plateaus': 'For 2-3 evaluation cycles',
    'eval_loss increases': 'Overfitting - stop immediately',
    'train_loss near zero': 'Likely overfitting',
    'F1 score plateaus': 'On validation set',
}
```

---

## **Quick Start Script (Complete Pipeline):**

```python
# quick_train.py
"""
Complete pipeline: Data formatting → Training → Evaluation
"""

import json
from datasets import Dataset, DatasetDict
import pandas as pd
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. Load and format data
def prepare_data(sec_file, finra_file):
    # Load
    with open(sec_file) as f:
        sec_data = [json.loads(line) for line in f]
    with open(finra_file) as f:
        finra_data = [json.loads(line) for line in f]
    
    all_data = sec_data + finra_data
    
    # Extract training examples
    examples = []
    for item in all_data:
        for version in ['compliant_version', 'non_compliant_version']:
            if version in item:
                v = item[version]
                instruction = "Analyze this financial advisor communication for compliance violations."
                user_input = f"COMMUNICATION:\n{v['scenario']}"
                output = json.dumps({
                    'final_verdict': v['final_verdict'],
                    'executive_summary': v['executive_summary'],
                    'rule_violated': v['rule_violated'],
                    'compliance_topic': v['compliance_topic'],
                    'severity': v['severity'],
                    'evidence_list': v['evidence_list'],
                    'reasoning_trace': v['reasoning_trace']
                }, indent=2)
                
                text = f"<s>[INST] {instruction}\n\n{user_input} [/INST]\n{output}</s>"
                examples.append({'text': text})
    
    # Create dataset
    df = pd.DataFrame(examples)
    dataset = Dataset.from_pandas(df)
    
    # Split
    split = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = split['test'].train_test_split(test_size=0.5, seed=42)
    
    return DatasetDict({
        'train': split['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    })

# 2. Train
def train_model(dataset):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            eval_steps=500,
            save_steps=500,
            report_to="tensorboard",
        ),
    )
    
    trainer.train()
    model.save_pretrained("./mistral-compliance-final")
    tokenizer.save_pretrained("./mistral-compliance-final")
    
    return model, tokenizer

# Run
if __name__ == "__main__":
    dataset = prepare_data("sec_data.jsonl", "finra_data.jsonl")
    print(f"Dataset prepared: {len(dataset['train'])} train, {len(dataset['validation'])} val")
    
    model, tokenizer = train_model(dataset)
    print("Training complete!")
```

Run it:
```bash
python quick_train.py
```

---

**That's it!** You now have a complete pipeline from data formatting to fine-tuned model. Start with Unsloth (Option A) for the easiest and fastest training experience.
