

## **Complete Fine-Tuning Script (No Unsloth)**

```python
"""
Fine-tune Mistral 7B for Compliance Detection
Using: transformers + PEFT + bitsandbytes
Optimized for 23GB GPU
"""

import os
import json
import torch
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # Quantization (set to None for full precision, faster but more memory)
    use_4bit = False  # Set True if memory constrained
    use_8bit = False
    
    # LoRA settings
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # Training hyperparameters (optimized for 23GB GPU)
    max_seq_length = 2048
    per_device_train_batch_size = 24  # Maximize for speed
    per_device_eval_batch_size = 24
    gradient_accumulation_steps = 1
    num_train_epochs = 3
    learning_rate = 2e-4
    warmup_steps = 100
    lr_scheduler_type = "cosine"
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # Optimization
    optim = "adamw_torch"  # or "adamw_8bit" for memory savings
    fp16 = False
    bf16 = True  # Use bf16 for speed
    gradient_checkpointing = True
    
    # Logging and saving
    output_dir = "./mistral-7b-compliance"
    logging_steps = 10
    eval_steps = 500
    save_steps = 500
    save_total_limit = 2
    load_best_model_at_end = True
    metric_for_best_model = "eval_loss"
    
    # Data
    dataset_path = "./compliance_dataset"
    
    # Other
    seed = 42

config = Config()

# ============================================================================
# SETUP
# ============================================================================

print("="*80)
print("MISTRAL 7B COMPLIANCE FINE-TUNING")
print("="*80)
print(f"Model: {config.model_name}")
print(f"Max sequence length: {config.max_seq_length}")
print(f"Batch size: {config.per_device_train_batch_size}")
print(f"Learning rate: {config.learning_rate}")
print(f"Quantization: {'4-bit' if config.use_4bit else '8-bit' if config.use_8bit else 'None (bf16)'}")
print("="*80)

# Set seed
torch.manual_seed(config.seed)

# ============================================================================
# LOAD TOKENIZER
# ============================================================================

print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Required for causal LM

print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n📥 Loading model...")

# Quantization config (optional)
if config.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("Using 4-bit quantization")
elif config.use_8bit:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    print("Using 8-bit quantization")
else:
    bnb_config = None
    print("Using full precision (bf16)")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if not config.use_4bit else None,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # Remove if not supported
)

print(f"✓ Model loaded")
print(f"  Device map: {model.hf_device_map}")
print(f"  Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

# Prepare for k-bit training if using quantization
if config.use_4bit or config.use_8bit:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing
    )
    print("✓ Model prepared for k-bit training")
else:
    # Enable gradient checkpointing for full precision
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

# ============================================================================
# SETUP LORA
# ============================================================================

print("\n🔧 Setting up LoRA...")

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=config.lora_target_modules,
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

print("✓ LoRA adapters added")
model.print_trainable_parameters()

# ============================================================================
# LOAD AND PREPARE DATASET
# ============================================================================

print("\n📥 Loading dataset...")

# Load dataset
dataset = load_from_disk(config.dataset_path)

print(f"✓ Dataset loaded")
print(f"  Train: {len(dataset['train'])} examples")
print(f"  Validation: {len(dataset['validation'])} examples")
print(f"  Test: {len(dataset['test'])} examples")

# Tokenization function
def tokenize_function(examples):
    """
    Tokenize the text field
    """
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=config.max_seq_length,
        padding=False,  # Don't pad yet, handled by data collator
        return_tensors=None,
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("\n🔄 Tokenizing dataset...")

# Tokenize
tokenized_train = dataset['train'].map(
    tokenize_function,
    batched=True,
    num_proc=4,  # Parallel processing
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing train dataset",
)

tokenized_val = dataset['validation'].map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset['validation'].column_names,
    desc="Tokenizing validation dataset",
)

print(f"✓ Tokenization complete")
print(f"  Train tokens: ~{sum(len(x) for x in tokenized_train['input_ids']) / 1e6:.2f}M")

# ============================================================================
# DATA COLLATOR
# ============================================================================

# Data collator handles padding and creates batches
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal LM, not masked LM
)

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

print("\n⚙️ Setting up training arguments...")

training_args = TrainingArguments(
    # Output
    output_dir=config.output_dir,
    overwrite_output_dir=True,
    
    # Training
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    
    # Optimization
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    max_grad_norm=config.max_grad_norm,
    optim=config.optim,
    
    # Learning rate schedule
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_steps=config.warmup_steps,
    
    # Precision
    fp16=config.fp16,
    bf16=config.bf16,
    
    # Logging
    logging_dir=f"{config.output_dir}/logs",
    logging_steps=config.logging_steps,
    logging_first_step=True,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=config.eval_steps,
    
    # Saving
    save_strategy="steps",
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    load_best_model_at_end=config.load_best_model_at_end,
    metric_for_best_model=config.metric_for_best_model,
    greater_is_better=False,  # Lower eval_loss is better
    
    # Misc
    report_to="tensorboard",  # or "wandb"
    seed=config.seed,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    group_by_length=False,  # Can enable for efficiency
    ddp_find_unused_parameters=False,
    
    # For faster data loading
    dataloader_prefetch_factor=2,
)

print(f"✓ Training arguments configured")
print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
print(f"  Total optimization steps: {len(tokenized_train) // (config.per_device_train_batch_size * config.gradient_accumulation_steps) * config.num_train_epochs}")

# ============================================================================
# TRAINER
# ============================================================================

print("\n🚀 Initializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("✓ Trainer initialized")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

# Check GPU memory before training
if torch.cuda.is_available():
    print(f"\n📊 GPU Memory before training:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Train
trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n💾 Saving model...")

# Save LoRA adapters
output_dir = f"{config.output_dir}/final"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Model saved to {output_dir}")

# Save training history
import json
with open(f"{output_dir}/training_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"✓ Training history saved")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

print("\n📊 Evaluating on test set...")

# Tokenize test set
tokenized_test = dataset['test'].map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=dataset['test'].column_names,
    desc="Tokenizing test dataset",
)

# Evaluate
test_results = trainer.evaluate(tokenized_test)

print("\n" + "="*80)
print("TEST SET RESULTS")
print("="*80)
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# Save test results
with open(f"{output_dir}/test_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

print(f"\n✓ Test results saved to {output_dir}/test_results.json")

# ============================================================================
# MERGE LORA WEIGHTS (OPTIONAL - FOR DEPLOYMENT)
# ============================================================================

print("\n🔀 Merging LoRA weights for deployment...")

# Merge LoRA weights into base model for easier deployment
merged_model = model.merge_and_unload()

# Save merged model
merged_output_dir = f"{config.output_dir}/merged"
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)

print(f"✓ Merged model saved to {merged_output_dir}")

print("\n" + "="*80)
print("ALL DONE! 🎉")
print("="*80)
print(f"\nYou can now use your model from:")
print(f"  LoRA adapters: {output_dir}")
print(f"  Merged model: {merged_output_dir}")
```

---

## **Optimized Version (Maximum Speed)**

If you want to squeeze out maximum performance:

```python
"""
MAXIMUM SPEED CONFIGURATION
For 23GB GPU - No quantization, large batch size
"""

class MaxSpeedConfig:
    # Model
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # NO quantization - use full bf16
    use_4bit = False
    use_8bit = False
    
    # LoRA settings
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # MAXIMIZED batch size
    max_seq_length = 2048
    per_device_train_batch_size = 32  # Max for 23GB GPU
    per_device_eval_batch_size = 32
    gradient_accumulation_steps = 1
    
    # Training
    num_train_epochs = 3
    learning_rate = 2.5e-4  # Slightly higher for larger batch
    warmup_steps = 150
    lr_scheduler_type = "cosine"
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # FASTEST optimization settings
    optim = "adamw_torch"  # Fastest optimizer
    fp16 = False
    bf16 = True  # Fastest precision
    gradient_checkpointing = False  # Disable for max speed (uses more memory)
    
    # Flash Attention 2 for speed
    use_flash_attention = True
    
    # Aggressive data loading
    dataloader_num_workers = 8  # More workers
    dataloader_prefetch_factor = 4  # More prefetching
    
    # Less frequent evaluation
    eval_steps = 1000
    save_steps = 1000
    logging_steps = 50
    
    # Output
    output_dir = "./mistral-7b-compliance-fast"
    save_total_limit = 2
    load_best_model_at_end = True
    
    # Data
    dataset_path = "./compliance_dataset"
    seed = 42

# Use this config instead
config = MaxSpeedConfig()
```

---

## **Inference Script (After Fine-Tuning)**

```python
"""
Inference with fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

class ComplianceDetector:
    def __init__(self, base_model_name, lora_adapter_path=None, merged_model_path=None):
        """
        Initialize detector with either LoRA adapters or merged model
        
        Args:
            base_model_name: Base model (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
            lora_adapter_path: Path to LoRA adapters (if using adapters)
            merged_model_path: Path to merged model (if using merged)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            merged_model_path if merged_model_path else base_model_name
        )
        
        if merged_model_path:
            # Load merged model
            print(f"Loading merged model from {merged_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                merged_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            # Load base model + LoRA adapters
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            print(f"Loading LoRA adapters from {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        self.model.eval()
        print("✓ Model loaded and ready")
    
    def analyze(self, conversation, max_new_tokens=1024, temperature=0.7):
        """
        Analyze a conversation for compliance violations
        """
        # Create prompt
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
        
        # Format for Mistral
        formatted_prompt = f"<s>[INST] {prompt} [/INST]\n"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                result = {"error": "No JSON found in response", "raw": response}
        except json.JSONDecodeError as e:
            result = {"error": f"JSON decode error: {str(e)}", "raw": response}
        
        return result

# Usage
if __name__ == "__main__":
    # Option 1: Load with LoRA adapters
    detector = ComplianceDetector(
        base_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        lora_adapter_path="./mistral-7b-compliance/final"
    )
    
    # Option 2: Load merged model (faster inference)
    # detector = ComplianceDetector(
    #     merged_model_path="./mistral-7b-compliance/merged"
    # )
    
    # Test conversation
    conversation = """
    Client: What kind of returns can I expect from this fund?
    
    Advisor: This fund will definitely deliver 10% annually. It's completely safe,
    you can't lose money with this investment. I guarantee you'll see great returns.
    """
    
    # Analyze
    result = detector.analyze(conversation)
    
    # Print result
    print("\n" + "="*80)
    print("ANALYSIS RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))
```

---

## **Memory Monitoring Script**

```python
"""
Monitor GPU memory during training
"""

import torch
from threading import Thread
import time

class GPUMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.peak_memory = 0
        
    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.thread.join()
        print(f"\n📊 Peak GPU Memory: {self.peak_memory / 1e9:.2f} GB")
    
    def _monitor(self):
        while self.runn
