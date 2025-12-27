import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk
from accelerate import Accelerator
import os

# ============================================================================
# INITIALIZE ACCELERATOR
# ============================================================================

accelerator = Accelerator()

# Only print from main process
def print_main(msg):
    if accelerator.is_main_process:
        print(msg)

print_main("="*80)
print_main("DISTRIBUTED TRAINING WITH ACCELERATE")
print_main("="*80)
print_main(f"Number of processes: {accelerator.num_processes}")
print_main(f"Process index: {accelerator.process_index}")
print_main(f"Device: {accelerator.device}")
print_main("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_length = 1024  # Adjust based on your data
    
    # Data
    dataset_path = "./dataset"
    
    # Training
    per_device_train_batch_size = 2  # Per GPU
    per_device_eval_batch_size = 2
    gradient_accumulation_steps = 4  # Effective batch = 8 per GPU
    num_train_epochs = 3
    learning_rate = 2e-4
    warmup_steps = 100
    max_grad_norm = 1.0
    
    # LoRA
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    # Output
    output_dir = "./mistral-accelerate"
    
    # Optimization
    use_4bit = True  # Use 4-bit quantization
    use_gradient_checkpointing = True
    
    # Other
    seed = 42
    logging_steps = 10
    save_steps = 500
    eval_steps = 500

config = Config()

# ============================================================================
# QUANTIZATION CONFIG (4-bit)
# ============================================================================

if config.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print_main("Using 4-bit quantization")
else:
    bnb_config = None
    print_main("Using bfloat16 (no quantization)")

# ============================================================================
# LOAD TOKENIZER
# ============================================================================

print_main("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True,
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"

print_main(f"✓ Tokenizer loaded")
print_main(f"  Vocab size: {len(tokenizer)}")
print_main(f"  Pad token: {tokenizer.pad_token}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print_main("\nLoading model...")

model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    quantization_config=bnb_config if config.use_4bit else None,
    torch_dtype=torch.bfloat16 if not config.use_4bit else None,
    trust_remote_code=True,
    device_map={"": accelerator.process_index},  # Important for multi-GPU
)

print_main(f"✓ Model loaded")
print_main(f"  Model dtype: {model.dtype}")
print_main(f"  Device: {model.device}")

# ============================================================================
# PREPARE MODEL FOR TRAINING
# ============================================================================

# Enable gradient checkpointing
if config.use_gradient_checkpointing:
    model.gradient_checkpointing_enable()
    print_main("✓ Gradient checkpointing enabled")

# Prepare for k-bit training (if using quantization)
if config.use_4bit:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
    )
    print_main("✓ Model prepared for 4-bit training")

# ============================================================================
# APPLY LORA
# ============================================================================

print_main("\nApplying LoRA...")

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=config.lora_target_modules,
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print_main("✓ LoRA applied")

# Print trainable parameters
if accelerator.is_main_process:
    model.print_trainable_parameters()

# ============================================================================
# LOAD DATASET
# ============================================================================

print_main("\nLoading dataset...")

dataset = load_from_disk(config.dataset_path)

print_main(f"✓ Dataset loaded")
print_main(f"  Train: {len(dataset['train'])} examples")
print_main(f"  Validation: {len(dataset['validation'])} examples")

# Filter sequences that are too long
def filter_length(example):
    """Filter out sequences longer than max_seq_length"""
    tokens = tokenizer(example['text'], truncation=False)['input_ids']
    return len(tokens) <= config.max_seq_length

if accelerator.is_main_process:
    print_main("\nFiltering long sequences...")
    original_train = len(dataset['train'])
    original_val = len(dataset['validation'])
    
    dataset['train'] = dataset['train'].filter(filter_length)
    dataset['validation'] = dataset['validation'].filter(filter_length)
    
    print_main(f"  Train: {original_train} → {len(dataset['train'])}")
    print_main(f"  Val: {original_val} → {len(dataset['validation'])}")

# Wait for main process to finish filtering
accelerator.wait_for_everyone()

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=config.output_dir,
    
    # Batch sizes
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    
    # Training schedule
    num_train_epochs=config.num_train_epochs,
    max_steps=-1,
    
    # Learning rate
    learning_rate=config.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=config.warmup_steps,
    
    # Optimization
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    weight_decay=0.01,
    max_grad_norm=config.max_grad_norm,
    
    # Precision
    bf16=True,
    fp16=False,
    
    # Logging
    logging_steps=config.logging_steps,
    logging_first_step=True,
    
    # Evaluation
    evaluation_strategy="steps",
    eval_steps=config.eval_steps,
    
    # Saving
    save_strategy="steps",
    save_steps=config.save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Other
    seed=config.seed,
    data_seed=config.seed,
    
    # Reporting
    report_to="tensorboard",
    
    # DataLoader
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Disable DDP-specific settings (Accelerate handles it)
    ddp_find_unused_parameters=False,
    
    # Gradient checkpointing
    gradient_checkpointing=config.use_gradient_checkpointing,
)

# ============================================================================
# TRAINER
# ============================================================================

print_main("\nInitializing trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    dataset_text_field="text",
    max_seq_length=config.max_seq_length,
    packing=False,  # Don't pack sequences (simpler, more stable)
)

print_main("✓ Trainer initialized")

# ============================================================================
# TRAIN
# ============================================================================

print_main("\n" + "="*80)
print_main("STARTING TRAINING")
print_main("="*80)

if accelerator.is_main_process:
    effective_batch_size = (
        config.per_device_train_batch_size 
        * config.gradient_accumulation_steps 
        * accelerator.num_processes
    )
    print_main(f"Per-device batch size: {config.per_device_train_batch_size}")
    print_main(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print_main(f"Number of GPUs: {accelerator.num_processes}")
    print_main(f"Effective batch size: {effective_batch_size}")
    print_main(f"Total optimization steps: ~{len(dataset['train']) * config.num_train_epochs // effective_batch_size}")

print_main("="*80 + "\n")

# Train
trainer.train()

print_main("\n✓ Training complete!")

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

if accelerator.is_main_process:
    print_main("\nSaving final model...")
    
    final_output_dir = f"{config.output_dir}/final"
    
    # Save model
    trainer.model.save_pretrained(final_output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(final_output_dir)
    
    print_main(f"✓ Model saved to {final_output_dir}")

print_main("\n" + "="*80)
print_main("DONE!")
print_main("="*80)

