#!/usr/bin/env python3
"""
GLiNER Fine-tuning Script for Legal NER
Simple, optimized version with constants at the top
"""

import argparse
import json
import os
import random
import warnings
from typing import List, Dict, Any
import numpy as np
import torch
from datasets import load_dataset
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollator
import wandb

# ============================================================================
# CONFIGURATION - ALL CONSTANTS DEFINED HERE
# ============================================================================

MODEL_NAME = "knowledgator/gliner-x-small"
OUTPUT_DIR = "./gliner-legal-finetuned"
DATASET_NAME = "liechticonsulting/NER_FILTERED_DATASET"
LABELS = ["doctrine", "jurisprudence", "articles de loi"]

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 4  # Same for train and eval
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch = 4 GPUs * 4 batch * 4 accum = 64
LEARNING_RATE = 5e-6
LEARNING_RATE_OTHERS = 1e-5  # For non-encoder parts
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# Data split
TRAIN_SPLIT = 0.9
RANDOM_SEED = 42

# Logging and checkpointing
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 3

# Performance optimization
NUM_WORKERS = 8  # Increase for faster data loading
USE_FP16 = True
PIN_MEMORY = True

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Suppress GLiNER truncation warnings - they're expected for long legal docs
warnings.filterwarnings("ignore", message=".*has been truncated.*")


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_token_spans(text, entity_text, tokens):
    """Find token indices for an entity text within tokenized text."""
    char_start = text.find(entity_text)
    if char_start == -1:
        return None
    char_end = char_start + len(entity_text)

    current_pos = 0
    start_token_idx = None
    end_token_idx = None

    for idx, token in enumerate(tokens):
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1

        token_start = current_pos
        token_end = current_pos + len(token)

        if start_token_idx is None and token_start < char_end and token_end > char_start:
            start_token_idx = idx

        if start_token_idx is not None:
            if token_end > char_start:
                end_token_idx = idx
            if token_end >= char_end:
                break

        current_pos = token_end

    if start_token_idx is not None and end_token_idx is not None:
        return [start_token_idx, end_token_idx]
    return None


def convert_to_gliner_format(examples):
    """Convert dataset to GLiNER format."""
    gliner_data = []

    for idx in range(len(examples['part_content'])):
        text = examples['part_content'][idx]
        analysis_str = examples['analysis'][idx]

        if not text or not analysis_str:
            continue

        try:
            analysis = json.loads(analysis_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for index {idx}")
            continue

        tokens = text.split()
        if not tokens:
            continue

        ner_entities = []
        for label in LABELS:
            if label not in analysis:
                continue

            entities = analysis[label]
            if not isinstance(entities, list):
                continue

            for entity_text in entities:
                if not entity_text:
                    continue

                span = find_token_spans(text, entity_text, tokens)
                if span is not None:
                    ner_entities.append([span[0], span[1], label])

        gliner_data.append({
            'tokenized_text': tokens,
            'ner': ner_entities
        })

    return gliner_data


def prepare_datasets():
    """Load and prepare datasets."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    print("Converting to GLiNER format...")
    gliner_data = convert_to_gliner_format(dataset)

    print(f"Converted {len(gliner_data)} examples")

    # Filter empty examples
    gliner_data = [item for item in gliner_data if len(item['tokenized_text']) > 0]

    print(f"After filtering: {len(gliner_data)} examples")

    # Stats
    with_entities = sum(1 for item in gliner_data if len(item['ner']) > 0)
    print(f"Examples with entities: {with_entities}")
    print(f"Examples without entities: {len(gliner_data) - with_entities}")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(gliner_data)

    split_idx = int(len(gliner_data) * TRAIN_SPLIT)
    train_data = gliner_data[:split_idx]
    val_data = gliner_data[split_idx:]

    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")

    return train_data, val_data


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(use_wandb):
    """Train the GLiNER model."""
    set_seed(RANDOM_SEED)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="gliner-legal-ner",
            name="gliner-x-small-legal",
            config={
                "model": MODEL_NAME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            }
        )

    # Prepare datasets
    train_data, val_data = prepare_datasets()

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model = GLiNER.from_pretrained(MODEL_NAME)

    # Save data samples
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "train_sample.json"), "w") as f:
        json.dump(train_data[:5], f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, "val_sample.json"), "w") as f:
        json.dump(val_data[:5], f, indent=2, ensure_ascii=False)

    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    print(f"\nDevice: {device}")
    print(f"Number of GPUs: {num_gpus}")

    effective_batch = BATCH_SIZE * num_gpus * GRADIENT_ACCUMULATION_STEPS if num_gpus > 0 else BATCH_SIZE
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Data workers: {NUM_WORKERS}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        others_lr=LEARNING_RATE_OTHERS,
        others_weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        dataloader_num_workers=NUM_WORKERS,
        dataloader_pin_memory=PIN_MEMORY,
        use_cpu=False,
        report_to="wandb" if use_wandb else "none",
        logging_steps=LOGGING_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=USE_FP16,
    )

    # Data collator
    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.train()

    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    if use_wandb:
        wandb.log({"final_eval": eval_results})
        wandb.finish()

    print("\nTraining complete!")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Finetune GLiNER for legal NER")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    args = parser.parse_args()

    print("=" * 80)
    print("GLiNER Legal NER Fine-tuning")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Labels: {LABELS}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 80)
    print("\nNote: Truncation warnings for long documents are normal and expected.")
    print("Legal texts often exceed the 1024 token limit - this is by design.\n")
    print("=" * 80)

    train_model(use_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
