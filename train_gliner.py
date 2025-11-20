#!/usr/bin/env python3
"""
GLiNER Fine-tuning Script for Legal NER
Finetunes knowledgator/gliner-x-large on liechticonsulting/NER_FILTERED_DATASET
for labels: doctrine, jurisprudence, articles de loi
"""

import argparse
import json
import os
import random
from typing import List, Dict, Any
import numpy as np
import torch
from datasets import load_dataset
from gliner import GLiNER, GLiNERConfig
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
import wandb


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_token_spans(text: str, entity_text: str, tokens: List[str]) -> List[int]:
    """
    Find the token indices for an entity text within the tokenized text.

    Args:
        text: The original text
        entity_text: The entity substring to find
        tokens: List of tokens from the text

    Returns:
        [start_idx, end_idx] or None if not found
    """
    # Find character positions
    char_start = text.find(entity_text)
    if char_start == -1:
        return None
    char_end = char_start + len(entity_text)

    # Reconstruct text from tokens to find token positions
    current_pos = 0
    start_token_idx = None
    end_token_idx = None

    for idx, token in enumerate(tokens):
        # Skip whitespace in original text
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1

        token_start = current_pos
        token_end = current_pos + len(token)

        # Check if this token overlaps with the entity
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


def tokenize_text(text: str) -> List[str]:
    """
    Simple whitespace tokenization.
    Note: GLiNER will use its own tokenizer internally, but we need word-level tokens.
    """
    return text.split()


def convert_to_gliner_format(examples: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Convert HuggingFace dataset to GLiNER training format.

    Expected input columns:
    - part_content: The text content
    - analysis: JSON string with labels {"doctrine": [...], "jurisprudence": [...], "articles de loi": [...]}

    Output format:
    - tokenized_text: List of tokens
    - ner: List of [start_idx, end_idx, label] for each entity
    """
    gliner_data = []
    labels_of_interest = ["doctrine", "jurisprudence", "articles de loi"]

    for idx in range(len(examples['part_content'])):
        text = examples['part_content'][idx]
        analysis_str = examples['analysis'][idx]

        if not text or not analysis_str:
            continue

        # Parse the analysis JSON
        try:
            analysis = json.loads(analysis_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for index {idx}")
            continue

        # Tokenize the text
        tokens = tokenize_text(text)

        if not tokens:
            continue

        # Extract entities
        ner_entities = []
        for label in labels_of_interest:
            if label not in analysis:
                continue

            entities = analysis[label]
            if not isinstance(entities, list):
                continue

            for entity_text in entities:
                if not entity_text:
                    continue

                # Find token spans for this entity
                span = find_token_spans(text, entity_text, tokens)
                if span is not None:
                    # Format: [start_idx, end_idx, label]
                    ner_entities.append([span[0], span[1], label])

        # Add to dataset (include examples without entities to reduce false positives)
        gliner_data.append({
            'tokenized_text': tokens,
            'ner': ner_entities
        })

    return gliner_data


def prepare_datasets(train_size: float = 0.9, seed: int = 42):
    """
    Load and prepare the dataset from HuggingFace.

    Args:
        train_size: Proportion of data to use for training (rest for validation)
        seed: Random seed for splitting

    Returns:
        train_data, val_data: Lists of examples in GLiNER format
    """
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("liechticonsulting/NER_FILTERED_DATASET", split="train")

    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"Columns: {dataset.column_names}")

    # Convert to GLiNER format
    print("Converting to GLiNER format...")
    gliner_data = convert_to_gliner_format(dataset)

    print(f"Converted {len(gliner_data)} examples")

    # Filter out examples with no tokenized text
    gliner_data = [item for item in gliner_data if len(item['tokenized_text']) > 0]

    print(f"After filtering: {len(gliner_data)} examples")

    # Count examples with/without entities
    with_entities = sum(1 for item in gliner_data if len(item['ner']) > 0)
    without_entities = len(gliner_data) - with_entities
    print(f"Examples with entities: {with_entities}")
    print(f"Examples without entities: {without_entities}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(gliner_data)

    split_idx = int(len(gliner_data) * train_size)
    train_data = gliner_data[:split_idx]
    val_data = gliner_data[split_idx:]

    print(f"Train set: {len(train_data)} examples")
    print(f"Validation set: {len(val_data)} examples")

    return train_data, val_data


def train_model(
    output_dir: str = "./gliner-legal-finetuned",
    model_name: str = "knowledgator/gliner-x-large",
    num_train_epochs: int = 10,
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    eval_steps: int = 500,
    save_steps: int = 500,
    logging_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    use_wandb: bool = True,
    deepspeed: str = None,
    seed: int = 42,
):
    """
    Train the GLiNER model.

    Args:
        output_dir: Directory to save the model
        model_name: Pre-trained model to finetune
        num_train_epochs: Number of training epochs
        train_batch_size: Batch size per GPU for training
        eval_batch_size: Batch size per GPU for evaluation
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_ratio: Ratio of warmup steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_grad_norm: Max gradient norm for clipping
        use_wandb: Whether to use Weights & Biases for logging
        seed: Random seed
    """
    set_seed(seed)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="gliner-legal-ner",
            name=f"gliner-x-large-legal-finetuning",
            config={
                "model": model_name,
                "epochs": num_train_epochs,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "seed": seed,
            }
        )

    # Prepare datasets
    train_data, val_data = prepare_datasets(train_size=0.9, seed=seed)

    # Load model
    print(f"Loading model: {model_name}")
    model = GLiNER.from_pretrained(model_name)

    # Save sample of data for inspection
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_sample.json"), "w") as f:
        json.dump(train_data[:5], f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "val_sample.json"), "w") as f:
        json.dump(val_data[:5], f, indent=2, ensure_ascii=False)

    # Determine device strategy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    print(f"Device: {device}")
    print(f"Number of GPUs: {num_gpus}")

    # Adjust batch size for multiple GPUs
    # With 2x RTX 4090 (24GB each), we can use moderate batch sizes
    effective_train_batch = train_batch_size * num_gpus if num_gpus > 0 else train_batch_size
    effective_eval_batch = eval_batch_size * num_gpus if num_gpus > 0 else eval_batch_size

    print(f"Effective train batch size: {effective_train_batch} (per_device: {train_batch_size})")
    print(f"Effective eval batch size: {effective_eval_batch} (per_device: {eval_batch_size})")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        others_lr=1e-5,  # Learning rate for non-encoder parts
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,  # Keep only last 3 checkpoints
        dataloader_num_workers=4,
        use_cpu=False,
        report_to="wandb" if use_wandb else "none",
        logging_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,  # Enable mixed precision training
        deepspeed=deepspeed,  # DeepSpeed config for model sharding
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
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

    # Final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")

    if use_wandb:
        wandb.log({"final_eval": eval_results})
        wandb.finish()

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Finetune GLiNER for legal NER")
    parser.add_argument("--output_dir", type=str, default="./gliner-legal-finetuned",
                        help="Directory to save the finetuned model")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliner-x-large",
                        help="Pre-trained model to finetune")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Evaluation batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config file for model sharding")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    train_model(
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_train_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        use_wandb=not args.no_wandb,
        deepspeed=args.deepspeed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
