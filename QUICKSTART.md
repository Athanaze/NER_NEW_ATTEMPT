# Quick Start Guide

Get started with finetuning GLiNER for legal NER in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Login to Wandb (Optional)

```bash
wandb login
```

Or skip wandb logging:
```bash
# Add --no_wandb flag when training
```

## Step 3: Start Training

```bash
python train_gliner.py
```

That's it! The script will:
- ✅ Download the dataset automatically
- ✅ Preprocess and convert to GLiNER format
- ✅ Split into train/validation (90/10)
- ✅ Detect and use all available GPUs
- ✅ Log training progress to wandb
- ✅ Save the best model based on validation loss

## Expected Output

```
Loading dataset from HuggingFace...
Dataset loaded: 19400 examples
Converting to GLiNER format...
Converted 19400 examples
Train set: 17460 examples
Validation set: 1940 examples
Loading model: knowledgator/gliner-x-small
Device: cuda
Number of GPUs: 4
Starting training...
```

## After Training

Test your model:

```bash
python inference.py --model_path ./gliner-legal-finetuned
```

Or with your own text:

```bash
python inference.py --text "Your legal text here" --threshold 0.5
```

## Customization

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for advanced options and troubleshooting.

## File Structure

```
.
├── train_gliner.py          # Main training script
├── inference.py             # Inference script
├── requirements.txt         # Python dependencies
├── TRAINING_GUIDE.md        # Detailed training guide
├── QUICKSTART.md           # This file
└── gliner-legal-finetuned/ # Output directory (created after training)
    ├── pytorch_model.bin
    ├── config.json
    └── ...
```
