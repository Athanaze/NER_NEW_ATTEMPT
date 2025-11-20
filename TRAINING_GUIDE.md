# GLiNER Legal NER Fine-tuning Guide

This guide explains how to finetune the `knowledgator/gliner-x-small` model on the legal NER dataset for detecting **doctrine**, **jurisprudence**, and **articles de loi** entities.

## Requirements

### Hardware
- **Recommended**: 4x RTX 4090 GPUs (24GB VRAM each)
- **Minimum**: 1x GPU with at least 8GB VRAM

### Software
- Python 3.8+
- CUDA 11.8+ or 12.0+
- PyTorch 2.0+

## Installation

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Login to Weights & Biases** (optional but recommended for tracking):
```bash
wandb login
```

## Dataset

The script automatically downloads the dataset from HuggingFace:
- **Dataset**: `liechticonsulting/NER_FILTERED_DATASET`
- **Labels**: doctrine, jurisprudence, articles de loi
- **Format**: JSON annotations with exact substring matches in the `analysis` column

## Training

### Basic Training Command

For training with default parameters on 2 GPUs:

```bash
python train_gliner.py
```

### Advanced Training Options

```bash
python train_gliner.py \
    --output_dir ./gliner-legal-finetuned \
    --model_name knowledgator/gliner-x-small \
    --num_epochs 10 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 100 \
    --gradient_accumulation_steps 2 \
    --seed 42
```

### Training Without Wandb

If you don't want to use Weights & Biases:

```bash
python train_gliner.py --no_wandb
```

### Multi-GPU Training

The script automatically detects and uses all available GPUs with PyTorch DataParallel. With 4x RTX 4090:
- Effective batch size = `train_batch_size` × `num_gpus` × `gradient_accumulation_steps`
- Example: 8 × 4 × 2 = 64 effective batch size

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output_dir` | `./gliner-legal-finetuned` | Directory to save the model |
| `--model_name` | `knowledgator/gliner-x-small` | Pre-trained model to finetune |
| `--num_epochs` | 10 | Number of training epochs |
| `--train_batch_size` | 8 | Batch size per GPU for training |
| `--eval_batch_size` | 16 | Batch size per GPU for evaluation |
| `--learning_rate` | 5e-6 | Learning rate for the encoder |
| `--weight_decay` | 0.01 | Weight decay for regularization |
| `--warmup_ratio` | 0.1 | Proportion of training for warmup |
| `--gradient_accumulation_steps` | 2 | Steps to accumulate gradients |
| `--eval_steps` | 500 | Evaluate every N steps |
| `--save_steps` | 500 | Save checkpoint every N steps |

## Data Processing

The script performs the following data processing steps:

1. **Loading**: Downloads dataset from HuggingFace
2. **Tokenization**: Splits text into word-level tokens
3. **Entity Mapping**: Converts exact substring matches to token-based spans [start_idx, end_idx, label]
4. **Splitting**: 90% train, 10% validation
5. **Filtering**: Includes both examples with and without entities to reduce false positives

## Monitoring Training

### With Wandb (Default)

1. Visit your Weights & Biases dashboard at https://wandb.ai
2. Look for the project `gliner-legal-ner`
3. Monitor:
   - Training loss
   - Validation loss
   - Learning rate
   - Gradient norms

### Without Wandb

Training logs will be printed to console with:
- Loss values every `--logging_steps`
- Validation results every `--eval_steps`

## Output

After training, you'll find in the output directory:

- `pytorch_model.bin` - The finetuned model weights
- `config.json` - Model configuration
- `training_args.bin` - Training arguments used
- `train_sample.json` - Sample of training data (for inspection)
- `val_sample.json` - Sample of validation data (for inspection)
- `checkpoint-*` - Intermediate checkpoints (last 3 saved)

## Using the Finetuned Model

After training, you can use the model for inference:

```python
from gliner import GLiNER

# Load the finetuned model
model = GLiNER.from_pretrained("./gliner-legal-finetuned")

# Predict entities
text = """Entscheid vom 12. April 2021 Besetzung lic.iur. Gion Tomaschett,
Vizepräsident Dr.med. Urs Gössi, Richter Dr.med. Pierre Lichtenhahn, Richter
MLaw Tanja Marty, a.o. Gerichtsschreiberin Parteien A."""

labels = ["doctrine", "jurisprudence", "articles de loi"]
entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(f"{entity['text']} => {entity['label']}")
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:
1. Reduce `--train_batch_size` (try 2 or 1)
2. Increase `--gradient_accumulation_steps` to maintain effective batch size
3. Reduce `--eval_batch_size`

### Slow Training

If training is too slow:
1. Increase `--train_batch_size` if you have VRAM available
2. Reduce `--eval_steps` to evaluate less frequently
3. Ensure you're using GPU (check with `nvidia-smi`)

### Poor Performance

If the model doesn't perform well:
1. Increase `--num_epochs` (try 15-20)
2. Adjust `--learning_rate` (try 3e-6 or 1e-5)
3. Check the data samples in `train_sample.json` to verify correct formatting
4. Monitor wandb to check if the model is overfitting or underfitting

## Expected Training Time

On 4x RTX 4090:
- **Dataset size**: ~19.4k examples
- **Model**: gliner-x-small (smaller, faster than x-large)
- **Epochs**: 10
- **Estimated time**: 1-2 hours (depends on sequence lengths)

## Performance Tips

1. **Use mixed precision training**: The script uses fp16 when available
2. **Optimize data loading**: Set `dataloader_num_workers=4` (adjust based on CPU cores)
3. **Monitor GPU utilization**: Use `nvidia-smi -l 1` in another terminal
4. **Adjust batch size**: Find the maximum batch size that fits in VRAM for faster training

## Citation

If you use this finetuned model in your research, please cite:

```bibtex
@inproceedings{zaratiana2023gliner,
    title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer},
    author={Urchade Zaratiana and Nadi Tomeh and Pierre Holat and Thierry Charnois},
    booktitle={Proceedings of NAACL 2024},
    year={2024}
}
```

## License

This training script is provided as-is. The GLiNER model follows the Apache 2.0 license.
