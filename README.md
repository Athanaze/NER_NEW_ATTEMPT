# GLiNER Legal NER Fine-tuning

Fine-tune GLiNER (Generalist and Lightweight Model for Named Entity Recognition) for legal document entity extraction in Swiss legal texts.

## Overview

This project finetunes the `knowledgator/gliner-x-small` model on legal documents to detect three key entity types:
- **doctrine**: Legal academic writings and scholarly sources
- **jurisprudence**: Case law and judicial decisions
- **articles de loi**: Legal articles and statutory provisions

## Dataset

- **Source**: [liechticonsulting/NER_FILTERED_DATASET](https://huggingface.co/datasets/liechticonsulting/NER_FILTERED_DATASET)
- **Size**: ~19,400 legal documents
- **Format**: JSON annotations with exact substring matches
- **Language**: Primarily German, French, and Italian (Swiss legal documents)

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Basic training (uses all default settings)
python train_gliner.py

# Training without wandb logging
python train_gliner.py --no_wandb

# Custom parameters
python train_gliner.py \
    --num_epochs 15 \
    --train_batch_size 16 \
    --learning_rate 3e-6
```

### 3. Use the Model

```python
from gliner import GLiNER

# Load finetuned model
model = GLiNER.from_pretrained("./gliner-legal-finetuned")

# Your legal text
text = """Entscheid vom 12. April 2021 Besetzung lic.iur. Gion Tomaschett,
Vizepräsident Dr.med. Urs Gössi, Richter Dr.med. Pierre Lichtenhahn, Richter
MLaw Tanja Marty, a.o. Gerichtsschreiberin Parteien A."""

labels = ["doctrine", "jurisprudence", "articles de loi"]
entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(f"{entity['text']} => {entity['label']}")
```

## Hardware Requirements

- **Recommended**: 4x RTX 4090 GPUs (24GB VRAM each)
- **Minimum**: 1x GPU with 8GB+ VRAM
- **Training time**: 1-2 hours on 4x RTX 4090

## Model Details

- **Base Model**: [knowledgator/gliner-x-small](https://huggingface.co/knowledgator/gliner-x-small)
- **Architecture**: Bidirectional transformer encoder (BERT-like)
- **Approach**: Zero-shot NER with entity type prompting
- **Training**: Multi-GPU with PyTorch DataParallel and FP16 mixed precision

## Default Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | `gliner-x-small` | Lightweight, fast inference |
| Epochs | 10 | Full passes through dataset |
| Batch Size (train) | 8 per GPU | With 4 GPUs = 32 total |
| Batch Size (eval) | 16 per GPU | With 4 GPUs = 64 total |
| Gradient Accumulation | 2 steps | Effective batch = 64 |
| Learning Rate | 5e-6 | For encoder layers |
| Warmup Ratio | 0.1 | Linear warmup |
| FP16 | Enabled | Mixed precision training |

## Files

```
.
├── train_gliner.py          # Main training script
├── inference.py             # Inference script for testing
├── requirements.txt         # Python dependencies
├── TRAINING_GUIDE.md        # Detailed training documentation
├── QUICKSTART.md           # Quick start guide
├── CLAUDE.md               # Original project instructions
└── gliner-legal-finetuned/ # Output directory (after training)
    ├── pytorch_model.bin
    ├── config.json
    └── ...
```

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - Get up and running in 3 steps
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Comprehensive training guide with advanced options

## Results

After training, the model will:
- ✅ Detect legal doctrine references in Swiss legal texts
- ✅ Extract jurisprudence citations and case law references
- ✅ Identify specific articles of law and statutory provisions
- ✅ Work across German, French, and Italian legal documents
- ✅ Support zero-shot detection of new entity types

## Citation

If you use this finetuned model in your research, please cite the original GLiNER paper:

```bibtex
@inproceedings{zaratiana2023gliner,
    title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer},
    author={Urchade Zaratiana and Nadi Tomeh and Pierre Holat and Thierry Charnois},
    booktitle={Proceedings of NAACL 2024},
    year={2024}
}
```

## License

This project is released under the Apache 2.0 License. The GLiNER model follows the same license.

## Troubleshooting

**Out of Memory (OOM)?**
- Reduce `--train_batch_size` (try 4 or 2)
- Increase `--gradient_accumulation_steps` proportionally
- Reduce `--eval_batch_size`

**Poor Performance?**
- Increase `--num_epochs` (try 15-20)
- Adjust `--learning_rate` (try 3e-6 or 1e-5)
- Check `train_sample.json` to verify data format

**Slow Training?**
- Increase batch sizes if you have VRAM
- Reduce `--eval_steps` to evaluate less frequently
- Verify GPU usage with `nvidia-smi`

## Contributing

This is a research project. For issues or improvements, please open an issue or pull request.
