Transformer Implementation (from Scratch)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/Attention-is-all-you-need-implementation-/blob/main/notebook.ipynb)

A complete, ground-up implementation of the Transformer architecture as proposed in the paper **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)**. 

This project implements the full Encoder-Decoder architecture for machine translation (English to Italian) without relying on high-level `nn.Transformer` abstractions, focusing on the mathematical building blocks.

## 🏗️ Architecture

The implementation details include:
- **Input Embeddings & Positional Encoding**: capturing semantic meaning and sequence order.
- **Multi-Head Attention**: implementing the split-head mechanism to attend to different parts of the sequence.
- **Layer Normalization & Residual Connections**: strictly following the "Add & Norm" structure.
- **Feed-Forward Networks**: position-wise fully connected networks.
- **Masking**: implementing both padding masks (encoder) and causal masks (decoder).

## 📂 File Structure

```text
.
├── config.py         # Configuration (Hyperparameters, file paths)
├── dataset.py        # Hugging Face dataset loading and processing
├── model.py          # The core Transformer architecture classes
├── train.py          # Training loop, validation, and checkpointing
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/aprv10/Attention-is-all-you-need-implementation-.git
cd Attention-is-all-you-need-implementation-
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio datasets tokenizers altair tqdm
```

### 3. Train the Model
The model is configured to train on the **Opus Books (en-it)** dataset.
```bash
python train.py
```
*Note: The first run will download the dataset and build the tokenizer automatically.*

## 📊 Performance & Visualization

The training loop includes validation steps that:
1. Perform greedy decoding to translate validation sentences.
2. Calculate input-output length constraints.
3. (Optional) Visualize attention weights using **Altair** to analyze which source words the model focuses on during translation.

## 🧠 References

1. **Paper**: Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
2. **Tutorial**: Code implementation based on the detailed walkthrough by [Umar Jamil](https://www.youtube.com/watch?v=ISNdQcPhsts).

---
