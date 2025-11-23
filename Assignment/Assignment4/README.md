# Assignment 4: Decoder-Only Transformer Language Model

## Overview

This assignment implements a decoder-only Transformer model for text generation, trained on Wikipedia sentences. The architecture is similar to GPT (Generative Pre-trained Transformer) and demonstrates key concepts in modern natural language processing.

## Table of Contents

- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Key Components](#key-components)
- [Implementation Details](#implementation-details)
- [Training Configuration](#training-configuration)
- [Usage](#usage)
- [Requirements](#requirements)
- [Results](#results)

## Project Structure

```
Assignment4/
├── Assignment4.ipynb    # Main implementation notebook
└── README.md           # This file
```

## Model Architecture

### Decoder-Only Transformer

The model implements a decoder-only transformer architecture with the following key features:

- **Embedding Layer**: Token embeddings with positional encoding
- **Multi-Head Self-Attention**: 6 attention heads for learning contextual relationships
- **Feed-Forward Networks**: Position-wise fully connected layers with GELU activation
- **Layer Normalization**: Pre-normalization for better convergence
- **Causal Masking**: Ensures autoregressive generation (left-to-right)

### Model Hyperparameters

```
Vocabulary Size:     5,000 tokens
Embedding Dimension: 384 (d_model)
Number of Heads:     6
Number of Layers:    4
Feed-Forward Size:   1,536 (4 × d_model)
Max Sequence Length: 512 tokens
Dropout Rate:        0.1
```

## Key Components

### 1. **SimpleTokenizer** (`class SimpleTokenizer`)

Custom tokenizer with vocabulary management:

- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- Word-level tokenization
- Vocabulary building from most frequent words
- Encoding/decoding functionality

### 2. **WikiDataset** (`class WikiDataset`)

Dataset class for processing Wikipedia text:

- Handles text tokenization and sequence creation
- Creates overlapping sequences for better coverage
- Automatic padding and sequence length management
- Input-target pair generation (shifted by 1 for language modeling)

### 3. **PositionalEncoding** (`class PositionalEncoding`)

Sinusoidal positional encoding:

- Adds positional information to token embeddings
- Uses sine/cosine functions of different frequencies
- Allows model to understand token positions in sequences

### 4. **MultiHeadAttention** (`class MultiHeadAttention`)

Core attention mechanism:

- 6 parallel attention heads
- Scaled dot-product attention
- Causal masking for autoregressive modeling
- Query, Key, Value projections with output projection

### 5. **FeedForward** (`class FeedForward`)

Position-wise feed-forward network:

- Two linear transformations with GELU activation
- Expansion to 1,536 dimensions (4× model dimension)
- Dropout for regularization

### 6. **DecoderBlock** (`class DecoderBlock`)

Single transformer decoder layer:

- Multi-head self-attention sublayer
- Feed-forward sublayer
- Pre-layer normalization (better than post-norm)
- Residual connections around each sublayer

### 7. **DecoderTransformer** (`class DecoderTransformer`)

Complete model implementation:

- Stacks 4 decoder blocks
- Xavier uniform weight initialization
- Final layer normalization and output projection
- Forward pass with autoregressive prediction

## Implementation Details

### Training Features

- **Optimizer**: AdamW with weight decay (0.01) and beta values (0.9, 0.98)
- **Learning Rate**: Base LR of 2e-3 with gradual warmup and decay
- **Warmup Scheduler**: 1 epoch warmup followed by linear decay
- **Loss Function**: Cross-entropy with label smoothing (0.1) and padding ignore
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Early Stopping**: Patience of 3 epochs based on validation loss
- **Batch Size**: 64 for better gradient estimates

### Text Generation

Implements top-k sampling for diverse text generation:

- **Temperature Scaling**: Controls randomness (default: 0.9)
- **Top-K Sampling**: Samples from top 50 most likely tokens
- **Causal Generation**: Uses only previous tokens (autoregressive)
- **Context Window**: Uses last 127 tokens as context

## Training Configuration

```python
DEVICE = "cuda" if available else "cpu"
VOCAB_SIZE = 5000
D_MODEL = 384
N_HEADS = 6
N_LAYERS = 4
D_FF = 1536
BATCH_SIZE = 64
EPOCHS = 15
SEQ_LENGTH = 128
BASE_LR = 2e-3
```

## Usage

### 1. Mount Google Drive (for Colab)

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

### 2. Train the Model

```python
# The notebook automatically:
# - Loads Wikipedia text data
# - Builds vocabulary from text
# - Creates train/validation split (80/20)
# - Trains model with warmup and early stopping
# - Saves best model checkpoint
```

### 3. Generate Text

```python
# Example prompts for text generation
prompts = ["the quick brown", "artificial intelligence", "deep learning"]
for prompt in prompts:
    generated = generate(model, tokenizer, prompt, device=DEVICE)
    print(f"Prompt: '{prompt}' -> '{generated}'")
```

## Requirements

### Python Libraries

```
torch >= 1.9.0
numpy
collections (built-in)
os (built-in)
math (built-in)
```

### Hardware

- **Recommended**: GPU (CUDA-enabled) for faster training
- **Minimum**: CPU (training will be slower)

### Data

- Wikipedia sentence data located at: `/content/gdrive/MyDrive/practical_data/assignment4`
- Text files (`.txt` format)
- Up to 10,000 sentences per file

## Results

The trained model can:

- Generate coherent text continuations given a prompt
- Learn contextual relationships between words
- Produce diverse outputs using top-k sampling
- Demonstrate understanding of language patterns

### Model Size

- Total parameters: ~2-3 million (exact count depends on vocabulary)
- Compact enough for quick training while maintaining performance

### Performance Metrics

- Training loss decreases over epochs
- Validation loss used for early stopping
- Generated text quality improves with training

## Key Learning Objectives

This assignment demonstrates:

1. **Transformer Architecture**: Understanding decoder-only transformers
2. **Attention Mechanisms**: Multi-head self-attention implementation
3. **Positional Encoding**: Adding positional information to embeddings
4. **Language Modeling**: Autoregressive text generation
5. **Training Optimization**: Learning rate scheduling, gradient clipping, early stopping
6. **Text Generation**: Top-k sampling and temperature scaling

## Notes

- Model uses pre-normalization (LayerNorm before attention/FFN) for better convergence
- Causal masking ensures the model only attends to previous tokens
- Label smoothing (0.1) helps prevent overconfidence in predictions
- Xavier initialization aids in stable training from the start
- Overlapping sequences maximize data utilization

---

**Course**: DAM202 (Data Analytics and Mining)  
**Semester**: 5  
**Academic Year**: 2025  
**Assignment**: 4 - Transformer Language Model