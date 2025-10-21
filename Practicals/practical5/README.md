# English-to-French Neural Machine Translation with Attention Mechanism

## [Repository Link](https://github.com/C-gyeltshen/DAM-english-to-franch.git)

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

---

## Overview

This project implements a **Sequence-to-Sequence (Seq2Seq) Neural Machine Translation model** with **Luong Attention Mechanism** to translate English sentences to French. The model uses a **Bidirectional LSTM Encoder** and an **LSTM Decoder with Attention** to capture contextual information and improve translation quality.

### Key Features:

- **Bidirectional LSTM Encoder** for better context understanding
- **Luong Attention Mechanism** for focused translation
- **Teacher Forcing** during training
- **BLEU Score** evaluation metric
- **Attention Visualization** for interpretability

---

## Dataset

### Source

The dataset is sourced from the **Anki English-French parallel corpus** available at [ManyThings.org](http://www.manythings.org/anki/).

- **Format**: Tab-separated text file (`fra.txt`)
- **Structure**: Each line contains an English sentence and its French translation separated by a tab
- **Size**: 30,000 sentence pairs used for training
- **Split**: 80% training, 20% validation

### Dataset Characteristics:

- **Language Pair**: English (source) → French (target)
- **Domain**: General conversational phrases and sentences
- **Quality**: Human-translated, high-quality translations
- **Variety**: Covers greetings, questions, statements, and common expressions

### Download Process:

The dataset is automatically downloaded and extracted using `urllib` with proper headers to avoid HTTP 406 errors:

```python
download_and_extract_dataset()
```

---

## Data Preprocessing

The preprocessing pipeline transforms raw text into numerical sequences suitable for neural network training.

### 1. **Text Normalization**

#### Unicode to ASCII Conversion

- Converts Unicode characters to ASCII equivalents
- Removes diacritical marks (accents, cedillas, etc.)
- Ensures consistent character representation

```python
unicode_to_ascii(s) → "café" becomes "cafe"
```

#### Text Cleaning Steps:

1. **Lowercase conversion**: All text converted to lowercase
2. **Punctuation spacing**: Adds spaces around punctuation (`.`, `?`, `!`, `,`)
3. **Character filtering**: Removes non-letter characters except allowed punctuation
4. **Whitespace normalization**: Removes extra spaces
5. **Token addition**: Adds `<start>` and `<end>` tokens to mark sentence boundaries

**Example:**

```
Original:  "How are you?"
Processed: "<start> how are you ? <end>"
```

### 2. **Tokenization**

Custom `LanguageTokenizer` class creates vocabulary and converts text to sequences:

#### Vocabulary Creation:

- Extracts all unique words from the dataset
- Sorts vocabulary alphabetically
- Creates bidirectional mappings:
  - `word2idx`: Word → Index
  - `idx2word`: Index → Word
- Reserves index `0` for padding (`<pad>`)

#### Encoding Process:

- Converts sentences to sequences of integer indices
- Pads sequences to maximum length for batch processing
- Ensures uniform input shapes

**Example:**

```
Sentence: "<start> hello <end>"
Encoded:  [145, 2341, 146, 0, 0, ...]  (padded to max_length)
```

### 3. **Dataset Preparation**

- **Input tensors**: English sentences (encoder input)
- **Target tensors**: French sentences (decoder input/output)
- **Maximum lengths**:
  - English: Determined by longest sentence in dataset
  - French: Determined by longest sentence in dataset
- **Vocabulary sizes**:
  - English: ~7,000-8,000 unique words
  - French: ~12,000-15,000 unique words

---

## Model Architecture

The model follows a **Seq2Seq architecture with Attention**, consisting of three main components:

![1](image/1.png)

### 1. **Encoder (Bidirectional LSTM)**

```
Input → Embedding → Bidirectional LSTM → Hidden States
```

#### Components:

- **Embedding Layer**:

  - Dimension: 256
  - Converts word indices to dense vectors
  - Learns semantic relationships between words

- **Bidirectional LSTM**:
  - Units: 512 (256 per direction)
  - Processes input in both forward and backward directions
  - Captures contextual information from both past and future
  - Returns:
    - **Output sequences**: All hidden states for attention
    - **Final states**: Forward and backward (h, c) states

#### State Combination:

- Forward and backward states are **summed** (merge_mode='sum')
- Combined states initialize the decoder
- Provides rich contextual information

**Architecture Diagram:**

```
English Sentence → Embedding(256) → Bi-LSTM(512) → Encoder Outputs + States
```

### 2. **Attention Mechanism (Luong Attention)**

The attention mechanism allows the decoder to focus on relevant parts of the input sequence.

#### Score Calculation:

```
score(h_t, h_s) = h_t^T · h_s
```

Where:

- `h_t`: Current decoder hidden state (query)
- `h_s`: Encoder hidden states (values)

#### Steps:

1. **Score Computation**: Matrix multiplication between decoder state and all encoder states
2. **Normalization**: Apply softmax to get attention weights (probabilities)
3. **Context Vector**: Weighted sum of encoder outputs
   ```
   context = Σ(attention_weights_i × encoder_output_i)
   ```

#### Benefits:

- Focuses on relevant source words for each target word
- Handles long sentences better
- Interpretable (can visualize which words are attended to)

**Formula:**

```
attention_weights = softmax(h_decoder^T × h_encoder)
context_vector = Σ(attention_weights × encoder_outputs)
```

### 3. **Decoder (LSTM with Attention)**

```
Target Input → Embedding → LSTM → Attention → Dense → Output Probability
```

#### Components:

- **Embedding Layer**: Dimension 256 for target language
- **LSTM Layer**:

  - Units: 512
  - Initialized with encoder final states
  - Generates hidden states at each timestep

- **Attention Layer**:

  - Computes context vector from encoder outputs
  - Combines with decoder state

- **Output Layer**:
  - Concatenation layer: Combines context + decoder state
  - Tanh activation layer (Wc): Non-linear transformation
  - Final Dense layer: Projects to vocabulary size
  - Outputs probability distribution over all French words

#### Decoding Process (Training):

1. Take target word at time t
2. Embed and pass through LSTM
3. Calculate attention over encoder outputs
4. Combine context with LSTM output
5. Generate prediction for next word

#### Decoding Process (Inference):

1. Start with `<start>` token
2. Generate one word at a time
3. Feed predicted word as next input (autoregressive)
4. Stop when `<end>` token is generated or max length reached

**Architecture Flow:**

```
Previous Word → Embedding → LSTM → [Concat: LSTM output + Context]
→ Tanh → Dense(vocab_size) → Next Word Probability
```

### Complete Model Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                 │
│  English Input → Embedding(256) → Bi-LSTM(512)                  │
│  Outputs: encoder_states, hidden_states                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ATTENTION                                  │
│  Query: decoder_state, Keys/Values: encoder_outputs             │
│  Output: context_vector, attention_weights                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                 │
│  French Input → Embedding(256) → LSTM(512) → Attention          │
│  → Concat → Tanh → Dense(vocab_size) → French Output            │
└─────────────────────────────────────────────────────────────────┘
```

### Hyperparameters:

- **Embedding Dimension**: 256
- **Encoder Units**: 512 (bidirectional)
- **Decoder Units**: 512
- **Batch Size**: 64
- **Optimizer**: Adam (learning rate: 0.001)
- **Epochs**: 20

---

## Training Process

### 1. **Loss Function**

**Sparse Categorical Cross-Entropy** with custom masking:

```python
loss = -Σ(y_true × log(y_pred))  # Only for non-padding tokens
```

- Computes cross-entropy between predicted and actual French words
- **Masking**: Ignores padding tokens (index 0) in loss calculation
- Ensures model doesn't learn from padded positions

### 2. **Teacher Forcing**

During training, the decoder receives the **actual target word** (not its prediction) as input:

```
Decoder Input:  <start> je   t'   aime
Decoder Output: je      t'   aime <end>
```

**Benefits:**

- Faster convergence
- Prevents error accumulation
- Stable training

### 3. **Optimization**

- **Optimizer**: Adam with learning rate 0.001
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Batch Processing**: 64 samples per batch
- **Steps per Epoch**: Dataset size / batch size

### 4. **Training Loop**

```python
for epoch in range(EPOCHS):
    for batch in dataset:
        1. Forward pass through encoder
        2. Forward pass through decoder with teacher forcing
        3. Calculate loss (ignoring padding)
        4. Backpropagate
        5. Clip gradients
        6. Update weights
```

### 5. **Checkpointing**

- Model weights saved every 5 epochs
- Final model saved at training completion
- Enables resuming training or inference

**Training Time**: ~2-4 hours (depending on hardware)

---

## Evaluation

### 1. **BLEU Score**

**BLEU (Bilingual Evaluation Understudy)** measures translation quality:

#### Formula:

```
BLEU = BP × exp(Σ(w_n × log(p_n)))
```

Where:

- `p_n`: Precision of n-grams (n=1,2,3,4)
- `BP`: Brevity penalty (penalizes short translations)
- `w_n`: Weight for each n-gram (typically uniform: 0.25)

#### N-gram Precision:

- **Unigrams (1-gram)**: Individual word matches
- **Bigrams (2-gram)**: Two consecutive word matches
- **Trigrams (3-gram)**: Three consecutive word matches
- **4-grams**: Four consecutive word matches

#### Interpretation:

- **0-20**: Very poor translation
- **20-40**: Poor, understandable with effort
- **40-60**: Good, generally understandable
- **60-80**: Very good translation
- **80-100**: Excellent (human-level)

### 2. **Qualitative Evaluation**

Sample translations are evaluated manually for:

- **Fluency**: Natural-sounding French
- **Accuracy**: Correct meaning preservation
- **Grammar**: Proper French grammar

### 3. **Attention Visualization**

Heatmaps show which English words the model focuses on when generating each French word:

- **X-axis**: English input words
- **Y-axis**: French output words
- **Color intensity**: Attention weight (darker = more attention)

**Benefits:**

- Verifies model learns alignment
- Debugging translation errors
- Interpretability for users

---

## Usage

### Installation

```bash
pip install tensorflow>=2.8.0
pip install numpy pandas matplotlib scikit-learn
```

### Training the Model

```python
# Run the main training script
main()
```

### Translating New Sentences

```python
# Load trained model
model.load_weights('checkpoints/final_model.weights.h5')

# Translate
sentence = "I love you."
translation, _, attention = translate(sentence, model, en_tokenizer,
                                      fr_tokenizer, max_length_en, max_length_fr)
print(f"Translation: {translation}")
```

### Visualizing Attention

```python
visualize_attention_for_sentence("How are you?", model, en_tokenizer,
                                 fr_tokenizer, max_length_en, max_length_fr)
```

---

## Results

### Expected Performance:

- **Average BLEU Score**: 30-50 (on validation set)
- **Training Loss**: Decreases from ~4.0 to ~1.5-2.0
- **Translation Quality**: Good for common phrases, moderate for complex sentences

### Sample Translations:

| English                    | Reference French           | Model Translation     |
| -------------------------- | -------------------------- | --------------------- |
| "I love you."              | "Je t'aime."               | "Je t'aime."          |
| "How are you?"             | "Comment allez-vous ?"     | "Comment vas-tu ?"    |
| "Good morning."            | "Bonjour."                 | "Bonjour."            |
| "This is a beautiful day." | "C'est une belle journée." | "C'est un beau jour." |

### Limitations:

- **Rare words**: May struggle with uncommon vocabulary
- **Long sentences**: Performance degrades with very long inputs
- **Idioms**: Literal translations of idiomatic expressions
- **Context**: Single-sentence context (no multi-turn dialogue)

### Improvements:

- Use **Transformer architecture** (more recent, better performance)
- Increase **dataset size** (100K+ sentence pairs)
- Apply **data augmentation** (back-translation)
- Use **pre-trained embeddings** (Word2Vec, GloVe)
- Implement **beam search** for better inference

---

## File Structure

```
practical5/
│
├── practical5.ipynb          # Main notebook with implementation
├── README.md                 # This file
├── image/                    # Directory for saving attention plots
├── checkpoints/              # Model checkpoints (created during training)
│   ├── ckpt-5.weights.h5
│   ├── ckpt-10.weights.h5
│   └── final_model.weights.h5
└── fra.txt                   # English-French dataset (downloaded)
```

---

## References

1. **Sutskever et al. (2014)**: "Sequence to Sequence Learning with Neural Networks"
2. **Bahdanau et al. (2015)**: "Neural Machine Translation by Jointly Learning to Align and Translate"
3. **Luong et al. (2015)**: "Effective Approaches to Attention-based Neural Machine Translation"
4. **Papineni et al. (2002)**: "BLEU: a Method for Automatic Evaluation of Machine Translation"

---

## License

This project is for educational purposes as part of the DAM202 course.

---

## Author

**Chimigyeltshen**  
Semester 5, DAM202 - Practical 5
