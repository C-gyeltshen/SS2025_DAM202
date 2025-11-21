A complete implementation of a Transformer Encoder architecture built from scratch using PyTorch for the TREC Question Classification task.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a Transformer Encoder architecture from scratch for question classification using the TREC dataset. The implementation includes:

- Custom Multi-Head Self-Attention mechanism
- Positional Encoding
- Feed-Forward Networks
- Layer Normalization and Residual Connections
- Complete training and evaluation pipeline
- Attention visualization
- Comprehensive ablation studies

## Features

- **Built from Scratch**: All core components (attention, positional encoding, encoder layers) implemented without using pre-built transformer libraries
- **End-to-End Pipeline**: Data preprocessing, training, evaluation, and visualization
- **Exploratory Data Analysis**: Comprehensive dataset analysis with visualizations
- **Attention Visualization**: Interpretable attention pattern visualizations
- **Ablation Studies**: Systematic analysis of model components
- **Production-Ready Code**: Well-structured, documented, and reproducible

## Project Structure

```
Assignment3/
├── Assignment3.ipynb          # Main Jupyter notebook with complete implementation
├── README.md                  # This file
├── config.json               # Model configuration (generated after training)
├── best_model.pt             # Trained model weights (generated after training)
├── eda_analysis.png          # EDA visualizations (generated after training)
├── training_curves.png       # Training/validation curves (generated after training)
├── confusion_matrix.png      # Test set confusion matrix (generated after training)
├── attention_visualization.png  # Attention patterns (generated after training)
├── ablation_results.csv      # Ablation study results (optional)
└── ablation_study.png        # Ablation comparison plots (optional)
```

## Requirements

### Python Version

- Python 3.8 or higher

### Core Dependencies

```
torch>=1.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.62.0
jupyter>=1.0.0
```

### Optional (for GPU acceleration)

- CUDA toolkit (if using NVIDIA GPU)
- cuDNN

## Installation

### Option 1: Using pip (Local Environment)

1. **Clone or download the repository**

```bash
cd Assignment3
```

2. **Create a virtual environment (recommended)**

```bash
# Using venv
python -m venv transformer_env

# Activate the environment
# On macOS/Linux:
source transformer_env/bin/activate
# On Windows:
transformer_env\Scripts\activate
```

3. **Install required packages**

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn tqdm jupyter
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n transformer_env python=3.9

# Activate environment
conda activate transformer_env

# Install PyTorch (visit https://pytorch.org for specific installation)
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
conda install numpy pandas matplotlib seaborn scikit-learn tqdm jupyter
```

### Option 3: Using Google Colab (Recommended for GPU access)

Google Colab already has most dependencies pre-installed. Simply:

1. Upload `Assignment3.ipynb` to Google Colab
2. Upload your dataset to Google Drive
3. Mount Google Drive in the first cell
4. Run all cells

## Dataset Setup

### Dataset Information

This project uses the **TREC Question Classification Dataset**, which contains questions categorized into 6 coarse-grained classes:

- **ABBR** (Abbreviation): Questions asking for abbreviations
- **DESC** (Description): Questions asking for descriptions
- **ENTY** (Entity): Questions asking about entities
- **HUM** (Human): Questions asking about persons
- **LOC** (Location): Questions asking about locations
- **NUM** (Numeric): Questions asking for numeric information

### Dataset Structure

The dataset should be in CSV format with the following columns:

- `label-coarse`: The coarse-grained question category
- `text`: The question text

### Download Instructions

#### Method 1: Manual Download

1. Download the TREC dataset from one of these sources:

   - [CogComp TREC Dataset](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
   - [Kaggle TREC Dataset](https://www.kaggle.com/datasets)

2. Ensure you have two files:

   - `train.csv` - Training data
   - `test.csv` - Test data

3. Place the files in your preferred location and update the paths in the notebook:

**For Local Environment:**

```python
CONFIG = {
    'train_path': 'path/to/train.csv',
    'test_path': 'path/to/test.csv',
    ...
}
```

**For Google Colab:**

```python
CONFIG = {
    'train_path': '/content/gdrive/MyDrive/your_folder/train.csv',
    'test_path': '/content/gdrive/MyDrive/your_folder/test.csv',
    ...
}
```

#### Method 2: Using Google Drive (for Colab)

1. Upload `train.csv` and `test.csv` to your Google Drive
2. In Colab, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

3. Update the file paths in the CONFIG dictionary to point to your Drive location

### Expected Dataset Format

**Example train.csv / test.csv:**

```csv
label-coarse,text
DESC,How did serfdom develop in and then leave Russia ?
ENTY,What films featured the character Popeye Doyle ?
HUM,Who was Galileo ?
NUM,What is the average speed of the horses at the Kentucky Derby ?
LOC,Where is Kamchatka ?
```

## Usage

### Running the Complete Pipeline

#### Option 1: Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook**

```bash
jupyter notebook
```

2. **Open Assignment3.ipynb**

3. **Update dataset paths** in the configuration cell (find the cell with `CONFIG = {...}`)

4. **Run all cells** sequentially:
   - Click `Cell > Run All` or
   - Use `Shift + Enter` to run each cell

#### Option 2: Google Colab

1. Upload `Assignment3.ipynb` to Google Colab
2. Mount Google Drive (run the first cell)
3. Update dataset paths in the CONFIG section
4. Run all cells: `Runtime > Run all`

### Step-by-Step Execution

The notebook is organized into clear sections:

#### **Part 1: Data Loading and Preprocessing**

- Loads train and test datasets
- Performs exploratory data analysis (EDA)
- Builds vocabulary and encodes text
- Creates PyTorch datasets and data loaders

#### **Part 2: Transformer Encoder Architecture**

- Implements Positional Encoding
- Implements Multi-Head Self-Attention
- Implements Feed-Forward Networks
- Builds complete Encoder Layers and full Transformer

#### **Part 3: Training and Evaluation**

- Trains the model with learning rate scheduling
- Validates performance on validation set
- Evaluates on test set with detailed metrics

#### **Part 4: Attention Visualization**

- Visualizes attention patterns
- Analyzes model interpretability

#### **Part 5: Ablation Study** (Optional)

- Tests different model configurations
- Compares performance across variations

#### **Main Execution**

- Runs the complete pipeline end-to-end

### Running Only Specific Parts

You can run specific sections independently:

**Only Training:**

```python
# After preparing data, create model and train
trainer = Trainer(model, train_loader, val_loader, test_loader, device)
trainer.train()
```

**Only Evaluation:**

```python
# Load saved model and evaluate
model.load_state_dict(torch.load('best_model.pt'))
trainer.detailed_evaluation(label_names)
```

**Only Visualization:**

```python
visualizer = AttentionVisualizer(model, processor, device)
visualizer.visualize_multiple_samples(test_texts, test_labels, predictions)
```

## Model Architecture

### Transformer Encoder Components

1. **Embedding Layer**

   - Converts token indices to dense vectors
   - Dimension: `vocab_size × d_model`

2. **Positional Encoding**

   - Sinusoidal position embeddings
   - Adds positional information to token embeddings

3. **Multi-Head Self-Attention**

   - Number of heads: 8 (configurable)
   - Scaled dot-product attention
   - Parallel attention computations

4. **Feed-Forward Network**

   - Two linear transformations with GELU activation
   - Hidden dimension: 1024 (configurable)

5. **Layer Normalization & Residual Connections**

   - Applied after attention and FFN
   - Stabilizes training

6. **Classification Head**
   - Uses [CLS] token representation
   - Linear layer to output classes

### Default Hyperparameters

```python
d_model = 256          # Model dimension
num_heads = 8          # Number of attention heads
num_layers = 4         # Number of encoder layers
d_ff = 1024           # Feed-forward dimension
max_length = 64        # Maximum sequence length
dropout = 0.1          # Dropout rate
batch_size = 32        # Batch size
learning_rate = 1e-4   # Learning rate
num_epochs = 10        # Training epochs
```

## Training Details

### Optimizer

- **AdamW** optimizer with weight decay (0.01)
- Learning rate: 1e-4

### Learning Rate Scheduling

- **OneCycleLR** scheduler
- Warm-up: 10% of total steps
- Cosine annealing strategy

### Loss Function

- **Cross-Entropy Loss** for multi-class classification

### Regularization

- Dropout (0.1) in attention and FFN layers
- Gradient clipping (max_norm=1.0)
- Weight decay in optimizer

### Training Process

1. Forward pass through encoder
2. Compute cross-entropy loss
3. Backward pass with gradient clipping
4. Update weights with AdamW
5. Step learning rate scheduler
6. Validate after each epoch
7. Save best model based on validation accuracy

## Evaluation Metrics

The model is evaluated using:

1. **Accuracy**: Overall classification accuracy
2. **Precision**: Per-class and macro-averaged
3. **Recall**: Per-class and macro-averaged
4. **F1-Score**: Per-class and macro-averaged
5. **Confusion Matrix**: Detailed error analysis
6. **Support**: Number of samples per class

### Sample Output

```
Test Accuracy: 92.50%
Macro F1-Score: 0.9123

Per-Class Metrics:
Class           Precision    Recall       F1-Score     Support
ABBR           0.9200       0.8900       0.9048       100
DESC           0.9300       0.9100       0.9199       138
ENTY           0.9100       0.9400       0.9248       94
HUM            0.8900       0.9200       0.9048       65
LOC            0.9500       0.9300       0.9399       81
NUM            0.9400       0.9600       0.9499       113
```

## Results

### Expected Performance

With the default configuration, you should achieve:

- **Test Accuracy**: ~90-93%
- **Macro F1-Score**: ~0.90-0.92
- **Training Time**: 5-10 minutes (GPU) / 30-60 minutes (CPU)

### Output Visualizations

1. **EDA Analysis** (`eda_analysis.png`)

   - Class distribution
   - Text length distribution

2. **Training Curves** (`training_curves.png`)

   - Training/validation loss
   - Training/validation accuracy
   - Learning rate schedule

3. **Confusion Matrix** (`confusion_matrix.png`)

   - Detailed classification results
   - Common error patterns

4. **Attention Visualization** (`attention_visualization.png`)
   - Attention patterns for sample questions
   - Model interpretability insights

## Output Files

After running the complete pipeline, the following files will be generated:

| File                          | Description                          | Size      |
| ----------------------------- | ------------------------------------ | --------- |
| `config.json`                 | Model configuration parameters       | ~1 KB     |
| `best_model.pt`               | Trained model weights                | ~10-50 MB |
| `eda_analysis.png`            | Exploratory data analysis plots      | ~200 KB   |
| `training_curves.png`         | Training history visualization       | ~150 KB   |
| `confusion_matrix.png`        | Test set confusion matrix            | ~200 KB   |
| `attention_visualization.png` | Attention pattern visualizations     | ~500 KB   |
| `ablation_results.csv`        | Ablation study results (optional)    | ~2 KB     |
| `ablation_study.png`          | Ablation comparison plots (optional) | ~200 KB   |

## Troubleshooting

### Common Issues and Solutions

#### 1. **CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solution:**

- Reduce batch size in CONFIG: `'batch_size': 16`
- Reduce model dimensions: `'d_model': 128`
- Use CPU: The model will automatically fall back to CPU if GPU is unavailable

#### 2. **File Not Found Error**

```
FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'
```

**Solution:**

- Check dataset paths in CONFIG dictionary
- Ensure files are in the correct location
- Use absolute paths if relative paths don't work

#### 3. **Module Import Errors**

```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**

- Install missing packages: `pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm`
- Check virtual environment is activated

#### 4. **Google Colab Drive Mount Issues**

```
Cannot find the requested files in the Drive
```

**Solution:**

- Re-run the drive mount cell
- Check the file paths in your Google Drive
- Ensure files are uploaded to the correct folder

#### 5. **Slow Training on CPU**

**Solution:**

- Use Google Colab with GPU: `Runtime > Change runtime type > GPU`
- Reduce number of epochs for testing
- Reduce batch size and model dimensions

#### 6. **Memory Issues**

```
MemoryError or system freezes
```

**Solution:**

- Close other applications
- Reduce batch size
- Use gradient accumulation for effective larger batches

### Performance Tips

1. **For faster training:**

   - Use GPU (Google Colab provides free GPU access)
   - Increase batch size if you have enough memory
   - Reduce number of epochs for quick testing

2. **For better accuracy:**

   - Increase model dimensions (`d_model`, `d_ff`)
   - Add more encoder layers
   - Train for more epochs
   - Use data augmentation

3. **For debugging:**
   - Start with small model (2 layers, `d_model=128`)
   - Use subset of data for quick iterations
   - Enable verbose logging in training loop

## Code Structure

### Main Classes

- **TRECDataProcessor**: Handles data loading and preprocessing
- **TRECDataset**: PyTorch Dataset wrapper
- **PositionalEncoding**: Sinusoidal position embeddings
- **MultiHeadAttention**: Multi-head self-attention mechanism
- **FeedForward**: Position-wise feed-forward network
- **EncoderLayer**: Single transformer encoder layer
- **TransformerEncoder**: Complete transformer encoder model
- **Trainer**: Training and evaluation handler
- **AttentionVisualizer**: Attention pattern visualization
- **AblationStudy**: Systematic model analysis

## Additional Resources

- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TREC Question Classification](https://cogcomp.seas.upenn.edu/Data/QA/QC/)

## Notes

- The model is trained from scratch without using pre-trained weights
- All transformer components are implemented manually
- The code is designed for educational purposes and production use
- Reproducibility is ensured with fixed random seeds (seed=42)
- The implementation follows best practices for deep learning projects

## License

This project is created for academic purposes as part of DAM202 coursework.

## Contributing

This is an academic assignment. Please refer to your course guidelines regarding collaboration and code sharing.

---

**Created by:** Chimig Yeltshen  
**Course:** DAM202 - Deep Learning and Advanced Machine Learning  
**Assignment:** Assignment 3 - Transformer Encoder from Scratch  
**Date:** November 2025
