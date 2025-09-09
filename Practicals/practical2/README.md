# Practical 2: Word2Vec Implementation and Text Processing

This practical demonstrates comprehensive text preprocessing and Word2Vec model training using Python. The notebook covers data preparation, quality assessment, advanced text preprocessing, and Word2Vec implementation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Features](#features)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- macOS 10.15 or higher
- Xcode Command Line Tools
- Virtual environment (recommended)

## Installation

### 1. Basic Setup

```bash
# Clone or navigate to the project directory
cd /Users/chimigyeltshen/Desktop/Sem5/DAM202/practical/Practicals/practical2

# Create a virtual environment (recommended)
python3 -m venv word2vec_env
source word2vec_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

#### Method 1: Using Pre-compiled Wheels (Recommended for macOS)

```bash
# Install with pre-compiled binaries to avoid compilation issues
pip install --only-binary=all gensim
pip install pandas nltk numpy
```

#### Method 2: Using Conda (Alternative)

```bash
# If you have conda installed
conda install -c conda-forge gensim pandas nltk numpy
```

#### Method 3: Step-by-step Installation

```bash
# Install dependencies one by one
pip install numpy
pip install scipy
pip install gensim==4.2.0  # Use stable version
pip install pandas nltk
```

## Common Issues and Solutions

### Issue 1: Gensim Installation Failure on macOS

**Problem:**

```
error: subprocess-exited-with-error
× Preparing metadata (pyproject.toml) did not run successfully.
```

**Solution 1: Install Xcode Command Line Tools**

```bash
xcode-select --install
```

**Solution 2: Use Pre-compiled Wheels**

```bash
pip install --only-binary=all gensim
```

**Solution 3: Use Conda**

```bash
conda install -c conda-forge gensim
```

**Solution 4: Install Older Stable Version**

```bash
pip install gensim==4.2.0
```

### Issue 2: Scipy Compilation Errors

**Problem:** Scipy fails to compile from source on macOS

**Solution:**

```bash
# Install scipy using pre-compiled wheels
pip install --only-binary=scipy scipy

# Or use conda
conda install scipy
```

### Issue 3: Virtual Environment Issues

**Problem:** Package conflicts or permission errors

**Solution:**

```bash
# Create a fresh virtual environment
python3 -m venv fresh_env
source fresh_env/bin/activate
pip install --upgrade pip
pip install --only-binary=all gensim pandas nltk numpy
```

### Issue 4: M1/M2 Mac Specific Issues

**Problem:** Architecture compatibility issues on Apple Silicon

**Solution:**

```bash
# Use conda-forge for better Apple Silicon support
conda install -c conda-forge gensim pandas nltk numpy

# Or ensure you're using the correct architecture
arch -arm64 pip install gensim
```

## Project Structure

```
practical2/
├── practical2.ipynb          # Main notebook with Word2Vec implementation
├── text.txt                  # Sample text data for training
├── README.md                 # This file
└── models/                   # Directory for saved models (created during runtime)
```

## Usage

### 1. Start Jupyter Notebook

```bash
# Activate virtual environment
source word2vec_env/bin/activate

# Start Jupyter
jupyter notebook practical2.ipynb
```

### 2. Run the Notebook

The notebook is organized into the following sections:

1. **Data Preparation**

   - Load and inspect text data
   - Basic data quality assessment

2. **Text Preprocessing Pipeline**

   - Advanced text cleaning
   - Tokenization and normalization
   - Customizable preprocessing options

3. **Parameter Selection**

   - Automatic parameter recommendation
   - Corpus analysis for optimal settings

4. **Word2Vec Training**
   - Model training with progress monitoring
   - Model evaluation and validation

### 3. Key Functions

#### Text Preprocessing

```python
preprocessor = AdvancedTextPreprocessor(
    lowercase=True,
    remove_punctuation=True,
    remove_numbers=True,
    remove_stopwords=False,  # Keep for Word2Vec
    lemmatize=False,
    keep_sentences=True
)
```

#### Parameter Recommendation

```python
params = recommend_parameters(
    corpus_size=len(processed_sentences),
    vocab_size=vocab_size,
    domain_type='general',
    computing_resources='moderate'
)
```

#### Model Training

```python
model = train_word2vec_model(
    processed_sentences,
    save_path='models/word2vec_model.model',
    **params
)
```

## Features

- **Comprehensive Text Preprocessing**: URL removal, email cleaning, punctuation handling
- **Quality Assessment**: Vocabulary analysis, sentence length statistics, word frequency analysis
- **Intelligent Parameter Selection**: Automatic parameter recommendation based on corpus characteristics
- **Progress Monitoring**: Real-time training progress with epoch logging
- **Model Persistence**: Save and load trained models
- **Flexible Configuration**: Customizable preprocessing and training parameters

## Troubleshooting

### Environment Issues

1. **Check Python Version**

   ```bash
   python --version  # Should be 3.8+
   ```

2. **Verify Virtual Environment**

   ```bash
   which python  # Should point to your virtual environment
   ```

3. **Clean Installation**
   ```bash
   pip cache purge
   pip install --no-cache-dir --only-binary=all gensim
   ```

### macOS Specific Solutions

1. **Update macOS and Xcode**

   ```bash
   softwareupdate --install --all
   xcode-select --install
   ```

2. **Check Available Disk Space**

   ```bash
   df -h  # Ensure sufficient space for compilation
   ```

3. **Use Rosetta 2 (Intel Macs with M1/M2)**
   ```bash
   arch -x86_64 pip install gensim
   ```

### Performance Optimization

1. **For Large Datasets**

   - Use `workers=multiprocessing.cpu_count()-1`
   - Increase `min_count` to filter rare words
   - Consider using hierarchical softmax (`hs=1`)

2. **For Limited Resources**
   - Reduce `vector_size`
   - Lower `epochs` count
   - Use smaller `window` size

## Alternative Installation Methods

### Using Poetry

```bash
# If you prefer using Poetry
poetry init
poetry add gensim pandas nltk numpy
poetry install
```

### Using Homebrew + pip

```bash
# Install Python via Homebrew
brew install python
pip3 install --only-binary=all gensim pandas nltk numpy
```

## Support

If you continue to experience issues:

1. Check the [Gensim GitHub Issues](https://github.com/RaRe-Technologies/gensim/issues)
2. Verify your macOS version compatibility
3. Consider using Google Colab as an alternative environment
4. Contact course instructors for additional support

## License

This project is for educational purposes as part of the DAM202 course.
