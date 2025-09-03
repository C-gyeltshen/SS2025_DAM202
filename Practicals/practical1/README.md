# Text Preprocessing and NLP Pipeline: Practical 1

# Text Preprocessing and NLP Pipeline: Practical 1

This repository contains a Jupyter Notebook that demonstrates essential text preprocessing and Natural Language Processing (NLP) techniques using Python. The notebook is designed for hands-on learning and practical application in data analysis and machine learning workflows.

## Contents

- **Section 0: Creating Data Sets**  
  Learn how to create and structure sample text data for NLP tasks.
- **Section 1: Preprocessing**  
  Covers normalization, text cleaning, tokenization, lemmatization, and stop word removal using spaCy.
- **Section 2: Creating Reusable Functions**  
  Shows how to modularize preprocessing steps for reusability.
- **Section 3: Complete NLP Pipeline**  
  Combines all preprocessing steps into a streamlined workflow.
- **Section 4: Word Representation (Vectorization)**  
  Introduces Count Vectorization (Bag of Words) and advanced vectorization techniques.
- **Section 5: TF-IDF (Term Frequency-Inverse Document Frequency)**  
  Explains and implements TF-IDF for feature extraction, including n-gram analysis.

## Key Features

- Step-by-step code and theory explanations
- Use of pandas, spaCy, and scikit-learn
- Practical examples for each preprocessing step
- Modular and reusable code for real-world projects

## Getting Started

1. **Clone the repository** and open `practical1.ipynb` in Jupyter Notebook or VS Code.
2. **Install required packages** (if not already installed):
   - pandas
   - spacy
   - scikit-learn
   - Download spaCy English model: `python -m spacy download en_core_web_sm`
3. **Run the notebook cells** sequentially to follow the workflow and experiment with the code.

## Usage

This notebook is ideal for students and practitioners who want to:

- Understand and apply text preprocessing techniques
- Build NLP pipelines for machine learning
- Learn best practices for modular and maintainable code

## Step-by-Step Outcomes and Examples

Below are the main steps with before-and-after examples and expected outcomes:

### 1. Creating Data Sets

**Before:**

```python
    data = [

        "When life gives you lemons, make lemonade! 🙂",

        "She bought 2 lemons for $1 at Maven Market.",

        "A dozen lemons will make a gallon of lemonade. [AllRecipes]",

        "lemon, lemon, lemons, lemon, lemon, lemons",

        "He's running to the market to get a lemon — there's a great sale today.",

        "Does Maven Market carry Eureka lemons or Meyer lemons?",

        "An Arnold Palmer is half lemonade, half iced tea. [Wikipedia]",

        "iced tea is my favorite"

    ]
```

**After:**
Data is loaded into a pandas DataFrame for further processing.

---

### 2. Normalization (Lowercasing)

**Before:**

```
"When life gives you lemons, make lemonade! 🙂"
```

**After:**

```
"when life gives you lemons, make lemonade! 🙂"
```

---

### 3. Text Cleaning

**Before:**

```
"A dozen lemons will make a gallon of lemonade. [AllRecipes]"
```

**After:**

```
"a dozen lemons will make a gallon of lemonade"
```

(Removes citations, special characters, and extra spaces)

---

### 4. Tokenization

**Before:**

```
"when life gives you lemons make lemonade"
```

**After:**

```
['when', 'life', 'gives', 'you', 'lemons', 'make', 'lemonade']
```

---

### 5. Lemmatization

**Before:**

```
['when', 'life', 'gives', 'you', 'lemons', 'make', 'lemonade']
```

**After:**

```
['when', 'life', 'give', 'you', 'lemon', 'make', 'lemonade']
```

---

### 6. Stop Words Removal

**Before:**

```
['when', 'life', 'give', 'you', 'lemon', 'make', 'lemonade']
```

**After:**

```
['life', 'give', 'lemon', 'lemonade']
```

---

### 7. Complete NLP Pipeline

**Before:**

```
"When life gives you lemons, make lemonade! 🙂"
```

**After:**

```
"life give lemon lemonade"
```

---

### 8. Count Vectorization (Bag of Words)

**Before:**
Raw text data
**After:**
| life | give | lemon | lemonade |
|------|------|-------|----------|
| 1 | 1 | 1 | 1 |
(Each row is a document, each column is a word, values are word counts)

---

### 9. TF-IDF Vectorization

**Before:**
Raw text data
**After:**
| life | give | lemon | lemonade |
|------|------|-------|----------|
|0.45 |0.45 |0.45 | 0.45 |
(Values are TF-IDF scores, showing importance of each word in each document)

---

### 10. N-gram Analysis

**Before:**
Unigrams only (single words)
**After:**
Unigrams and bigrams (e.g., "life give", "give lemon") with their TF-IDF scores

---

## License

This project is for educational purposes.

## Objectives

1. Understand the importance of text preprocessing in NLP.
2. Learn various text preprocessing techniques.
3. Build a simple NLP pipeline using Python.
4. Apply the learned techniques on a sample dataset.

## Section 0
