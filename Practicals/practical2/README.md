# Practical 2: Word Embeddings with Word2Vec

This notebook demonstrates the process of training and evaluating a Word2Vec model using Gensim on a text dataset.

## Table of Contents

1.  [Section 1: Connecting Google Drive and Data Loading](#section-1-connecting-google-drive-and-data-loading)
2.  [Section 2: Data Preprocessing](#section-2-data-preprocessing)
3.  [Section 3: Training the Model](#section-3-training-the-model)
4.  [Section 4: Model Evaluation](#section-4-model-evaluation)

## Section 1: Connecting Google Drive and Data Loading

This section focuses on setting up the environment to access the dataset stored in Google Drive and loading it into the notebook.

-   Connecting Google Drive to the Colab environment.
-   Navigating to the directory containing the dataset (`text.txt`).
-   Loading the text data from the file.
-   Performing a basic data quality assessment, including calculating vocabulary size, average sentence length, and word frequencies.

## Section 2: Data Preprocessing

This section covers the essential steps to clean and prepare the text data for training a Word2Vec model.

-   Setting up the Python environment with necessary NLP libraries (nltk, re, string, etc.).
-   Downloading required NLTK data (punkt, stopwords, wordnet, averaged\_perceptron\_tagger).
-   Implementing an `AdvancedTextPreprocessor` class to handle various preprocessing tasks:
    -   Lowercasing the text.
    -   Removing punctuation and numbers.
    -   Removing stopwords (optional, kept for Word2Vec training).
    -   Lemmatization (optional, not used in this example).
    -   Removing URLs and email addresses.
    -   Handling sentence tokenization for the Word2Vec model input.
-   Applying the preprocessing steps to the loaded text data.
-   Cross-checking the preprocessed data by printing sample sentences.
-   Calculating the corpus size (number of sentences) and vocabulary size (unique words) of the preprocessed data.
-   Defining a function `recommend_parameters` to suggest optimal Word2Vec parameters based on corpus characteristics and computing resources.
-   Using the `recommend_parameters` function to get recommended parameters for the current dataset.

## Section 3: Training the Model

This section details the process of training the Word2Vec model using the preprocessed data and recommended parameters.

-   Installing the Gensim library for efficient Word2Vec implementation.
-   Defining an `EpochLogger` callback to monitor the training progress epoch by epoch.
-   Implementing the `train_word2vec_model` function which:
    -   Takes tokenized sentences and Word2Vec parameters as input.
    -   Sets default parameters and updates them with provided values.
    -   Initializes and trains the `gensim.models.Word2Vec` model.
    -   Includes the `EpochLogger` callback for logging.
    -   Saves the trained model to a specified path.
-   Training the Word2Vec model with the preprocessed sentences and recommended parameters.
-   Printing the final vocabulary size of the trained model.
-   Displaying a sample of the words present in the model's vocabulary.

## Section 4: Model Evaluation

This section provides tools and examples for evaluating the performance of the trained Word2Vec model.

-   Defining a `Word2VecEvaluator` class with methods to assess the model's quality:
    -   `evaluate_word_similarity`: Measures the correlation between the model's word similarity scores and human judgments.
    -   `evaluate_analogies`: Tests the model's ability to solve analogy tasks.
    -   `evaluate_odd_one_out`: Evaluates the model's capacity to identify the outlier word in a group. (Note: This method in the code currently only checks if the model *can* identify an odd word, not against a specific ground truth).
    -   `analyze_vocabulary_coverage`: Reports how much of a given test text vocabulary is present in the model's vocabulary.
    -   `compare_with_baseline`: Compares the current model's similarity patterns to a baseline model.
-   Providing example datasets for word similarity pairs and analogy tasks.
-   Using the `Word2VecEvaluator` to perform word similarity and analogy evaluations on the example datasets.
-   Demonstrating how to find the most similar words to a given word using the trained model.
-   Showing how to calculate the similarity between two specific words using the trained model.