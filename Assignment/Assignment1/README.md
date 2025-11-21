# Word Embedding Analysis for Job Description Classification and Clustering

[Repository Link](https://github.com/C-gyeltshen/DAM-Assignment1.git)

## Abstract

This project presents a comprehensive analysis of word embedding techniques applied to job description data, focusing on the implementation and evaluation of Word2Vec models for semantic understanding of employment-related textual content. The study employs natural language processing (NLP) techniques to preprocess, analyze, and cluster job descriptions using both traditional TF-IDF vectorization and modern word embedding approaches. Through systematic evaluation using clustering metrics and visualization techniques, this research demonstrates the effectiveness of word embeddings in capturing semantic relationships within job description datasets.

## Table of Contents

- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Literature Review](#2-literature-review)
- [3. Methodology](#3-methodology)
- [4. Dataset Description](#4-dataset-description)
- [5. Implementation](#5-implementation)
- [6. Evaluation Framework](#6-evaluation-framework)
- [7. Requirements and Installation](#7-requirements-and-installation)
- [8. Usage Instructions](#8-usage-instructions)
- [9. Results and Discussion](#9-results-and-discussion)
- [10. Conclusion](#10-conclusion)
- [11. References](#11-references)
- [12. Acknowledgments](#12-acknowledgments)

## 1. Introduction

### 1.1 Problem Statement

The exponential growth of digital recruitment platforms has generated vast amounts of unstructured textual data in the form of job descriptions. Traditional keyword-based matching systems often fail to capture the semantic relationships between different job requirements, skills, and responsibilities. This project addresses the challenge of developing an intelligent system for job description analysis using advanced word embedding techniques.

### 1.2 Objectives

The primary objectives of this research are:

1. **Preprocessing and Cleaning**: Implement comprehensive text preprocessing pipelines for job description data
2. **Word Embedding Implementation**: Deploy Word2Vec models to generate semantic representations of job-related terminology
3. **Comparative Analysis**: Evaluate the performance of word embeddings against traditional TF-IDF approaches
4. **Clustering and Classification**: Apply unsupervised learning techniques to identify patterns in job categories
5. **Visualization and Interpretation**: Develop methods for visualizing high-dimensional embedding spaces

### 1.3 Significance

This work contributes to the field of computational linguistics and human resource technology by:

- Demonstrating practical applications of word embeddings in recruitment analytics
- Providing a replicable framework for job description analysis
- Offering insights into the semantic structure of professional terminology

## 2. Literature Review

Word embeddings have revolutionized natural language processing by providing dense vector representations that capture semantic relationships between words (Mikolov et al., 2013). The Word2Vec model, introduced by Mikolov and colleagues, employs either Continuous Bag of Words (CBOW) or Skip-gram architectures to learn word representations from large text corpora.

In the domain of job description analysis, traditional approaches have relied heavily on keyword matching and rule-based systems (Furnham, 2008). However, recent advances in NLP have enabled more sophisticated approaches to understanding job requirements and candidate matching (Qin et al., 2018).

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

The preprocessing methodology follows established best practices in text mining:

1. **Text Normalization**: Conversion to lowercase and removal of special characters
2. **Tokenization**: Segmentation of text into individual tokens using NLTK
3. **Stopword Removal**: Elimination of common English stopwords
4. **Feature Engineering**: Creation of combined text representations

### 3.2 Word Embedding Implementation

#### 3.2.1 Word2Vec Configuration

- **Architecture**: Skip-gram model for better performance on infrequent words
- **Vector Dimensions**: 100-dimensional embeddings for optimal performance-complexity trade-off
- **Window Size**: Context window of 5 words
- **Minimum Word Count**: Threshold of 2 occurrences to filter rare terms
- **Training Epochs**: 100 iterations for convergence

#### 3.2.2 TF-IDF Baseline

- **Vectorization**: Term Frequency-Inverse Document Frequency with 1000 features
- **Preprocessing**: Built-in English stopword removal
- **Normalization**: L2 normalization for cosine similarity computation

### 3.3 Clustering Methodology

**K-Means Algorithm**: Unsupervised clustering with the following specifications:

- **Cluster Count**: Empirically determined optimal number of clusters
- **Initialization**: K-means++ for improved convergence
- **Distance Metric**: Euclidean distance for geometric interpretation
- **Random State**: Fixed seed (42) for reproducibility

### 3.4 Dimensionality Reduction

**Principal Component Analysis (PCA)**:

- **Components**: 2D projection for visualization
- **Variance Retention**: Analysis of explained variance ratios
- **Visualization**: Scatter plots with cluster color coding

## 4. Dataset Description

### 4.1 Dataset Overview

The job description dataset (`job_dataset.csv`) contains 1,068 job postings across various technology roles, primarily focusing on .NET development positions. The dataset structure includes:

| Column            | Description                                            | Data Type |
| ----------------- | ------------------------------------------------------ | --------- |
| JobID             | Unique identifier for each job posting                 | String    |
| Title             | Job position title                                     | String    |
| ExperienceLevel   | Required experience level (Fresher, Experienced, etc.) | String    |
| YearsOfExperience | Years of experience required                           | String    |
| Skills            | Required technical skills and competencies             | Text      |
| Responsibilities  | Job duties and responsibilities                        | Text      |
| Keywords          | Relevant keywords for the position                     | Text      |

### 4.2 Data Characteristics

- **Total Records**: 1,068 job descriptions
- **Missing Values**: One missing value in the Title column
- **Text Features**: Three primary text columns (Skills, Responsibilities, Keywords)
- **Domain Focus**: Technology sector with emphasis on software development roles

## 5. Implementation

### 5.1 Development Environment

The project is implemented in Python 3.x using Jupyter Notebook for interactive data analysis and visualization. The modular code structure enables reproducible research and easy extension.

### 5.2 Core Components

1. **Data Loading and Exploration Module**: Initial dataset analysis and statistical summaries
2. **Text Preprocessing Engine**: Comprehensive text cleaning and normalization
3. **Word Embedding Training**: Word2Vec model implementation and training
4. **Clustering Analysis**: Comparative clustering using multiple algorithms
5. **Visualization Framework**: PCA-based dimensionality reduction and plotting

### 5.3 Key Functions

- `clean_text()`: Text preprocessing and normalization
- `preprocess_text()`: Tokenization and stopword removal
- `get_avg_word_vectors()`: Document-level embedding computation
- Clustering evaluation and comparison utilities

## 6. Evaluation Framework

### 6.1 Clustering Evaluation Metrics

#### 6.1.1 Silhouette Score

The Silhouette coefficient measures the quality of clustering by evaluating:

- **Cohesion**: How close points are to their cluster center
- **Separation**: How far points are from neighboring clusters
- **Range**: [-1, 1] where higher values indicate better clustering
- **Interpretation**: Values > 0.5 suggest reasonable clustering structure

#### 6.1.2 Inertia (Within-Cluster Sum of Squares)

- **Definition**: Sum of squared distances from points to cluster centroids
- **Optimization**: Lower values indicate more compact clusters
- **Use Case**: Complementary metric for cluster quality assessment

### 6.2 Comparative Analysis Framework

The evaluation compares two primary approaches:

1. **TF-IDF Clustering**: Traditional sparse vector representation
2. **Word2Vec Clustering**: Dense semantic embeddings

Performance metrics are computed for both approaches to demonstrate the effectiveness of word embeddings in capturing semantic relationships.

## 7. Requirements and Installation

### 7.1 System Requirements

- **Python**: Version 3.7 or higher
- **Memory**: Minimum 4GB RAM recommended
- **Storage**: 500MB free disk space for dependencies
- **Platform**: Cross-platform (Windows, macOS, Linux)

### 7.2 Dependencies

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn gensim plotly wordcloud
```

### 7.3 Additional NLTK Data

The following NLTK datasets are required:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### 7.4 Virtual Environment Setup

For isolated dependency management:

```bash
# Create virtual environment
python -m venv myVenv

# Activate virtual environment
# On macOS/Linux:
source myVenv/bin/activate
# On Windows:
myVenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 8. Usage Instructions

### 8.1 Quick Start

1. **Clone/Download** the project repository
2. **Navigate** to the project directory
3. **Activate** the virtual environment (if using)
4. **Launch** Jupyter Notebook: `jupyter notebook Assignment1.ipynb`
5. **Run** cells sequentially from top to bottom

### 8.2 Detailed Execution Steps

1. **Environment Setup**: Install required packages and download NLTK data
2. **Data Loading**: Load the job description dataset
3. **Preprocessing**: Execute text cleaning and tokenization
4. **Word2Vec Training**: Train the word embedding model
5. **Clustering Analysis**: Perform comparative clustering evaluation
6. **Visualization**: Generate PCA plots and clustering visualizations

### 8.3 Configuration Options

Users can modify the following parameters:

- Word2Vec vector dimensions (default: 100)
- Number of clusters for K-means (empirically determined)
- TF-IDF feature count (default: 1000)
- Text preprocessing options

## 9. Results and Discussion

### 9.1 Expected Outcomes

The analysis is expected to demonstrate:

1. **Semantic Relationships**: Word2Vec's ability to capture meaningful relationships between job-related terms
2. **Clustering Performance**: Comparative analysis showing potential advantages of semantic embeddings over traditional methods
3. **Visualization Insights**: Clear cluster separation in the reduced-dimension embedding space
4. **Practical Applications**: Framework applicability for real-world recruitment analytics

### 9.2 Performance Metrics

Results are evaluated using:

- **Quantitative Metrics**: Silhouette scores and inertia values for both TF-IDF and Word2Vec approaches
- **Qualitative Analysis**: Visual inspection of cluster coherence and separation
- **Comparative Assessment**: Relative performance between traditional and embedding-based methods

### 9.3 Limitations and Future Work

**Current Limitations**:

- Dataset limited to technology sector jobs
- Binary comparison between only two embedding approaches
- Clustering-based evaluation without ground truth labels

**Future Enhancements**:

- Extension to multiple industry domains
- Implementation of additional embedding models (FastText, BERT)
- Supervised evaluation with labeled job categories
- Real-time deployment considerations

## 10. Conclusion

This project provides a comprehensive framework for applying word embedding techniques to job description analysis. Through systematic preprocessing, model training, and evaluation, the research demonstrates the practical utility of semantic embeddings in understanding employment-related textual data. The comparative analysis framework established here can serve as a foundation for more advanced recruitment analytics systems.

The methodology presented offers several contributions to the field:

- **Technical Implementation**: Replicable pipeline for job description analysis
- **Evaluation Framework**: Systematic comparison of embedding approaches
- **Practical Applications**: Direct relevance to human resource technology

## 11. References

Furnham, A. (2008). _HR competencies: Personality, cognitive ability and emotional intelligence_. Springer.

Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In _Advances in neural information processing systems_ (pp. 3111-3119).

Qin, C., Zhu, H., Xu, T., Zhu, C., Jiang, L., Chen, E., & Xiong, H. (2018). Enhancing person-job fit for talent recruitment: An ability-aware neural network approach. In _The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval_ (pp. 25-34).

Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In _Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks_.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. _Journal of machine learning research_, 12(Oct), 2825-2830.

## 12. Acknowledgments

This project utilizes several open-source libraries and frameworks:

- **Pandas & NumPy**: Data manipulation and numerical computing (McKinney, 2010; Harris et al., 2020)
- **Scikit-learn**: Machine learning algorithms and evaluation metrics (Pedregosa et al., 2011)
- **Gensim**: Word embedding model implementation (Rehurek & Sojka, 2010)
- **NLTK**: Natural language processing toolkit (Loper & Bird, 2002)
- **Matplotlib & Seaborn**: Data visualization libraries (Hunter, 2007; Waskom, 2021)

Special acknowledgment to the contributors of the job description dataset and the open-source community for providing the foundational tools that make this research possible.

---

## Project Structure

```
Assignment1/
├── Assignment1.ipynb          # Main Jupyter notebook with analysis
├── job_dataset.csv           # Job description dataset
├── README.md                 # This documentation file
├── myVenv/                   # Python virtual environment
└── requirements.txt          # Python dependencies (if available)
```

## Contact Information

For questions, suggestions, or collaboration opportunities related to this project, please refer to the course materials or contact through appropriate academic channels.

---

_This project was developed as part of DAM202 coursework, demonstrating practical applications of natural language processing and machine learning techniques in the domain of human resource analytics._