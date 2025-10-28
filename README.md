# Implicit-Sentiment-Joint-Extraction-Automotive

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

`Implicit-Sentiment-Joint-Extraction-Automotive` is a comprehensive research platform designed for the in-depth analysis of user reviews in the automotive industry. This project systematically compares and evaluates various Natural Language Processing (NLP) techniques, from traditional fine-tuned Transformer models to advanced Large Language Model (LLM) prompting strategies.

The core objective is to tackle complex, domain-specific NLP challenges, including sentiment analysis of implicit opinions, joint extraction of product features and their associated sentiment, and unsupervised discovery of user complaint topics.

## Key Features

*   **LLM-based Sentiment Analysis**: Leverages the in-context learning capabilities of LLMs (e.g., Qwen) for high-accuracy sentiment classification using few-shot prompting, without any model training.
*   **Baseline Comparison**: Provides robust baselines using local Transformer models (e.g., BERT, RoBERTa) for both zero-shot and fine-tuning scenarios.
*   **LLM-powered Data Augmentation**: Utilizes LLMs as a "data factory" to paraphrase and expand the training set, enhancing the performance of smaller, fine-tuned models.
*   **Advanced Aspect-Based Sentiment Analysis (ABSA)**:
    *   **Extractive Method**: An LLM-based approach to simultaneously extract feature terms *directly from the review text* and determine their sentiment.
    *   **Selective Method**: A more constrained approach where the LLM *selects* the most relevant feature from a *pre-defined list* and assigns a sentiment, enabling cleaner, more structured analysis.
*   **Innovative Unsupervised Clustering**: A novel two-stage LLM workflow ("Discover & Classify") that first identifies emergent complaint categories from raw negative reviews and then classifies each review into these dynamically generated topics.
*   **Flexible Evaluation**: Includes custom evaluation scripts that implement complex, rule-based logic for a more nuanced and business-relevant assessment of model performance.

## Project Structure

```
.
├── readme.md                       # Original project description draft
├──.gitclone
├── api.py                     # Method 1: LLM Few-Shot Sentiment Analysis
├── baseline_1.py              # Baseline A: Local Model Zero-Shot Sentiment Analysis
├── baseline_2.py              # Baseline B: Local Model Fine-Tuning Sentiment Analysis
├── feature_extract2.py        # Task A: "Selective" Aspect & Sentiment Extraction (LLM)
├── clustering.py              # Task B: Unsupervised Negative Feedback Clustering (LLM)
├── ensure.py                  # Custom evaluation script for Task A
├── get_dict.py                # Utility to generate feature list from data
├── features_list.json         # Pre-defined list of automotive features for Task B
├── product_issue_leaderboard.csv # Example output from feature_extract2.py
```

## Getting Started

Follow these steps to set up and run the project.

### 1. Prerequisites

Ensure you have Python 3.8 or higher installed.

### 2. Installation & Configuration

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Implicit-Sentiment-Joint-Extraction-Automotive
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn torch transformers datasets openai sentence-transformers tqdm
    ```

3.  **Prepare Data:**
    Place your dataset (e.g., `test.xlsx`) in the project's root directory. The file must contain at least the following columns:
    *   `comment`: The raw user review text.
    *   `polarity`: The sentiment label (e.g., `1` for positive, `-1` for negative, `0` for neutral).
    *   `feature`: The core feature/aspect discussed in the review (can be comma-separated).
    *   `featureNum`: The number of features mentioned in the `feature` column.

4.  **Generate Feature List (for `feature_extract2.py`):**
    If you are using your own dataset, run the `get_dict.py` script to generate the `features_list.json` file. This is crucial for the "selective" aspect extraction task.
    ```bash
    python get_dict.py
    ```

5.  **Configure Scripts:**
    Before running any `.py` script, open the file and modify the configuration variables at the top to match your environment:
    *   `api_key` & `base_url`: Your credentials for the LLM API service.
    *   `data_path` / `DATA_FILE_PATH`: The path to your input data file (e.g., `./test.xlsx`).
    *   `LOCAL_MODEL_PATH`: The local path to your pre-trained Transformer model (e.g., `./bge-large-zh-v1.5/`).

## Modules Explained

This project is structured as a series of experiments and tasks.

### Sentiment Analysis Experiments

These scripts compare different methods for classifying the overall sentiment of a review.

*   **`api.py` (LLM Few-Shot Learning)**: Directly uses an LLM for sentiment classification via a carefully crafted prompt containing a few examples (3 negative, 2 positive). It runs a 10-fold cross-validation to ensure robust evaluation.
*   **`baseline_1.py` (Zero-Shot Baseline)**: Establishes a baseline by evaluating a pre-trained local Transformer model's performance on the dataset *without any fine-tuning*.
*   **`baseline_2.py` (Fine-Tuning Baseline)**: Measures the performance ceiling of traditional methods by fine-tuning a local Transformer model on the training data for each fold of a 10-fold cross-validation.


### Aspect-Based Sentiment Analysis (ABSA)

These scripts tackle the more complex task of identifying *what* is being discussed and *how* the user feels about it.

*   **`feature_extract.py` (Extractive Method)**: The LLM is instructed to extract the key feature (a noun or noun phrase) *directly from the original review text* and determine its sentiment. This method is flexible and can discover new, unlisted features.
*   **`feature_extract2.py` (Selective Method)**: The LLM's task is transformed from generation to classification. It must **choose** the most relevant feature from a pre-defined list (`features_list.json`) and assign a sentiment. This approach yields cleaner, more standardized data ideal for quantitative analysis.

### Unsupervised Topic Clustering

*   **`clustering.py` (Unsupervised Clustering)**: This script demonstrates an innovative two-stage LLM workflow for analyzing large volumes of unstructured feedback (e.g., all negative reviews).
    1.  **Discover**: The LLM is given *all* negative comments at once and tasked with acting as an analyst to summarize and define a set of core problem categories.
    2.  **Classify**: Using the categories generated in the first stage, the LLM then classifies each individual comment, effectively performing unsupervised clustering with human-readable labels.

### Utilities

*   **`ensure.py` (Custom Evaluation)**: A script designed to evaluate the results from `feature_extract.py` using a flexible, rule-based approach that more accurately reflects real-world needs than a simple accuracy score.
*   **`get_dict.py` (Feature List Generator)**: A helper script that parses the `feature` column of your dataset to create the unique, sorted list of features required by `feature_extract2.py`.

## 📊 Example Insights

The "selective" ABSA approach (`feature_extract2.py`) can generate powerful business insights. By aggregating the structured output, we can create a leaderboard of product issues, ranking features by the volume and proportion of negative feedback.

**Example: Product Issue Leaderboard Snippet (`product_issue_leaderboard.csv`)**

| pred_feature | 正面 (Positive) | 负面 (Negative) | 总计 (Total) | 负面比例 (Negative Ratio) |
| :----------- | :-------------- | :-------------- | :----------- | :------------------------ |


This kind of analysis can directly inform product development, quality control, and marketing strategies.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
