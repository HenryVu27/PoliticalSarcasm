# Political Sarcasm Detection on Reddit: NLP vs. Transformers

This repository contains the code and report for a project focused on detecting sarcasm in political comments sourced from Reddit. It compares traditional Natural Language Processing (NLP) techniques, enhanced with extensive feature engineering, against a fine-tuned transformer model (DistilRoBERTa).

## Overview

Sarcasm detection is a challenging NLP task, especially in nuanced domains like political discourse. This project utilizes a dataset of Reddit comments labeled for sarcasm (often via the '/s' marker) and filters it to focus specifically on politically-oriented subreddits.

The core components of this work include:
1.  **Data Curation:** Isolating political comments from a large Reddit dataset.
2.  **Traditional NLP Pipeline:** Implementing Bag-of-Words, TF-IDF, N-grams, hyperparameter tuning, and ensemble methods using scikit-learn, augmented with novel linguistic, sentiment, contextual, and readability features.
3.  **Transformer Fine-tuning:** Fine-tuning a DistilRoBERTa model for sequence classification using PyTorch and the Transformers library.
4.  **Analysis & Interpretability:** Exploring feature importance (NLP), attention mechanisms, token importance (Transformer), error analysis, and LDA topic modeling to understand model behavior and thematic differences.
5.  **Comprehensive Reporting:** A detailed LaTeX report discussing the methodology, results, and insights.

## Key Features

*   Comparative analysis of traditional NLP vs. transformer models for political sarcasm detection.
*   Focus on a specific, challenging domain: political discourse on Reddit.
*   Novel feature engineering suite for traditional NLP models.
*   Fine-tuning and evaluation of DistilRoBERTa.
*   In-depth model interpretability analysis (feature/token importance, attention visualization).
*   Latent Dirichlet Allocation (LDA) topic modeling to uncover themes in sarcastic vs. non-sarcastic comments.
*   Extensive visualizations for data exploration, model performance, and analysis.
## Data Source

This project uses data derived from the **Reddit Sarcasm Dataset**.

*   **Original Dataset:** The original balanced dataset (`train-balanced-sarcasm.csv`) can typically be found on [Kaggle](https://www.kaggle.com/datasets/danofer/sarcasm)
*   **Filtering:** The notebooks operate on a filtered subset of this data, containing only comments from a predefined list of politically relevant subreddits (see `SarcasmNLP.ipynb` for the list and filtering code).

**Important:** Due to its size, the raw or filtered dataset is **not included** in this repository. To run the notebooks, you will need to:
1.  Download the original `train-balanced-sarcasm.csv` file from the source linked above.
2.  Place the file in the expected location (e.g., `/content/drive/MyDrive/SarcasmDetection/` if using the Google Colab setup, or update the paths in the notebooks for local execution).
3.  The filtering logic is present in the `SarcasmNLP.ipynb` notebook and will be executed when run.

## Setup and Installation

This project uses Python 3 and relies on several libraries. It's recommended to use a virtual environment (`venv` or `conda`).

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HenryVu27/PoliticalSarcasm.git
    cd PoliticalSarcasm
    ```

2.  **Install dependencies:**
    You can install the required packages using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud torch torchvision torchaudio transformers plotly jupyterlab ipywidgets google-colab
    ```
    *Note: `google-colab` is primarily for Colab compatibility; remove if running locally.*

3.  **Download NLTK data:**
    Run the following in a Python interpreter after installing `nltk`:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    ```

4.  **GPU Requirements:** Running the `SarcasmLLM.ipynb` notebook (transformer fine-tuning) is computationally intensive and highly recommended to be done on a machine with a CUDA-enabled GPU. The code automatically detects and uses a GPU if available via PyTorch.

## Usage

1.  **Environment:** Ensure your Python environment is set up with the required dependencies.
2.  **Data:** Download the original dataset and place it where the notebooks expect it (update paths if necessary).
3.  **Run Notebooks:** Open and run the Jupyter notebooks (`.ipynb` files) using Jupyter Lab or Jupyter Notebook.
    *   `SarcasmNLP.ipynb`: Executes the traditional NLP pipeline, including data loading, preprocessing, feature engineering, model training/evaluation, and LDA.
    *   `SarcasmLLM.ipynb`: Executes the transformer pipeline, including data loading, model fine-tuning, evaluation, and interpretability analysis.
4.  **Google Colab:** Both notebooks contain setup code for Google Colab (mounting Google Drive). If running locally, you will need to comment out or modify the `google.colab` related cells and adjust file paths accordingly.
5.  **Execution Time:** Be aware that the transformer fine-tuning in `SarcasmLLM.ipynb` can take a significant amount of time, especially without a GPU. Feature extraction in `SarcasmNLP.ipynb` can also be time-consuming on the full dataset.

## Results Summary

The fine-tuned DistilRoBERTa model achieved the best performance, significantly outperforming the traditional NLP approaches. However, the novel feature engineering implemented substantially improved the baseline performance of the traditional models. LDA topic modeling revealed distinct thematic differences between sarcastic and non-sarcastic political comments.

For detailed results, performance metrics, visualizations, and analysis, please refer to the comprehensive report (`report.pdf`).
