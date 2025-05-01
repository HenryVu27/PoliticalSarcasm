
## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download NLTK data (if running locally):**
    Run the following in a Python interpreter within your environment:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    ```
5.  **Data:** Ensure the `train-balanced-sarcasm.csv` file is placed in the `data/` directory (or download it from the source if not included).
6.  **Google Colab:** If using Colab, the notebooks include setup cells to mount Google Drive. **You MUST update the `base_dir` variable in those cells to point to the correct path in *your* Google Drive where the data is stored.**

## Usage

The project consists of two main Jupyter notebooks:

1.  **`SarcasmNLP_Classical.ipynb`:**
    *   Explores classical NLP techniques.
    *   Covers data loading, filtering, EDA, preprocessing, BoW/TF-IDF feature extraction, Logistic Regression, feature engineering (contrast, sentiment, etc.), LDA topic modeling, subreddit analysis, and model comparison.
    *   Run cells sequentially to reproduce the analysis and results.
2.  **`SarcasmNLP_Transformers.ipynb`:**
    *   Implements a sarcasm detection model using a pre-trained Transformer (e.g., DistilBERT).
    *   Includes data preparation for Transformers (minimal preprocessing, combining parent comment), tokenization, fine-tuning using the Hugging Face `Trainer` API, evaluation, and inference examples.
    *   Requires installing `transformers`, `datasets`, and `evaluate`. A GPU is recommended for reasonable training times.
    *   Run cells sequentially. Update `base_dir` if using Colab.

## Methodology

### Classical NLP Approach (`SarcasmNLP_Classical.ipynb`)

*   **Preprocessing:** Lowercasing, removal of special characters/digits, tokenization, stopword removal.
*   **EDA:** Class distribution, comment length analysis, word clouds for sarcastic/non-sarcastic comments.
*   **Feature Extraction:**
    *   Bag-of-Words (BoW)
    *   Term Frequency-Inverse Document Frequency (TF-IDF)
    *   TF-IDF with N-grams (unigrams and bigrams)
*   **Feature Engineering:**
    *   `contrast_score`: Heuristic based on contrastive words and patterns.
    *   `sentiment_shift`: Max change in VADER sentiment score between sentences.
    *   Punctuation/Capitalization counts.
    *   POS Tag counts (e.g., interjections).
*   **Modeling:**
    *   Logistic Regression (baseline and tuned)
    *   (Optional, if added) Multinomial Naive Bayes, Support Vector Machines (SVM)
    *   Combined models using TF-IDF + engineered features.
*   **Analysis:**
    *   LDA Topic Modeling to compare themes.
    *   Analysis of sarcasm rates per subreddit.
    *   Error analysis on misclassified examples.
    *   Feature importance analysis for Logistic Regression.

### Transformer-Based Approach (`SarcasmNLP_Transformers.ipynb`)

*   **Model:** Fine-tuned a pre-trained Transformer model (e.g., `distilbert-base-uncased`).
*   **Context:** Incorporated the `parent_comment` by concatenating it with the `comment` using the `[SEP]` token.
*   **Preprocessing:** Minimal (lowercasing, stripping whitespace).
*   **Tokenization:** Used Hugging Face `AutoTokenizer` with padding and truncation.
*   **Training:** Employed the Hugging Face `Trainer` API for efficient fine-tuning with evaluation and best model saving.
*   **Evaluation:** Assessed using accuracy, F1-score, classification report, confusion matrix, and precision-recall curve on a held-out test set.

## Key Findings & Results

*(Summarize your main results here. Be quantitative where possible.)*

*   The baseline TF-IDF with N-grams and Logistic Regression achieved an accuracy of approximately **XX.X%** and a weighted F1-score of **X.XXX**.
*   Adding engineered features (contrast, sentiment, etc.) to the classical model provided a **[small/moderate/negligible]** improvement, reaching **XX.X%** accuracy / **X.XXX** F1-score, suggesting these cues offer some signal but aren't sufficient alone.
*   The fine-tuned Transformer model (DistilBERT with parent comment context) significantly outperformed the classical methods, achieving an accuracy of **YY.Y%** and a weighted F1-score of **Y.YYY** on the test set.
*   Incorporating the parent comment context proved crucial for the Transformer model's success. *(Quantify if you tested with/without)*
*   LDA revealed distinct topics associated with sarcastic vs. non-sarcastic comments. For example, Topic Z (keywords: ...) was more prevalent in sarcastic comments, while Topic Y (keywords: ...) was more common in non-sarcastic ones. *(Refer to your LDA results table/plots)*.
*   Subreddit analysis confirmed varying baseline sarcasm rates across different political communities. *(Mention top/bottom subreddits if interesting)*.
*   Error analysis showed that both model types struggled most with highly subtle sarcasm requiring significant real-world knowledge or deep conversational context.

## Challenges & Solutions

*   **Context Dependency:** Sarcasm often requires understanding the preceding conversation.
    *   *Solution Attempted:* Incorporated `parent_comment` in the Transformer model.
*   **Subtlety:** Many sarcastic comments lack clear lexical markers.
    *   *Solution Attempted:* Engineered features like `contrast_score` and `sentiment_shift` for classical models. Transformers inherently capture some semantic nuance.
*   **Subjectivity:** Sarcasm perception varies between individuals.
    *   *Mitigation:* Used a large, pre-labeled dataset. *[Add findings from user testing if performed]*.
*   **Computational Cost:** Training large Transformer models can be resource-intensive.
    *   *Mitigation:* Used DistilBERT for a balance of performance and efficiency. Utilized Google Colab GPU.

## Future Work

*   Fine-tune larger Transformer models (BERT-large, RoBERTa) for potentially higher accuracy.
*   Incorporate context beyond just the immediate parent comment (e.g., thread title, earlier comments if feasible).
*   Develop more sophisticated features capturing rhetorical devices or pragmatic context.
*   Explore ensemble methods combining classical and Transformer model predictions.
*   Conduct more extensive error analysis to guide model improvements.
*   Evaluate model performance across different political ideologies or non-political domains.
*   Deploy the best model as an interactive web application (e.g., using Streamlit/Gradio).

## Requirements

*(List the main libraries needed. Create a `requirements.txt` file with specific versions if possible)*
