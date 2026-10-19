# Machine Learning Foundations & Data Science

A practical repository documenting daily progress from Python & Pandas to Machine Learning engineering.

---

## 📅 Roadmap Overview

### Day 1: Python, Pandas & Data Preprocessing
- **Pandas Core:** Series vs. DataFrames, `.loc` vs. `.iloc`, CSV and Excel file I/O.
- **Data Cleaning:** Deduplication, missing value imputation (median vs. mode), dirty string parsing with regex, and outlier capping.
- **Data Manipulation:** Multi-condition filtering, sorting, multi-metric `groupby`, and table joins (`pd.merge`).
- **Exploratory Data Analysis (EDA):** Statistical summaries, correlation matrices, and 4-panel Seaborn visual dashboards.
- **Machine Learning Preprocessing:** Feature selection, One-Hot Encoding (`pd.get_dummies`), train-test splitting (stratified), and feature scaling (`StandardScaler`).
- **Hands-on Capstone:** End-to-end pipeline training a Random Forest Classifier.
- **Reference Handbook:** High-yield PDF study notes and LaTeX documentation in `day1/notes/`.

### Day 2: Machine Learning & Neural Networks
- **Core Concepts:** AI vs. Machine Learning vs. Deep Learning (the 3 concentric circles).
- **Paradigms & Tasks:** Supervised vs. Unsupervised learning; Regression (numbers) vs. Classification (categories).
- **5 Core Algorithms:** Linear Regression, Logistic Regression, Decision Trees, Random Forest, and K-Means Clustering.
- **Evaluation & Diagnostics:** Underfitting vs. Overfitting, Regression metrics (MAE, RMSE, R²), and Classification metrics (Accuracy, Precision, Recall, F1).
- **Neural Networks:** Neurons, activation functions (ReLU, Sigmoid, Softmax), Loss functions, and Gradient Descent downhill optimization.
- **Reference Handbook:** High-yield PDF study notes and LaTeX documentation in `day2/notes/`.

### Day 3: NLP, Encoder-Decoder & Transformers
- **NLP Foundations & Tokenization:** Character vs. Word vs. Subword (BPE) tokenization, token IDs, and vocabulary lookups.
- **Word Embeddings:** Dense continuous vectors, vector arithmetic ($King - Man + Woman \approx Queen$), and cosine similarity.
- **Attention & Self-Attention:** Overcoming the RNN/LSTM bottleneck, the Library Analogy (Query, Key, Value), Scaled Dot-Product math, and Multi-Head Attention.
- **Transformer Architecture & Positional Encodings:** Permutation invariance, Sinusoidal Positional Encoding formulas, Residual Connections, LayerNorm, and Feed-Forward Networks.
- **The Transformer Trinity:** BERT (Encoder-only, bidirectional), GPT (Decoder-only, causal autoregressive), and T5 (Encoder-Decoder, text-to-text).
- **How LLMs Generate Text:** The autoregressive next-token loop, logits to softmax probabilities, temperature scaling, greedy vs. top-k vs. nucleus (top-p) sampling.
- **Hands-on Capstone:** Exploring tokenization quirks and running pretrained inference with DistilGPT-2.
- **Reference Handbook:** High-yield PDF study notes and LaTeX documentation in `day3/notes/`.

---

## 📂 Project Structure

```
pandasbasicslearn/
├── day1/
│   ├── 01_pandas_basics_and_io.py          # Series, DataFrames, I/O
│   ├── 02_data_cleaning.py                 # Missing values, regex, outliers
│   ├── 03_data_manipulation.py             # Filtering, groupby, joins
│   ├── 04_eda_and_visualization.py         # Statistical & visual EDA
│   ├── 05_ml_preprocessing.py              # One-Hot Encoding, scaling, split
│   ├── 06_hands_on_capstone.py             # Full end-to-end pipeline
│   ├── notes/
│   │   ├── Day_1_Complete_Notes.pdf        # PDF Study Handbook
│   │   ├── DAY_1_COMPLETE_NOTES.md         # Markdown Notes
│   │   └── Day_1_Notes.tex                 # LaTeX source
│   ├── messy_employee_data.csv             # Raw messy dataset
│   ├── department_budget.xlsx              # Supplementary Excel data
│   └── eda_dashboard.png                   # Visual EDA Dashboard
├── day2/
│   ├── 01_linear_regression.py             # Linear Regression & Metrics (MAE, RMSE, R²)
│   ├── 02_logistic_regression.py           # Logistic Regression & S-Curve Probabilities
│   ├── 03_decision_trees_and_random_forest.py # Trees vs. Forest Overfitting Demo
│   ├── 04_kmeans_clustering.py             # Unsupervised K-Means Customer Clustering
│   ├── 05_neural_networks_and_relu.py      # Artificial Neurons, ReLU, & Gradient Descent
│   ├── notes/
│   │   ├── Day_2_Complete_Notes.pdf        # PDF Study Handbook
│   │   ├── DAY_2_COMPLETE_NOTES.md         # Markdown Notes
│   │   └── Day_2_Notes.tex                 # LaTeX source
│   ├── linear_regression_fit.png           # Regression plot
│   ├── logistic_regression_scurve.png      # S-curve probability plot
│   ├── kmeans_clusters.png                 # Cluster visualization
│   └── loss_curve.png                      # Gradient Descent loss curve
├── day3/
│   ├── 01_tokenization_and_embeddings.py   # BPE, Embeddings, King-Man+Woman=Queen
│   ├── 02_attention_and_self_attention.py  # Q, K, V, Scaled Dot-Product, Multi-Head
│   ├── 03_transformer_architecture_and_positional_encoding.py # Positional Waves, FFN, LayerNorm
│   ├── 04_bert_gpt_t5_architectures.py     # BERT vs GPT vs T5, Causal vs Bidir Masks
│   ├── 05_llm_text_generation_and_hands_on.py # Next-token loop, Top-P, DistilGPT-2 inference
│   ├── notes/
│   │   ├── Day_3_Complete_Notes.pdf        # PDF Study Handbook
│   │   ├── DAY_3_COMPLETE_NOTES.md         # Markdown Notes
│   │   └── Day_3_Notes.tex                 # LaTeX source
│   ├── word_embeddings_2d.png              # 2D Embedding space visualization
│   ├── self_attention_heatmap.png          # Coreference self-attention heatmap
│   ├── positional_encoding_matrix.png      # Sinusoidal positional encoding matrix
│   ├── attention_masks_comparison.png      # Bidirectional vs Causal mask comparison
│   └── next_token_probabilities.png        # Top candidate probability distribution
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/shrivatsarpncsystems/<your-repo-name>.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl reportlab torch transformers tokenizers tiktoken
   ```
3. Run any module:
   ```bash
   python day3/05_llm_text_generation_and_hands_on.py
   ```
