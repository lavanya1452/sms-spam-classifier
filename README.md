# SMS Spam Classification Engine

A comparative machine learning project that classifies SMS messages into **Ham** (Legitimate) or **Spam** using both classical NLP (TF-IDF + Naive Bayes) and transformer-based Deep Learning (DistilBERT).

---

## 📌 Features

- **Data Preprocessing & Cleaning:** Normalizes text, removes noise, converts digits/URLs, and handles duplicate removal.
- **Classical ML Pipeline:** Scikit-Learn `TfidfVectorizer` paired with `MultinomialNB` for lightweight, ultra-fast inference.
- **Transformer Fine-Tuning:** Fine-tuned `distilbert-base-uncased` model using Hugging Face `Transformers` and `Datasets` with dynamic batch padding via `DataCollatorWithPadding`.
- **Reproducible Workflow:** Includes stratified dataset splitting to maintain class balance across training and testing sets.

---

## 🛠️ Project Structure

```text
├── spam.csv                 # Raw dataset (Category, Message)
├── spam_nlp_pipeline.joblib # Saved Naive Bayes pipeline artifact
├── main.ipynb / script.py   # Model training & evaluation scripts
├── .gitignore               # Ignores large model weights and checkpoints
└── README.md                # Project documentation
