[README.md](https://github.com/user-attachments/files/25257051/README.md)
# 🗞️ Fake News Detection using Machine Learning

## Overview
A machine learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques and Logistic Regression. Trained on ~45,000 news articles, the model achieves **99.01% accuracy** on the test set.

## Dataset
- **Source**: Two CSV files — `Fake.csv` (23,481 articles) and `True.csv` (21,417 articles)
- **Total samples**: 44,898 news articles
- **Features used**: Article text (after preprocessing)
- **Target**: Binary label — `0` (Fake) / `1` (Real)

## Tech Stack
| Category | Tools / Libraries |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK (stopwords), Scikit-learn (CountVectorizer, TF-IDF) |
| Modeling | Logistic Regression (Scikit-learn Pipeline) |
| Evaluation | Accuracy Score, Confusion Matrix |

## Preprocessing Pipeline
1. **Label creation** — Assigned `target = 0` for fake and `target = 1` for real news.
2. **Concatenation & shuffle** — Merged both datasets and shuffled rows.
3. **Column removal** — Dropped `title`, `subject`, and `date` to focus on article text only.
4. **Lowercasing** — Converted all text to lowercase.
5. **Punctuation removal** — Stripped all punctuation characters.
6. **Stopword removal** — Removed English stopwords using NLTK.

## Model & Results
- **Vectorization**: CountVectorizer → TF-IDF Transformer (via Scikit-learn Pipeline)
- **Classifier**: Logistic Regression
- **Train/Test split**: 80/20 (random_state = 42)

| Metric | Score |
|---|---|
| **Accuracy** | **99.01%** |

A confusion matrix is also generated to visualize true vs. predicted labels for both classes.

## Project Structure
```
Fake-News-Detection-using-Machine-Learning/
├── Fake.csv              # Fake news dataset
├── True.csv              # Real news dataset
├── main.ipynb            # Full notebook (EDA, preprocessing, modeling, evaluation)
├── requirement.txt       # Python dependencies
└── README.md             # Project documentation (this file)
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Dhruv-Singh-Poniya/Fake-News-Detection-using-Machine-Learning.git
   cd Fake-News-Detection-using-Machine-Learning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirement.txt
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## Key Findings
- The model achieves **99.01% accuracy**, indicating strong separability between fake and real news in this dataset.
- TF-IDF features combined with Logistic Regression provide a simple yet highly effective baseline for text classification.
- Removing stopwords and punctuation significantly cleans the feature space for better model performance.

## Future Improvements
- Add precision, recall, and F1-score to the evaluation for a more complete picture.
- Experiment with other models (e.g., Random Forest, SVM, or deep learning approaches).
- Test on unseen, out-of-distribution news articles to assess real-world generalization.
- Deploy as a simple web app using Streamlit or Flask.
