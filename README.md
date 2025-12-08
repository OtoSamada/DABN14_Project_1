# Fake News Detection Using Support Vector Machines

A machine learning project that classifies news articles as **Real** or **Fake** using Support Vector Machines (SVMs) with TF-IDF features and engineered linguistic features.

## Project Overview

This project compares three SVM kernel variants (Linear, Polynomial, RBF) for fake news detection, demonstrating that a well-tuned **Linear SVM achieves near-perfect accuracy** while being significantly more computationally efficient than complex kernels.

### Key Results

| Model | Test Accuracy | F1 Score | Training Time |
|-------|---------------|----------|---------------|
| Linear SVM | 99.89% | 0.9990 | ~3s |
| Polynomial SVM | 100.00% | 1.0000 | ~180s |
| RBF SVM | 99.96% | 0.9996 | ~120s |

## Project Structure

```
├── data/                          
├── models/                        
├── modules/                       
├── notebooks/                     
├── Final_Report.ipynb                
└── requirements.txt
```

### Folders

- **data/**: Raw dataset (Fake.csv, True.csv), processed feature matrices, labels, and preprocessing artifacts (vectorizer, scaler)

- **models/**: Trained SVM models saved as .joblib files

- **modules/**: Python utilities for experiment tracking and the Optuna-based training pipeline

- **notebooks/**: Jupyter notebooks for data processing, model training, and analysis

### Notebooks

- **data_preprocess.ipynb**: Cleans text data, removes duplicates and leakage words, engineers features (sentiment, emotion, stylometric), and generates TF-IDF vectors

- **model_experiments.ipynb**: Trains Linear, Polynomial, and RBF SVMs with Optuna hyperparameter optimization and evaluates performance

- **shap.ipynb**: Analyzes feature importance using SHAP values to interpret model predictions

- **Final_Report.ipynb**: Complete project report with methodology, results, and conclusions

## Installation

```bash
pip install -r requirements.txt
```

## Features Used

- **Text**: TF-IDF (50K features, unigrams to trigrams)
- **Sentiment**: Polarity & subjectivity scores
- **Emotion**: 7-class emotion classification
- **Stylometric**: Capital ratio, exclamation count, sentence length
- **Temporal**: Cyclical date encoding (sin/cos)

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- **44,898** articles (52% Fake, 48% Real)
- **39,100** after deduplication

## Authors

- Otari Samadashvili
- Nana Jaoshvili

*Advanced Machine Learning - Project 1 (December 2025)*