# 📰 Indian Express News Credibility Detector

An end-to-end Machine Learning web application designed to evaluate news headlines and assess whether they represent authentic reporting or potential disinformation. Built with Scikit-Learn, Pandas, and Streamlit.

---

## 🚀 Live Demo

- **Application URL:** [https://fakenews-detectord.streamlit.app](https://fakenews-detectord.streamlit.app)

---

## 📌 Project Overview

Digital misinformation often exhibits noticeable stylistic, linguistic, and lexical patterns—such as sensational punctuation, abnormal capitalization ratios, and emotionally charged rhetoric. 

This detector analyzes both the **textual semantics** and the **stylistic features** of an input headline using a unified machine learning pipeline to output real-time authenticity probabilities.

---

## ✨ Features

- **Text Normalization:** Strips URLs, HTML artifacts, punctuation, and extraneous whitespace.
- **Engineered Stylistic Signals:**
  - `caps_ratio`: Measures sensationalist uppercase usage.
  - `exclamations` & `questions`: Tracks emotional emphasis and clickbait markers.
  - `char_count` & `word_count`: Captures headline structural patterns.
- **Probabilistic Scoring:** Displays distinct percentages for both authenticity and disinformation markers.
- **Interactive UI:** Clean two-column interface with instant reset options built using Streamlit.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries & Frameworks:** Streamlit, Scikit-Learn, Pandas, NumPy, Joblib, Regex
- **Deployment Platform:** Streamlit Community Cloud
- **Version Control:** Git & GitHub

---

## 📂 Repository Structure

```text
├── app.py                   # Streamlit web application interface
├── fake_news_pipeline.pkl   # Serialized ML pipeline (vectorizer + classifier + scaler)
├── requirements.txt         # Project dependencies for deployment
├── .gitignore               # Excludes raw datasets, virtual environments, and cache
└── README.md                # Project documentation
