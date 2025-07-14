# Twitter Sentiment Analysis on Covid-19 Vaccination using Machine Learning

This project performs sentiment analysis on COVID-19 vaccination-related tweets using various supervised machine learning models. The analysis aims to classify tweets into **positive**, **negative**, or **neutral** sentiments and evaluate the effectiveness of multiple models based on precision, recall, and accuracy.

---

## Project Objective

The goal of this project is to build and compare multiple machine learning classifiers to detect sentiment in tweets related to COVID-19 vaccination. This helps in understanding public opinion and identifying sentiment trends in social media conversations.

---

## Technologies & Libraries Used

- Python 3
- Pandas, NumPy
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Seaborn & Matplotlib
- WordCloud
- Confusion Matrix tools

---

## Methodology

### 1. Data Preprocessing
- Removed stopwords, punctuations, and user mentions
- Performed tokenization and stemming
- Labeled sentiments as: Positive, Negative, Neutral

### 2. Text Vectorization
- TF-IDF Vectorizer used for feature extraction

### 3. Model Training & Evaluation
- Classifiers used:
  - Decision Tree (DT)
  - Random Forest (RF)
  - Logistic Regression (LR)
  - Support Vector Machine (SVM)
  - Naive Bayes (NB)
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

## Key Results

| Model              | Accuracy (%) |
|-------------------|--------------|
| Decision Tree (DT) | **89.81**     |
| Random Forest (RF) | 84.73        |
| Logistic Regression (LR) | 84.64   |
| Support Vector Machine (SVM) | ~83 |
| Naive Bayes (NB)   | ~81          |

- **Best Model:** Decision Tree (DT)
  - Highest accuracy, recall, and precision across all sentiment classes
- **Observations:**
  - SVM and NB struggled with negative sentiments
  - Random Forest had strong precision but lower recall for negative class
  - Logistic Regression performed moderately well across all metrics

---

## Visualization Highlights

### 1. **Word Clouds**
- **Negative Tweets:** Common words: "arm", "little", "death", "side effects"
- **Neutral Tweets:** Words like "dose", "safety", "Pfizer vaccine"
- **Positive Tweets:** Words like "happy", "vaccinated", "Covid-19", "Pfizer-BioNTech"

### 2. **Clustered Column Chart**
- Displays metric-wise performance comparison
- DT excels in both Neutral and Positive sentiments
- NB performs poorly in Negative classification

### 3. **Confusion Matrix**
- Sentiment distribution:
  - Neutral: **48.9%**
  - Positive: **40.9%**
  - Negative: **10.2%**
- Indicates bias in some models toward positive and neutral sentiments

