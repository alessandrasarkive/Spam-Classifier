# Spam-Classifier
SMS Spam Classifier using machine learning to detect spam messages. The project includes data preprocessing, TF-IDF vectorization, and model training with Multinomial Naive Bayes and Logistic Regression. It features accuracy evaluation, custom SMS prediction, and model saving for easy reuse. Ideal for NLP beginners.

# SMS Spam Classifier

This project builds a machine learning model to classify SMS messages as **spam** or **ham** (not spam).

## Dataset

The dataset used is the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), containing 5,574 SMS messages labeled as spam or ham.

## Features

- Text preprocessing: cleaning, tokenization, stopword removal, and stemming.
- TF-IDF vectorization with unigrams and bigrams.
- Model training with Multinomial Naive Bayes (baseline) and Logistic Regression (optional).
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
- Custom SMS prediction function to test messages interactively.
- Saving and loading of trained model and vectorizer using `joblib`.

## How to Run

1. Clone the repository.
2. Install required packages:
   ```bash
   pip install -r requirements.txt
Run the main script:

bash
Copy
Edit
python spam_classifier.py
Use the predict_sms() function to classify custom messages.

Results
Achieved around 94% accuracy on the test set.

Low false positive rate (legitimate messages rarely flagged as spam).

Model handles both common and rare spam phrases effectively.
