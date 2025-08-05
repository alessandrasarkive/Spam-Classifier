import pandas as pd

# Replace this with your actual file path
file_path = r"C:\Users\Asus\Desktop\SMSSpamCollection"

# Load the dataset
df = pd.read_csv(file_path, sep='\t', header=None, names=["label", "message"])

# Preview the data
print(df.head())
print(df['label'].value_counts())

import nltk
nltk.download('stopwords')

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize (split into words)
    words = text.split()
    
    # Remove stopwords and apply stemming
    cleaned = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(cleaned)

# Apply preprocessing to all messages
df['cleaned_message'] = df['message'].apply(preprocess_text)

# Preview cleaned messages
print(df[['message', 'cleaned_message']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Create the vectorizer
vectorizer = TfidfVectorizer()

# Transform the cleaned messages into numerical features
X = vectorizer.fit_transform(df['cleaned_message'])

# Encode labels: spam = 1, ham = 0
y = df['label'].map({'ham': 0, 'spam': 1})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

# Use bigrams + unigrams in vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df['cleaned_message'])

# Same label encoding and split
y = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

def predict_sms(text):
    # Preprocess the input text just like training data
    cleaned = preprocess_text(text)
    
    # Vectorize the cleaned text
    vect_text = vectorizer.transform([cleaned])
    
    # Predict label
    pred = model.predict(vect_text)[0]
    
    return "Spam" if pred == 1 else "Ham"

print(predict_sms("Congratulations! You have won a free ticket. Call now!"))
print(predict_sms("Hey, are we still meeting for lunch today?"))
print(predict_sms("Win cash prizes by texting WIN to 12345"))
print(predict_sms("Can you send me the report by tomorrow?"))
