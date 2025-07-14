import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
csv_path = os.path.join(DATA_DIR, 'spam.csv')

df = pd.read_csv(csv_path, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# Feature Extraction 
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': LinearSVC()
}

# Train & Evaluate Each Model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nResults for {model_name}:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Save Best Model (SVM as default here)
best_model = models['Support Vector Machine']
MODEL_DIR = os.path.dirname(__file__)
joblib.dump(best_model, os.path.join(MODEL_DIR, 'spam_detector.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))

print("\nModel and vectorizer saved successfully.")

# Prediction Function
def predict_spam(text):
    text = clean_text(text)
    vect_text = vectorizer.transform([text])
    prediction = best_model.predict(vect_text)
    return 'Spam' if prediction[0] == 1 else 'Ham'

# Example Usage
sample_sms = "Congratulations! You've won a free ticket to Bahamas. Call now!"
print(sample_sms)
print("\nSample SMS Prediction:", predict_spam(sample_sms))
