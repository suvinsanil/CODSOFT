import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys

# 1. Load Raw Lines from the TXT File (Path Fixed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'Data', 'train_data.txt')

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 2. Parse Each Line into Structured Data
data = []
for line in lines:
    parts = line.strip().split(' ::: ')
    if len(parts) == 4:
        _, movie_title, genre, plot = parts
        data.append({'title': movie_title, 'genre': genre, 'plot_summary': plot})
df = pd.DataFrame(data)

print(df.isnull().sum())
null_count = df.isnull().sum().sum()
if null_count > 0:
    print(f"There are {null_count} null values. Please fix it.")
    sys.exit()
else:
    print("Code processed successfully. No null values.")

print("Data Loaded:", df.shape)
print(df.head())

# 3. Clean the Plot Summaries
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_plot'] =df['plot_summary'].apply(clean_text)

# 4.Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_plot'])

y = df['genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)

# 5.Model Training
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# 6.Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7.Prediction
def predict_genre(plot_text):
    cleaned = clean_text(plot_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return prediction[0]

# Example Prediction:
sample = "A young boy discovers he has superpowers and battles villains to save the world."
predicted_genre = predict_genre(sample)
print(f"\nPredicted Genre: {predicted_genre}")
