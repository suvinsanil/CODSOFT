import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# Step 1: Load Data Using Relative Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
train_path = os.path.join(DATA_DIR, 'fraudTrain.csv')
test_path = os.path.join(DATA_DIR, 'fraudTest.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Step 2: Preprocessing Function

def preprocess(df):
    df = df.drop(['Unnamed: 0', 'trans_num', 'first', 'last', 'gender', 'dob', 'job', 'merchant', 'unix_time', 'street', 'city', 'state'], axis=1)
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df = df.drop(['trans_date_trans_time'], axis=1)

    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])

    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Step 3: Undersample Majority Class
fraud = train_df[train_df['is_fraud'] == 1]
non_fraud = train_df[train_df['is_fraud'] == 0].sample(n=len(fraud)*5, random_state=42)
train_balanced = pd.concat([fraud, non_fraud])

X_train = train_balanced.drop('is_fraud', axis=1)
y_train = train_balanced['is_fraud']

X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

# Step 4: Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Class distribution before SMOTE:", np.bincount(y_train))
print("Class distribution after SMOTE:", np.bincount(y_train_res))

# Step 5: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Models and Evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

for name, model in models.items():
    print(f"\n {name} ")
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Step 7: Precision-Recall Curve for Random Forest
best_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
best_model.fit(X_train_scaled, y_train_res)
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

from sklearn.metrics import f1_score

# Step 1: Get model predicted probabilities
y_scores = best_model.predict_proba(X_test_scaled)[:, 1]

# Step 2: Compute Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Step 3: Find Optimal Threshold (Max F1-score)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)  # Avoid division by zero
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print("Optimal Threshold:", optimal_threshold)
print("Best F1-Score:", f1_scores[optimal_idx])

# Step 4: Apply the Optimal Threshold
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)

# Step 5: Evaluate with New Threshold
print("\nConfusion Matrix (Optimized Threshold):\n", confusion_matrix(y_test, y_pred_optimal))
print("\nClassification Report (Optimized Threshold):\n", classification_report(y_test, y_pred_optimal))
print("ROC-AUC Score (unchanged):", roc_auc_score(y_test, y_scores))

# Step 6: Plot Precision-Recall Curve (already in your code)
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve (Random Forest)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

