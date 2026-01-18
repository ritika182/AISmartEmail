import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


df = pd.read_csv("data/processed/final_email_dataset.csv")
urgency_map = {"low": 0, "medium": 1, "high": 2}
df["urgency_label"] = df["urgency"].map(urgency_map)

X_train, X_test, y_train, y_test = train_test_split(
    df["email_text"],
    df["urgency_label"],
    test_size=0.2,
    random_state=42,
    stratify=df["urgency_label"]
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

urgent_keywords = [
    "urgent", "immediately", "asap", "refund", "failed",
    "not working", "complaint", "error", "issue", "delay"
]

def keyword_urgency(text):
    text = text.lower()
    for word in urgent_keywords:
        if word in text:
            return 2  
    return None

ml_predictions = model.predict(X_test_vec)
final_predictions = []

for text, ml_pred in zip(X_test, ml_predictions):
    rule_pred = keyword_urgency(text)
    if rule_pred is not None:
        final_predictions.append(rule_pred)
    else:
        final_predictions.append(ml_pred)

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_predictions))
print("F1 Score (weighted):",
      f1_score(y_test, final_predictions, average="weighted"))
