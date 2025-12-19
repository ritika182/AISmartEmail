import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


DATA_PATH = "data/processed/final_email_dataset.csv"
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)


df = pd.read_csv(DATA_PATH)

X = df["email_text"]
y = df["category"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

lr_preds = lr_model.predict(X_test_vec)

print("\n Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))


nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_preds = nb_model.predict(X_test_vec)

print("\n Naive Bayes Results")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print(classification_report(y_test, nb_preds))


joblib.dump(lr_model, f"{MODEL_PATH}/logistic_regression.pkl")
joblib.dump(nb_model, f"{MODEL_PATH}/naive_bayes.pkl")
joblib.dump(vectorizer, f"{MODEL_PATH}/tfidf_vectorizer.pkl")


