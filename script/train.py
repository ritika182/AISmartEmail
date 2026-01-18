import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- Load Dataset ----------------
df = pd.read_csv("data/processed/final_email_dataset.csv")

# ---------------- TF-IDF Vectorization ----------------
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
X = tfidf.fit_transform(df["email_text"])   # ✅ X is defined here

# ---------------- Category Model ----------------
category_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
category_model.fit(X, df["category"])

# ---------------- Urgency Model ----------------
urgency_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
urgency_model.fit(X, df["urgency"])

# ---------------- Save Models ----------------
with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("models/category_model.pkl", "wb") as f:
    pickle.dump(category_model, f)

with open("models/urgency_model.pkl", "wb") as f:
    pickle.dump(urgency_model, f)

print("✅ Models trained and saved successfully")
