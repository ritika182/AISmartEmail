import pandas as pd
import re
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

os.makedirs(PROCESSED_PATH, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

frames = []

# ================= COMPLAINTS =================
complaints = pd.read_csv(
    f"{RAW_PATH}/complaints.csv",
    encoding="latin1",
    low_memory=False
)

# take first column as text
complaints = complaints.iloc[:, 0].to_frame(name="email_text")
complaints["email_text"] = complaints["email_text"].apply(clean_text)

complaints = complaints.dropna()
complaints = complaints[complaints["email_text"].str.strip() != ""]

complaint_sample_size = min(4000, len(complaints))
complaints = complaints.sample(n=complaint_sample_size, random_state=42)

complaints["category"] = "complaint"
complaints["urgency"] = "high"

frames.append(complaints)

# ================= REQUESTS =================
requests = pd.read_csv(
    f"{RAW_PATH}/Request.csv",
    encoding="latin1"
)

requests = requests.iloc[:, 0].to_frame(name="email_text")
requests["email_text"] = requests["email_text"].apply(clean_text)

requests = requests.dropna()
requests = requests[requests["email_text"].str.strip() != ""]

request_sample_size = min(3000, len(requests))
requests = requests.sample(n=request_sample_size, random_state=42)

requests["category"] = "request"
requests["urgency"] = "medium"

frames.append(requests)

# ================= FEEDBACK =================
feedback = pd.read_csv(
    f"{RAW_PATH}/feddback.csv",
    encoding="latin1"
)

feedback = feedback.iloc[:, 0].to_frame(name="email_text")
feedback["email_text"] = feedback["email_text"].apply(clean_text)

feedback = feedback.dropna()
feedback = feedback[feedback["email_text"].str.strip() != ""]

feedback["category"] = "feedback"
feedback["urgency"] = "low"

frames.append(feedback)

# ================= SPAM (FIXED PROPERLY) =================
spam = pd.read_csv(
    f"{RAW_PATH}/spam.csv",
    encoding="latin1"
)

# assume: column 0 = label (ham/spam), column 1 = message
spam = spam.iloc[:, 1].to_frame(name="email_text")

spam["email_text"] = spam["email_text"].apply(clean_text)

spam = spam.dropna()
spam = spam[spam["email_text"].str.strip() != ""]

spam_sample_size = min(2000, len(spam))
spam = spam.sample(n=spam_sample_size, random_state=42)

spam["category"] = "spam"
spam["urgency"] = "low"

frames.append(spam)

# ================= FINAL MERGE =================
final_df = pd.concat(frames, ignore_index=True)
# Limit final dataset size for GitHub
MAX_ROWS = 4000
if len(final_df) > MAX_ROWS:
    final_df = final_df.sample(n=MAX_ROWS, random_state=42)

final_df.to_csv(f"{PROCESSED_PATH}/final_email_dataset.csv", index=False)

print("✅ Final dataset created successfully")
print("📊 Total rows:", len(final_df))
print("📁 Saved at: data/processed/final_email_dataset.csv")

print("\nUrgency distribution:")
print(final_df["urgency"].value_counts())
