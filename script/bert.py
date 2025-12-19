import pandas as pd
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
import torch
import os


DATA_PATH = "data/processed/final_email_dataset.csv"
MODEL_PATH = "models/distilbert"
os.makedirs(MODEL_PATH, exist_ok=True)


df = pd.read_csv(DATA_PATH)


label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["category"])

num_labels = len(label_encoder.classes_)


dataset = Dataset.from_pandas(df[["email_text", "label"]])


dataset = dataset.train_test_split(test_size=0.2, seed=42)


tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["email_text"],
        padding=True,
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)


training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)


trainer.train()


trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print(" DistilBERT model trained and saved")
print(" Categories:", list(label_encoder.classes_))
