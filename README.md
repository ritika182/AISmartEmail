 Project Title:AI Powered Smart Email Classifier for Enterprises

An AI-based system that automatically classifies enterprise emails into meaningful categories and assigns urgency levels using Natural Language Processing (NLP) and Machine Learning.

 Project Description:

Enterprises receive a large volume of customer support emails daily, including complaints, service requests, feedback, and spam. Manual email triaging is inefficient and delays response times.

This project builds an AI-powered smart email classification engine that automatically categorizes emails and assigns urgency levels, helping customer support teams prioritize critical issues and improve productivity.

Objectives:

-> Automatically classify emails into:
- Complaint - Request - Feedback - Spam

-> Assign urgency levels:
- High - Medium - Low

-> Reduce manual email sorting
-> Improve customer response time


Tech Stack:

Language: Python
Libraries & Frameworks:
-pandas, numpy
-scikit-learn
-Hugging Face Transformers
-PyTorch

Models:
- Logistic Regression
- Naive Bayes
- DistilBERT


Milestone 1: Data Collection & Preprocessing

Tasks Completed

- Collected email datasets (complaints, requests, feedback, spam)
- Cleaned email text (HTML removal, URL removal, normalization)
- Selected correct email content columns
- Labeled emails with:
   Category  , Urgency level
- Created a  processed dataset

Output: "final_email_dataset.csv"

Milestone 2: Email Categorization Engine

Baseline Model:

Models Used:
- Logistic Regression
-  Naive Bayes

Approach:

- TF-IDF vectorization
-Train-test split
-Evaluation using accuracy and classification metrics

Results:
 ->Logistic Regression: ~97% accuracy
 ->Naive Bayes: ~96% accuracy
 
Transformer Model:

Model Used:
-> DistilBERT

Approach:
- Tokenization using Hugging Face tokenizer
- Fine-tuning DistilBERT for multi-class email classification
- Training completed for 2 epochs on CPU

Results:
- Successful training and model saving
-Low training and evaluation loss
- Supports all categories: Complaint, Request, Feedback , Spam


