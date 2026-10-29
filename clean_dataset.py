
import pandas as pd
import re

# Load dataset
data = pd.read_csv("sentiment_data.csv")

print("Original dataset shape:", data.shape)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Remove rows with missing text or sentiment
data = data.dropna(subset=["text", "sentiment"])

# Remove duplicate rows
data = data.drop_duplicates()

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["text"] = data["text"].apply(clean_text)

# Remove empty sentences
data = data[data["text"] != ""]

# Standardize sentiment labels
data["sentiment"] = data["sentiment"].str.strip().str.lower()

# Keep only valid sentiment labels
valid_labels = ["positive", "negative", "neutral"]
data = data[data["sentiment"].isin(valid_labels)]

# Save cleaned dataset
data.to_csv("cleaned_sentiment_data.csv", index=False)

print("\nCleaned dataset shape:", data.shape)
print("\nSentiment counts:")
print(data["sentiment"].value_counts())

print("\nCleaned dataset preview:")
print(data.head())

print("\nSaved as cleaned_sentiment_data.csv")
