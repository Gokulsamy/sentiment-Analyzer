
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
data = pd.read_csv("cleaned_sentiment_data.csv")

# Check dataset
print("Dataset size:", len(data))
print("\nSentiment distribution:")
print(data["sentiment"].value_counts())

# Separate input and output
X = data["text"]
y = data["sentiment"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("\nAccuracy:", f"{accuracy_score(y_test, predictions):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save model
joblib.dump(model, "sentiment_model.pkl")

print("\nFinal model saved as sentiment_model.pkl")
