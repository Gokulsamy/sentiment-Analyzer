
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

# Load cleaned dataset
data = pd.read_csv("cleaned_sentiment_data.csv")

X = data["text"]
y = data["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Build and train model
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Display evaluation metrics
print("Accuracy:", f"{accuracy_score(y_test, predictions):.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["negative", "neutral", "positive"],
    zero_division=0
))

# Create confusion matrix
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

# Save graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot()
plt.title("Final Sentiment Model Evaluation")
plt.tight_layout()
plt.savefig("final_confusion_matrix.png")

print("\nGraph saved as final_confusion_matrix.png")
plt.close()
