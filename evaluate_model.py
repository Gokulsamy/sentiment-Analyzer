
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Example dataset
texts = [
    "I love this product",
    "This is wonderful",
    "Amazing experience",
    "I am very happy",
    "The food is delicious",
    "I hate this product",
    "This is terrible",
    "Very bad experience",
    "I am very disappointed",
    "The food is awful",
    "It is okay",
    "The product is average",
    "Nothing special",
    "It is neither good nor bad",
    "The experience was normal",
    "Absolutely fantastic",
    "I feel great",
    "Worst service ever",
    "I dislike this",
    "The quality is acceptable"
]

labels = [
    "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "positive", "positive", "negative", "negative", "neutral"
]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# Create the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))
