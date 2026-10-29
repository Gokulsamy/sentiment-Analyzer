
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Larger example dataset for practice
positive = [
    "I love this product",
    "This is wonderful",
    "Amazing experience",
    "I am very happy",
    "The food is delicious",
    "Absolutely fantastic",
    "I feel great",
    "The service is excellent",
    "I really enjoyed this",
    "This is the best",
    "Very useful and helpful",
    "I am satisfied",
    "Beautiful and amazing",
    "The quality is great",
    "I highly recommend it",
    "The experience was perfect",
    "I am pleased with the result",
    "This made my day",
    "Excellent performance",
    "I like it very much"
]

negative = [
    "I hate this product",
    "This is terrible",
    "Very bad experience",
    "I am very disappointed",
    "The food is awful",
    "Worst service ever",
    "I dislike this",
    "This is horrible",
    "The quality is poor",
    "I regret buying this",
    "Completely useless",
    "I am unhappy",
    "Very frustrating experience",
    "The product broke immediately",
    "I do not recommend this",
    "This was a waste of money",
    "The service was disappointing",
    "I feel sad about this",
    "Really bad performance",
    "I am not satisfied"
]

neutral = [
    "It is okay",
    "The product is average",
    "Nothing special",
    "The experience was normal",
    "The product arrived yesterday",
    "It is neither good nor bad",
    "The meeting starts at ten",
    "I received the package",
    "The item is available",
    "This is a standard product",
    "The result is as expected",
    "It has three parts",
    "The shop opens today",
    "The product is on the table",
    "I read the instructions",
    "The delivery took two days",
    "It has a simple design",
    "The file was uploaded",
    "The box contains a manual",
    "The price is fifty rupees"
]

texts = positive + negative + neutral
labels = (
    ["positive"] * len(positive)
    + ["negative"] * len(negative)
    + ["neutral"] * len(neutral)
)

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the improved model
joblib.dump(model, "sentiment_model.pkl")
print("\nImproved model saved as sentiment_model.pkl")
