
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Small example dataset for learning
texts = [
    "I love this product",
    "This is amazing",
    "The food was delicious",
    "I am very happy",
    "Excellent service",
    "I really like this",
    "This is wonderful",
    "I hate this product",
    "This is terrible",
    "The food was awful",
    "I am very disappointed",
    "Worst service ever",
    "I really dislike this",
    "This is horrible",
    "The product is okay",
    "It is an average experience",
    "The item arrived today",
    "Nothing special about it",
    "The product is ordinary",
    "It is neither good nor bad"
]

labels = [
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative",
    "neutral", "neutral", "neutral", "neutral", "neutral", "neutral"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)

print("Model training complete!")

test_sentences = [
    "I love it",
    "This is awful",
    "It is okay"
]

for sentence in test_sentences:
    prediction = model.predict([sentence])[0]
    print(f"Text: {sentence}")
    print(f"Predicted sentiment: {prediction}")
    print("-" * 30)

import joblib
joblib.dump(model, "sentiment_model.pkl")
print("Model saved as sentiment_model.pkl")
