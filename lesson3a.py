
from textblob import TextBlob

sentences = [
    "I love this product!",
    "This product is terrible.",
    "The product arrived yesterday."
]

for sentence in sentences:
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print("Sentence:", sentence)
    print("Sentiment:", sentiment)
    print("Polarity:", round(polarity, 2))
    print("Subjectivity:", round(subjectivity, 2))
    print("-" * 40)
