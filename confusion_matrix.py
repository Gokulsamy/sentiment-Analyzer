
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Compare actual and predicted sentiments
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "neutral", "positive"]
)

print("Confusion Matrix:")
print(cm)

# Display as a graph
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot()
plt.title("Sentiment Analysis - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
print("Confusion matrix saved as confusion_matrix.png")
plt.close()
