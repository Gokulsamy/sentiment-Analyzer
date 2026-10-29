
import joblib

# Load the trained ML model
model = joblib.load("sentiment_model.pkl")

# Get sentence from user
text = input("Enter a sentence: ")

if not text.strip():
    print("Please enter a sentence.")
else:
    # Predict sentiment
    prediction = model.predict([text])[0]

    # Calculate probability estimates
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    print("\nPredicted Sentiment:", prediction.upper())
    print("\nProbability Estimates:")

    for sentiment, probability in zip(classes, probabilities):
        print(f"{sentiment.capitalize()}: {probability:.2%}")
