
def analyze_tanglish(text):
    text = text.lower()

    positive_words = [
        "nalla", "super", "semma", "awesome",
        "happy", "pidichirukku", "mass", "vera level"
    ]

    negative_words = [
        "mokka", "worst", "bad", "mosam",
        "pidikala", "boring", "sad", "kevalam"
    ]

    positive_score = sum(
        1 for word in positive_words if word in text
    )

    negative_score = sum(
        1 for word in negative_words if word in text
    )

    # Simple negation handling
    negation_words = ["illa", "illa", "not", "illaama"]

    if any(word in text for word in negation_words):
        positive_score, negative_score = (
            negative_score,
            positive_score
        )

    if positive_score > negative_score:
        return "Positive 😊"
    elif negative_score > positive_score:
        return "Negative 😞"
    else:
        return "Neutral 😐"


print("Tanglish Sentiment Analyzer")
print("Type 'exit' to stop.")

while True:
    sentence = input("\nEnter Tanglish sentence: ")

    if sentence.lower() == "exit":
        print("Thank you!")
        break

    result = analyze_tanglish(sentence)
    print("Sentiment:", result)
