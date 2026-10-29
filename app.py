
import streamlit as st
import joblib

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬"
)

@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

model = load_model()


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

    positive_score = sum(1 for word in positive_words if word in text)
    negative_score = sum(1 for word in negative_words if word in text)

    if any(word in text for word in ["illa", "not", "illaama"]):
        positive_score, negative_score = negative_score, positive_score

    if positive_score > negative_score:
        return "Positive"
    elif negative_score > positive_score:
        return "Negative"
    return "Neutral"


st.title("💬 Sentiment Analyzer")
st.write("Analyze English and basic Tanglish sentences.")

language = st.selectbox(
    "Choose input language",
    ["English (ML Model)", "Tanglish (Keyword Rules)"]
)

user_text = st.text_area(
    "Enter your sentence",
    placeholder="Example: I love this product",
    height=120
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a sentence.")
    else:
        if language == "English (ML Model)":
            prediction = model.predict([user_text])[0]
            probabilities = model.predict_proba([user_text])[0]

            st.subheader("Prediction")

            if prediction == "positive":
                st.success("😊 Positive")
            elif prediction == "negative":
                st.error("😞 Negative")
            else:
                st.info("😐 Neutral")

            st.subheader("Model probability estimates")
            for sentiment, probability in zip(
                model.classes_, probabilities
            ):
                st.write(f"{sentiment.capitalize()}: {probability:.2%}")
                st.progress(float(probability))

        else:
            prediction = analyze_tanglish(user_text)
            st.subheader("Prediction")

            if prediction == "Positive":
                st.success("😊 Positive")
            elif prediction == "Negative":
                st.error("😞 Negative")
            else:
                st.info("😐 Neutral")

            st.caption("Tanglish prediction uses simple keyword rules.")

st.divider()
st.caption("Python | NLP | TF-IDF | Logistic Regression | Streamlit")
