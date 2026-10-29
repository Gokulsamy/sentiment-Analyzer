# Sentiment Analyzer — NLP Project

## Project Overview
This project analyzes text and predicts whether the sentiment is Positive, Negative, or Neutral.

It includes:
- English sentiment prediction using Machine Learning
- TF-IDF text feature extraction
- Logistic Regression classification
- Basic Tanglish sentiment analysis using keyword rules
- Streamlit web interface
- Model evaluation using accuracy, classification report, and confusion matrix

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Matplotlib
- Joblib

## Project Files
- `app.py` — Web application
- `sentiment_data.csv` — Example labeled dataset
- `clean_dataset.py` — Dataset cleaning
- `train_csv_model.py` — Train from CSV
- `train_clean_model.py` — Train from cleaned CSV
- `evaluate_model.py` — Model evaluation
- `confusion_matrix.py` — Confusion matrix visualization
- `tanglish_sentiment.py` — Basic Tanglish rule-based analyzer
- `sentiment_model.pkl` — Saved trained model
- `requirements.txt` — Python dependencies

## How to Run

1. Install Python.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install dependencies:

   `python -m pip install -r requirements.txt`

5. Train the model if needed:

   `python train_clean_model.py`

6. Start the web application:

   `python -m streamlit run app.py`

## Limitations
The sample dataset is small and manually created. Predictions and probability estimates are for learning purposes and are not reliable measures of real-world sentiment. Tanglish analysis uses simple keyword rules.

## Future Improvements
- Use a larger, independently labeled dataset.
- Evaluate with a separate test dataset.
- Improve Tamil and Tanglish language support.
- Compare Logistic Regression with other models.
- Add more tests and improve the user interface.
