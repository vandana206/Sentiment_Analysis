import streamlit as st
import joblib
import re
import nltk 
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

st.set_page_config(page_title= "sentiment_Analysis form movies",layout = "centered")
st.title("Sentiment Analysis app for services")
st.markdown("Enter a movie")

user_input = st.text_area("Enter the review")

if st.button("predict Sentiment"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.success(f"Prediction is: {sentiment}")



