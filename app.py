import streamlit as st
import pickle
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Kenyan Language Identifier")
st.write("A Language Identification System for Kenyan Languages using Machine Learning")
st.write("NLP Project by: Prudence Kiprop")


text = st.text_area(
    "Enter your text here:",
    placeholder="Type in Kiswahili, Luo, Kikuyu, Sheng' or any other language..."
)
if st.button("Detect Language"):
    if text.strip():
        cleaned = preprocess(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Language: **{prediction}**")
    else:
        st.warning("Please enter some text.")

