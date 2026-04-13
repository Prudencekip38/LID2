import streamlit as st
import pickle
import re

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Language Identifier")
st.write("NLP Project by: 110062733")

text = st.text_area("Enter your text here:")

if st.button("Detect Language"):
    if text.strip():
        cleaned = preprocess(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Language: **{prediction}**")
    else:
        st.warning("Please enter some text.")