import streamlit as st
from transformers import pipeline

st.title("Campus Comment Sentiment Checker")

@st.cache_resource
# load the model once, not on every click
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

text = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please type a comment first.")
    else:
        result = classifier(text)[0]
        st.write("Label:", result["label"].lower())
        st.write("Score:", round(result["score"], 3))