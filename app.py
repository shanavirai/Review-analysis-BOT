import openai
import spacy
import streamlit as st

# Function to set the OpenAI API key securely
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Ensure to set this in Streamlit's Secrets Manager
except KeyError:
    st.error("Please set the OpenAI API key in the Streamlit secrets.")

# Function to ensure spaCy model is installed
def ensure_spacy_model_installed():
    try:
        spacy.load("en_core_web_sm")
        return True
    except OSError:
        return False

# Check if spaCy model is installed; show error and stop execution if not
if not ensure_spacy_model_installed():
    st.error("The spaCy model 'en_core_web_sm' is not installed. Ensure that your deployment script includes:\n\npython -m spacy download en_core_web_sm")
    st.stop()  # Stop execution if the model isn't found

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Function to analyze sentiment with word-level contributions using GPT
def analyze_sentiment_with_words(review, category):
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        sentiment_analysis = response['choices'][0]['message']['content']
        return sentiment_analysis.strip()
    except Exception as e:  # Handle any OpenAI error
        st.error(f"OpenAI API error: {e}")
        return ""

# Function to perform Named Entity Recognition (NER)
def extract_entities(review):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Streamlit UI
st.title("Sentiment Analysis and Named Entity Recognition (NER)")

# Input: Review Category and Content
category = st.selectbox("Select Review Category:", ["Food", "Product", "Place", "Other"])
review = st.text_area(f"Enter your {category} review:")

# If a review is provided
if review:
    # Perform Sentiment Analysis
    if st.button("Analyze Sentiment"):
        st.subheader("Sentiment Analysis with Word-Level Contributions")
        sentiment_with_contributions = analyze_sentiment_with_words(review, category)
        if sentiment_with_contributions:
            st.markdown(sentiment_with_contributions)
        else:
            st.warning("Failed to retrieve sentiment analysis.")

    # Perform Named Entity Recognition (NER)
    if st.button("Extract Named Entities"):
        st.subheader("Named Entities in the Review")
        entities = extract_entities(review)
        if entities:
            st.write(entities)
        else:
            st.write("No named entities found.")
else:
    st.warning("Please enter a review to analyze.")
