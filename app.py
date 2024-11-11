import openai
import spacy
import streamlit as st
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to set the OpenAI API key securely
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Ensure to set this in Streamlit's Secrets Manager
    logger.info("OpenAI API key set successfully.")
except KeyError:
    st.error("Please set the OpenAI API key in the Streamlit secrets.")
    logger.error("OpenAI API key not found in secrets.")

# Function to ensure spaCy model is installed without using subprocess
def install_spacy_model():
    try:
        spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully.")
    except OSError:
        st.warning("The spaCy model 'en_core_web_sm' is not installed. Please install it by running the following command in your environment:\n\npython -m spacy download en_core_web_sm")
        logger.error("spaCy model 'en_core_web_sm' not found.")

# Ensure the model is installed
install_spacy_model()

# Load spaCy NER model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy NER model loaded successfully.")
except Exception as e:
    st.error("Failed to load spaCy NER model.")
    logger.error(f"Error loading spaCy NER model: {e}")

# Function to analyze sentiment with word-level contributions using GPT
def analyze_sentiment_with_words(review, category):
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"
    logger.info("Sending request to OpenAI API for sentiment analysis.")
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        sentiment_analysis = response['choices'][0]['message']['content']
        logger.info("Received response from OpenAI API.")
        return sentiment_analysis.strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        logger.error(f"OpenAI API error: {e}")
        return ""

# Function to perform Named Entity Recognition (NER)
def extract_entities(review):
    try:
        doc = nlp(review)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.info("Entities extracted successfully.")
        return entities
    except Exception as e:
        logger.error(f"Error during NER extraction: {e}")
        return []

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
