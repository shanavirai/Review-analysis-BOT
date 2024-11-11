import openai
import spacy
import streamlit as st

# Set the OpenAI API key securely
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Please set the OpenAI API key in the Streamlit secrets.")

# Try loading the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("The spaCy model 'en_core_web_sm' is not installed. Ensure that the deployment environment includes:\n\n"
             "1. A 'setup.sh' script that installs the model using `python -m spacy download en_core_web_sm`.\n"
             "2. Correct dependencies listed in 'requirements.txt'.")
    st.stop()

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
    except openai.error.OpenAIError as e:
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
