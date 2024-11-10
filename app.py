!pip install openai==0.28
!python -m spacy download en_core_web_sm
import openai
import spacy
import streamlit as st

# Set OpenAI API key here
openai.api_key = st.secrets["OPENAI_API_KEY"]  # It's better to use Streamlit's Secrets Manager for security

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Function to analyze sentiment with word-level contributions using GPT
def analyze_sentiment_with_words(review, category):
    prompt = f"Analyze the sentiment of the following {category} review and provide sentiment contributions for each word (percentage):\n\nReview: {review}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    sentiment_analysis = response['choices'][0]['message']['content']
    return sentiment_analysis.strip()

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
        st.markdown(sentiment_with_contributions)

    # Perform Named Entity Recognition (NER)
    if st.button("Extract Named Entities"):
        st.subheader("Named Entities in the Review")
        entities = extract_entities(review)
        st.write(entities)

else:
    st.warning("Please enter a review to analyze.")
