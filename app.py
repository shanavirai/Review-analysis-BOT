import openai
import spacy
import streamlit as st
import subprocess

# Function to set the OpenAI API key securely
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Ensure to set this in Streamlit's Secrets Manager
except KeyError:
    st.error("Please set the OpenAI API key in the Streamlit secrets.")

# Ensure spaCy model is installed by running the system command directly
try:
    spacy.load("en_core_web_sm")
except OSError:
    st.info("Installing the spaCy model... Please wait.")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        st.error("Failed to load spaCy model 'en_core_web_sm' after installation. Please ensure it is installed correctly.")
        st.stop()

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Rest of the code remains unchanged...
