import openai
import spacy
import streamlit as st
import os

# Function to set the OpenAI API key securely
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Ensure to set this in Streamlit's Secrets Manager
except KeyError:
    st.error("Please set the OpenAI API key in the Streamlit secrets.")

# Check if spaCy model is installed; download it if not
def ensure_spacy_model_installed():
    model_name = "en_core_web_sm"
    try:
        spacy.load(model_name)
    except OSError:
        os.system(f"python -m spacy download {model_name}")
        st.info("Installing the spaCy model... Please wait.")
        try:
            spacy.load(model_name)
        except OSError:
            st.error(f"Failed to load spaCy model '{model_name}'. Please ensure it is installed correctly.")
            st.stop()

# Ensure the model is installed
ensure_spacy_model_installed()

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Rest of the code remains unchanged
