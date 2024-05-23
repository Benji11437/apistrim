import streamlit as st
import numpy as np
import joblib
from joblib import load
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re, nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator

st.title("Prediction des sentiments")
st.markdown("Cette application analyse le texte et nous donne le sentiment associé au texte ")

# Charger le modèle et le vectoriseur
model = load('new_model.joblib')
vectorizer = load('vectoriz.joblib')

# L'utilisateur saisit le texte de son tweet
label = st.text_input(label="Pseudo :")
corpus = st.text_area("Please Enter your text")

def cleaned_text(text: str):
    text = str(text)
    clean = re.sub(r"\n", " ", text)
    clean = clean.lower()
    clean = re.sub(r"[~.,%/:;?_&+*=!-]", " ", clean)
    clean = re.sub(r"[^a-z]", " ", clean)
    clean = clean.strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean

def analyze_sentiment(text):
    # Traduire le texte en anglais
    translated_text = GoogleTranslator(source="fr", target="en").translate(text)
    
    # Nettoyer le texte
    clean_text = cleaned_text(translated_text)
    
    # Vectoriser le texte
    text_vector = vectorizer.transform([clean_text])
    
    # Prédire le sentiment avec le modèle de régression logistique
    prediction = model.predict(text_vector)
    
    return "Positive" if prediction == 1 else "Negative"

def display_sentiment_emoji(sentiment):
    emojis = {"Positive": "😊", "Negative": "😢"}
    st.write(f"Emoji: {emojis.get(sentiment, '🤔')}")

if 'result' not in st.session_state:
    st.session_state.result = None
if 'confirmation' not in st.session_state:
    st.session_state.confirmation = None


if st.button("Analyz Sentiment"):
    if corpus:
        st.session_state.result = analyze_sentiment(corpus)
        st.session_state.confirmation = None  # Reset confirmation when new analysis is done
        st.write(label)
        st.write("Votre texte a un sentiment:", st.session_state.result)
        display_sentiment_emoji(st.session_state.result)
    else:
        st.warning("Veuillez entrer le texte à analyser.")

if st.session_state.result:
    confirmation = st.radio(
        "La prédiction est-elle correcte ?",
        ("", "Oui", "Non"),
        index=None)        
    

    if confirmation == "Oui":
        
        st.session_state.confirmation = "Merci pour votre confirmation ! 😊"
    elif confirmation == "Non":
        st.session_state.confirmation = "Désolé, nous ferons mieux la prochaine fois. 😢"

if st.session_state.confirmation:
    st.write(st.session_state.confirmation)


    
    
       

           