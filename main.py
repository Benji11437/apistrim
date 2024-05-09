import streamlit as st
import numpy as np
import joblib
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re, nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator

st.title("Prediction des sentiments")
st.markdown("Cette application analyse les phrases et nous donne le sentiment exprimé dans cette phrase ")

# 
#model = joblib.load(filename='model_joblib')



# L'utilisateur saisie le text de son twitt
label = st.text_input(label="Pseudo :")
corpus = st.text_area("Please Enter your text")

def cleaned_text(text):
    clean = re.sub("\n"," ",text)
    clean=clean.lower()
    clean=re.sub(r"[~.,%/:;?_&+*=!-]"," ",clean)
    clean=re.sub("[^a-z]"," ",clean)
    clean=clean.lstrip()
    clean=re.sub("\s{2,}"," ",clean)
    return clean

clean_corpus = cleaned_text(corpus)
text_en = GoogleTranslator(source="fr", target="en").translate(clean_corpus)


def analyze_sentiment(text):
    blob = TextBlob(text)    
    polarity = blob.sentiment.polarity

    if polarity > 0:
        return "Positive"
    
    else:
        return "Negative"
    

def display_sentiment_emoji(sentiment):
    emojis = {"Positive": "😊", "Negative": "😢"}
    st.write(f"Emoji: {emojis.get(sentiment, '🤔')}")       


if st.button("Analyze Sentiment"):
        # Vérification si le champ de texte n'est pas vide
        if text_en:
            # Prédiction du sentiment
            result = analyze_sentiment(text_en)

            # Affichage du résultat
            if result == 1:
                st.success("Sentiment positif")
                display_sentiment_emoji(result)
            else:
                st.error("Sentiment négatif")
                display_sentiment_emoji(result)
        else:
            st.warning("Veuillez entrer le texte du tweet")


        #if st.button("Analyze Sentiment"):
        #result = analyze_sentiment(text_en)
        #st.write(label)
        #st.write("Votre tweet(X) a un sentiment:", result)
        #display_sentiment_emoji(result)
        