import joblib
from langdetect import detect
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from translate import translation




# Simple model
# Load the saved classifier and vectorizer
# API Local
# model_filename = '../predict/model/model.joblib'
# API Deploy
model_filename = '/usr/src/predict/model/model.joblib'
# Local
# model_filename = 'model.joblib'
loaded_classifier, loaded_tfidf_vectorizer = joblib.load(model_filename)

def language_detection(user_text):
    lang = detect(user_text)
    return lang

# Function to predict suicidal text using the loaded model
def predict_suicidal_text_loaded(text):
    # Detect the language of user text
    lang_d = language_detection(text)
    if lang_d != "en":
        # If it's not English, translate it to English automatically
        text = translation.translate_to_english(text, lang_d)
    text_tfidf = loaded_tfidf_vectorizer.transform([text])
    score = loaded_classifier.predict_proba(text_tfidf)[0]
    prediction = loaded_classifier.predict(text_tfidf)
    # Return prediction, Translated text to English and Scores
    return prediction[0], text, score, lang_d

