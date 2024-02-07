import joblib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.lang_detection import language_detection
from lib.translation import translate_to_english

# Simple model
# Load the saved classifier and vectorizer
# API
model_filename = '../model.joblib'
# Local
# model_filename = 'model.joblib'
loaded_classifier, loaded_tfidf_vectorizer = joblib.load(model_filename)


# Function to predict suicidal text using the loaded model
def predict_suicidal_text_loaded(text):
    # Detect the language of user text
    lang = language_detection(text)
    if lang != "en":
        # If it's not English, translate it to English automatically
        text = translate_to_english(text, lang)
    text_tfidf = loaded_tfidf_vectorizer.transform([text])
    score = loaded_classifier.predict_proba(text_tfidf)[0]
    prediction = loaded_classifier.predict(text_tfidf)
    # Return prediction, Translated text to English and Scores
    return prediction[0], text, score

