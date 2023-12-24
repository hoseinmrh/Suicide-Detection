import joblib
from lib import translation

# Simple model
# Load the saved classifier and vectorizer
# API
model_filename = '../model.joblib'
# Local
# model_filename = 'model.joblib'
loaded_classifier, loaded_tfidf_vectorizer = joblib.load(model_filename)


# Function to predict suicidal text using the loaded model
def predict_suicidal_text_loaded(text):
    english_text = translation.translate_persian_to_english(text)
    text_tfidf = loaded_tfidf_vectorizer.transform([english_text])
    score = loaded_classifier.predict_proba(text_tfidf)[0]
    prediction = loaded_classifier.predict(text_tfidf)
    return prediction[0], english_text, score


# Example usage with the loaded model
# user_text = "فوتبال قشنگ است"
# # english_text = translation.translate_persian_to_english(user_text)
# # print("User Input:", english_text)
# loaded_prediction = predict_suicidal_text_loaded(user_text)
# print(f'Text classification: {loaded_prediction}')
