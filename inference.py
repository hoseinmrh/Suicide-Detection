import joblib
from lib import tranlation

# Simple model
# Load the saved classifier and vectorizer
model_filename = '../model.joblib'
loaded_classifier, loaded_tfidf_vectorizer = joblib.load(model_filename)


# Function to predict suicidal text using the loaded model
def predict_suicidal_text_loaded(text):
    english_text = tranlation.translate_persian_to_english(text)
    text_tfidf = loaded_tfidf_vectorizer.transform([english_text])
    prediction = loaded_classifier.predict(text_tfidf)
    return prediction[0], english_text


# Example usage with the loaded model
# user_text = "احساس خوبی به زندگی ندارم و نمیخوام که ادامه بدم"
# english_text = tranlation.translate_persian_to_english(user_text)
# print("User Input:", english_text)
# loaded_prediction = predict_suicidal_text_loaded(english_text)
# print(f'Text classification: {loaded_prediction}')
