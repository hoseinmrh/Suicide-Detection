import joblib
from text_processing import create_tfidf_vectorizer

# Simple model
# Load the saved classifier and vectorizer
model_filename = 'model.joblib'
loaded_classifier, loaded_tfidf_vectorizer = joblib.load(model_filename)


# Function to predict suicidal text using the loaded model
def predict_suicidal_text_loaded(text):
    text_tfidf = loaded_tfidf_vectorizer.transform([text])
    prediction = loaded_classifier.predict(text_tfidf)
    return prediction[0]


# Example usage with the loaded model
user_text = "I fill so empty without her and I can't live that way"
loaded_prediction = predict_suicidal_text_loaded(user_text)
print(f'Text classification using the loaded model: {loaded_prediction}')
