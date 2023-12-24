import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from text_processing import create_tfidf_vectorizer
import joblib

# Load your dataset
dataset_path = 'dataset/Suicide_Detection.csv'
df = pd.read_csv(dataset_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['class'], test_size=0.2, random_state=42)

# Create a TfidfVectorizer
tfidf_vectorizer = create_tfidf_vectorizer(X_train)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.transform(X_train)

# Create a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# Save the trained classifier and vectorizer
model_filename = 'model.joblib'
joblib.dump((classifier, tfidf_vectorizer), model_filename)
print(f'Model saved at: {model_filename}')
