from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectorizer(train_texts):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    return tfidf_vectorizer
