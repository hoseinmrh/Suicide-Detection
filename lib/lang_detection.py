from langdetect import detect


def language_detection(user_text):
    lang = detect(user_text)
    return lang
