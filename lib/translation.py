from googletrans import Translator

def translate_persian_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='fa', dest='en')
    return translation.text

# Example usage
# persian_text = "آرزو میکنم که ای کاش صبح از خواب بلند نشم"  # Replace with your Persian text
# english_translation = translate_persian_to_english(persian_text)
# print(f'Persian Text: {persian_text}')
# print(f'English Translation: {english_translation}')
