from google.cloud import translate_v3 as translate

def translate_text(text):
    '''This function is written to just translate one string and not a list of strings'''
    client = translate.TranslationServiceClient()
    
    response = client.translate_text(
            parent='projects/hinglish-banker-inwvsx',
            contents=[text],
            mime_type="text/plain",  # mime types: text/plain, text/html
            source_language_code="en-US",
            target_language_code="hi",
            )
            
    return response.translations[0].translated_text

