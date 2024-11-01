import unicodedata
import re

class TextCleaner:
    def __init__(self) -> None:
        pass
        
    def remove_accents(self, text):
        # Normalize text to decompose characters (NFD - Normal Form Decomposition)
        normalized_text = unicodedata.normalize('NFD', text)
        # Filter out the accent marks (category "Mn" denotes non-spacing marks)
        ascii_text = ''.join([c for c in normalized_text if not unicodedata.category(c) == 'Mn'])
        # Normalize back to NFC to recombine any remaining composed characters
        return unicodedata.normalize('NFC', ascii_text)
    
    def remove_advertisement_str(self, text):
        return text.replace('ADVERTISEMENT', '').strip()
    
    def replace_whitespace(self, text):
        replace_dict = {
            '\n':'',
            '\t':''
        }
        for k, v in replace_dict.items():
            text = text.replace(k, v)

        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def replace_phrases(self, text):
        replace_dict = {
            'Watch how to make this recipe.':''
        }
        for k, v in replace_dict.items():
            text = text.replace(k, v)
        return text
    
    def remove_html_tags(self, text):
        # Use regex to find any HTML tags and replace them with an empty string
        return re.sub(r'<[^>]+>.*?</[^>]+>', '', text)
    
    def remove_special_characters(self, text):
        return re.sub(r"[^a-zA-Z0-9 .,/:;\'\"[\]{}+=\-_–()*&^%$#@!~\\|°<>?]", '', text)