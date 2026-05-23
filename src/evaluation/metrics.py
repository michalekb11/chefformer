from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

def calculate_ngram_diversity(text, n=3):
    """Returns the ratio of unique n-grams to total n-grams."""
    words = re.findall(r'\w+', text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

def calculate_self_bleu(texts: list[str]):
    """Calculates average BLEU score between generations. Lower is more diverse."""
    if len(texts) < 2:
        return 0.0
    scores = []
    smooth = SmoothingFunction().method1
    for i, target in enumerate(texts):
        references = [re.findall(r'\w+', t.lower()) for j, t in enumerate(texts) if i != j]
        hypothesis = re.findall(r'\w+', target.lower())
        scores.append(sentence_bleu(references, hypothesis, smoothing_function=smooth))
    return sum(scores) / len(scores)

def calculate_lexical_validity(text: str, common_words_set: list[str]):
    """% of words in the text that exist in a 'valid' dictionary."""
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    valid_count = sum(1 for w in words if w in common_words_set)
    return valid_count / len(words)
