import re
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Similarity:
    levenshtein = {}
    jaccard = {}
    tfidf = {}

    def clean_text(self, text):
        # Remove pontuações e converte para minúsculas
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    def get_levenshtein_similarity(self, text_a, text_b):
        return difflib.SequenceMatcher(None, text_a, text_b).ratio()

    def get_jaccard_similarity(self, text_a, text_b):
        set_a = set(text_a.split())
        set_b = set(text_b.split())

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return intersection / union if union != 0 else 0  # Evita divisão por zero

    def get_tfidf_similarity(self, text_a, text_b):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    def get_similarities(self, model_name: str, text_a: str, text_b: str):
        cleaned_text_a = self.clean_text(text_a)
        cleaned_text_b = self.clean_text(text_b)

        similaridade_1 = self.get_levenshtein_similarity(cleaned_text_a, cleaned_text_b)

        self.levenshtein[model_name] = similaridade_1 * 100

        similaridade_2 = self.get_jaccard_similarity(cleaned_text_a, cleaned_text_b)

        self.jaccard[model_name] = similaridade_2 * 100

        similaridade_3 = self.get_tfidf_similarity(cleaned_text_a, cleaned_text_b)

        self.tfidf[model_name] = similaridade_3 * 100
