from sentence_transformers import SentenceTransformer
from typing import List

class Embeddings:
    """
    Wrapper for text → vector using SentenceTransformers.
    Default model: all-MiniLM-L6-v2 (384 dimensions).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text)
        return [float(x) for x in vec]