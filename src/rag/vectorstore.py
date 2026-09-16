"""
Local Vector Store and Semantic Search module.
Generates embeddings for documentation chunks and performs top-k similarity search.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.rag.ingestion import DocChunk


class LocalVectorStore:
    """
    Local Vector Store providing embedding indexing and semantic retrieval.
    Supports SentenceTransformers + FAISS / NumPy cosine similarity search.
    Includes deterministic fallback encoder if PyTorch/Transformers models are offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: List[DocChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._st_model = None
        self._use_sentence_transformers = False

        # Attempt loading sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            self._use_sentence_transformers = True
        except Exception:
            self._use_sentence_transformers = False

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates embedding vector for a given text."""
        if self._use_sentence_transformers and self._st_model is not None:
            vec = self._st_model.encode(text, convert_to_numpy=True)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        else:
            # Deterministic term-frequency hash vectorizer fallback (384 dimensions)
            vec = np.zeros(384, dtype=np.float32)
            words = text.lower().split()
            for word in words:
                idx = sum(ord(c) for c in word) % 384
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

    def build_index(self, chunks: List[DocChunk]) -> int:
        """Indexes documentation chunks into vector memory."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            return 0

        vec_list = [self._get_embedding(c.content) for c in self.chunks]
        self.embeddings = np.array(vec_list, dtype=np.float32)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[DocChunk, float]]:
        """
        Performs cosine similarity search for query and returns top_k (DocChunk, similarity_score) pairs.
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_vec = self._get_embedding(query)
        # Compute cosine similarities
        scores = np.dot(self.embeddings, query_vec)

        # Sort indices by highest score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[DocChunk, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            results.append((self.chunks[idx], round(score, 4)))

        return results
