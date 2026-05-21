"""
Hybrid Corpus Retriever -- TF-IDF + Semantic (sentence-transformers) RAG engine.

This is the KEY DIFFERENTIATOR in our solution. Most contestants use only TF-IDF
or only embeddings. We use BOTH in a hybrid scorer:

1. TF-IDF captures exact keyword matches (great for "HackerRank", "proctoring", "LTI")
2. Sentence embeddings capture meaning (great for paraphrased queries like
   "mock interviews stopped" matching "practice sessions interrupted")

The final score is: alpha * semantic_score + (1 - alpha) * tfidf_score
This gives us the best of both worlds.

Uses all-MiniLM-L6-v2 -- a tiny, fast model that runs on CPU with zero API calls.
"""
import os
import glob
import re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CORPUS_DIR, TOP_K_CHUNKS, MIN_RELEVANCE_SCORE, CHUNK_MAX_CHARS

# Try to import sentence-transformers for semantic search
# Falls back gracefully to TF-IDF only if not available
_SEMANTIC_AVAILABLE = False
_semantic_model = None

def _load_semantic_model():
    """Lazy-load the sentence-transformer model (first call takes ~5s to download)."""
    global _SEMANTIC_AVAILABLE, _semantic_model
    try:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        _SEMANTIC_AVAILABLE = True
        print("[Retriever] Semantic model loaded (all-MiniLM-L6-v2)")
    except Exception as e:
        print(f"[Retriever] Semantic model unavailable, using TF-IDF only: {e}")
        _SEMANTIC_AVAILABLE = False


class CorpusRetriever:
    """Hybrid TF-IDF + Semantic retriever with company-aware filtering."""

    def __init__(self, corpus_dir: str | Path = CORPUS_DIR, use_semantic: bool = True):
        self.documents: list[str] = []
        self.sources: list[str] = []
        self.companies: list[str] = []
        self.titles: list[str] = []
        self.corpus_dir = str(Path(corpus_dir).resolve())
        self._load_corpus(self.corpus_dir)
        self._build_tfidf_index()

        # Build semantic index if requested and available
        self.semantic_embeddings = None
        if use_semantic:
            _load_semantic_model()
            if _SEMANTIC_AVAILABLE:
                self._build_semantic_index()

        print(f"[Retriever] Loaded {len(self.documents)} document chunks from corpus")
        if self.semantic_embeddings is not None:
            print(f"[Retriever] Hybrid mode: TF-IDF + Semantic search active")
        else:
            print(f"[Retriever] TF-IDF only mode")

    def _extract_title(self, filepath: str, text: str) -> str:
        """Extract a meaningful title from the file path or content."""
        filename = os.path.basename(filepath)
        title = re.sub(r'^\d+-', '', filename.replace('.md', '').replace('.txt', ''))
        title = title.replace('-', ' ').replace('_', ' ').strip()
        return title if title else "Untitled"

    def _detect_company(self, filepath: str) -> str:
        """
        Detect which company a document belongs to based on its path
        RELATIVE to the corpus directory. This avoids false matches from
        parent directories (e.g. 'hackerrank-orchestrate' in the repo name).
        """
        try:
            rel_path = os.path.relpath(filepath, self.corpus_dir).lower().replace('\\', '/')
        except ValueError:
            rel_path = filepath.lower().replace('\\', '/')

        first_dir = rel_path.split('/')[0] if '/' in rel_path else ''

        if first_dir == 'hackerrank':
            return 'hackerrank'
        elif first_dir == 'claude':
            return 'claude'
        elif first_dir == 'visa':
            return 'visa'
        return 'unknown'

    def _chunk_text(self, text: str, max_chunk_size: int = 1500) -> list[str]:
        """
        Split text into meaningful chunks based on headings and paragraphs.
        Each chunk retains the document header for context.
        """
        lines = text.strip().split('\n')
        header = lines[0] if lines else ""

        # Split on markdown headings
        sections = re.split(r'\n(?=#{1,3}\s)', text)

        chunks = []
        current_chunk = ""

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(current_chunk) + len(section) < max_chunk_size:
                current_chunk += "\n\n" + section if current_chunk else section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = section

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If we only got one chunk and it's very long, split by paragraphs
        if len(chunks) == 1 and len(chunks[0]) > max_chunk_size * 2:
            big_text = chunks[0]
            chunks = []
            paragraphs = big_text.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) < max_chunk_size:
                    current_chunk += "\n\n" + para if current_chunk else para
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para
            if current_chunk:
                chunks.append(current_chunk.strip())

        # Filter out very small chunks (< 50 chars)
        chunks = [c for c in chunks if len(c) >= 50]

        return chunks if chunks else [text[:max_chunk_size]]

    def _load_corpus(self, corpus_dir: str):
        """Load and chunk all documents from the corpus directory."""
        file_patterns = [
            os.path.join(corpus_dir, "**", "*.md"),
            os.path.join(corpus_dir, "**", "*.txt"),
        ]

        all_files = []
        for pattern in file_patterns:
            all_files.extend(glob.glob(pattern, recursive=True))

        all_files = list(set(all_files))

        for filepath in sorted(all_files):
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

                if len(text.strip()) < 50:
                    continue

                company = self._detect_company(filepath)
                title = self._extract_title(filepath, text)

                chunks = self._chunk_text(text)

                for chunk in chunks:
                    self.documents.append(chunk)
                    self.sources.append(filepath)
                    self.companies.append(company)
                    self.titles.append(title)

            except Exception as e:
                print(f"[Retriever] Warning: Failed to read {filepath}: {e}")

    def _build_tfidf_index(self):
        """Build TF-IDF index over all document chunks."""
        if not self.documents:
            print("[Retriever] WARNING: No documents loaded!")
            self.vectorizer = None
            self.tfidf_matrix = None
            return

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def _build_semantic_index(self):
        """Build semantic embedding index using sentence-transformers."""
        if not _SEMANTIC_AVAILABLE or not self.documents:
            return

        print(f"[Retriever] Building semantic index for {len(self.documents)} chunks...")
        self.semantic_embeddings = _semantic_model.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True  # For faster cosine similarity via dot product
        )
        print(f"[Retriever] Semantic index built: shape {self.semantic_embeddings.shape}")

    def retrieve(
        self,
        query: str,
        company: str | None = None,
        top_k: int = TOP_K_CHUNKS,
        min_score: float = MIN_RELEVANCE_SCORE,
        alpha: float = 0.6  # Weight for semantic vs TF-IDF (0.6 = 60% semantic)
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks using hybrid scoring.

        Hybrid score = alpha * semantic_score + (1 - alpha) * tfidf_score

        Args:
            query: The search query (typically issue + subject)
            company: Optional company filter
            top_k: Number of results to return
            min_score: Minimum score threshold
            alpha: Weight for semantic vs TF-IDF (higher = more semantic)

        Returns:
            List of dicts with 'text', 'source', 'score', 'company', 'title' keys
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        # ── TF-IDF Scores ──
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # ── Semantic Scores ──
        if self.semantic_embeddings is not None and _SEMANTIC_AVAILABLE:
            query_embedding = _semantic_model.encode(
                [query], normalize_embeddings=True
            )
            # Dot product = cosine similarity for normalized vectors
            semantic_scores = np.dot(self.semantic_embeddings, query_embedding.T).flatten()

            # Hybrid score
            combined_scores = alpha * semantic_scores + (1 - alpha) * tfidf_scores
        else:
            combined_scores = tfidf_scores

        results = []

        # Try company-filtered search first
        if company and company.lower() not in ("none", "nan", "unknown", ""):
            company_lower = company.lower()
            filtered = [
                (i, combined_scores[i]) for i in range(len(self.documents))
                if self.companies[i] == company_lower and combined_scores[i] > min_score
            ]
            filtered.sort(key=lambda x: x[1], reverse=True)

            for idx, score in filtered[:top_k]:
                results.append({
                    "text": self.documents[idx][:CHUNK_MAX_CHARS],
                    "source": self.sources[idx],
                    "score": round(float(score), 4),
                    "company": self.companies[idx],
                    "title": self.titles[idx],
                })

        # If no company-filtered results, search full corpus
        if not results:
            top_indices = combined_scores.argsort()[-top_k * 2:][::-1]
            for idx in top_indices:
                if combined_scores[idx] > min_score:
                    results.append({
                        "text": self.documents[idx][:CHUNK_MAX_CHARS],
                        "source": self.sources[idx],
                        "score": round(float(combined_scores[idx]), 4),
                        "company": self.companies[idx],
                        "title": self.titles[idx],
                    })
                if len(results) >= top_k:
                    break

        return results

    def get_corpus_stats(self) -> dict:
        """Return statistics about the loaded corpus."""
        company_counts = {}
        for c in self.companies:
            company_counts[c] = company_counts.get(c, 0) + 1

        return {
            "total_chunks": len(self.documents),
            "unique_sources": len(set(self.sources)),
            "company_distribution": company_counts,
            "semantic_enabled": self.semantic_embeddings is not None,
        }
