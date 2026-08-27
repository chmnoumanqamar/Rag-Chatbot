"""Vector database and embedding management module for Simple RAG Chatbot.

Integrates ChromaDB for persistent vector storage and manages embedding models
with automatic duplicate detection and similarity search.
"""

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_DB_DIR,
)
from text_splitter import DocumentChunk
from utils import logger


class FallbackDeterministicEmbeddings:
    """Deterministic hash-based dense embedding generator for offline testing without API keys.

    Generates normalized 1536-dimensional vectors using token hashing and n-gram term frequencies.
    Allows testing vector storage, search algorithms, and pipeline flow offline.
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h1 = hash(token) % self.dimension
            h2 = hash(token + f"_{idx}") % self.dimension
            vec[h1] += 1.0
            vec[h2] += 0.5

        # Also add bigrams for basic sequence context
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            h = hash(bigram) % self.dimension
            vec[h] += 1.5

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class EmbeddingManager:
    """Manages text embedding generation using OpenAI or local fallback."""

    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = EMBEDDING_MODEL):
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model_name
        self.is_fallback = False

        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.embedder = OpenAIEmbeddings(
                    openai_api_key=self.api_key,
                    model=self.model_name,
                )
                logger.info("Initialized OpenAI Embeddings with model: %s", self.model_name)
            except Exception as e:
                logger.warning("Could not initialize OpenAI Embeddings: %s. Using fallback.", e)
                self.embedder = FallbackDeterministicEmbeddings()
                self.is_fallback = True
        else:
            logger.info("No valid OpenAI API key found. Using deterministic offline embeddings.")
            self.embedder = FallbackDeterministicEmbeddings()
            self.is_fallback = True

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text strings to vector embeddings."""
        if not texts:
            return []
        try:
            return self.embedder.embed_documents(texts)
        except Exception as e:
            logger.error("Error generating embeddings with primary embedder: %s", e)
            if not self.is_fallback:
                logger.info("Falling back to deterministic embeddings.")
                fallback = FallbackDeterministicEmbeddings()
                return fallback.embed_documents(texts)
            raise e

    def get_query_embedding(self, query: str) -> list[float]:
        """Convert a single query string to a vector embedding."""
        try:
            return self.embedder.embed_query(query)
        except Exception as e:
            logger.error("Error generating query embedding: %s", e)
            if not self.is_fallback:
                fallback = FallbackDeterministicEmbeddings()
                return fallback.embed_query(query)
            raise e


class VectorStoreManager:
    """Coordinates persistent vector storage, indexing, and retrieval via ChromaDB."""

    COLLECTION_NAME = "rag_knowledge_base"

    def __init__(
        self,
        db_directory: str | Path = VECTOR_DB_DIR,
        embedding_manager: EmbeddingManager | None = None,
    ):
        self.db_directory = Path(db_directory)
        self.db_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.db_directory / "index_meta.json"
        self.embedding_manager = embedding_manager or EmbeddingManager()

        self._init_chroma()
        self._load_metadata()

    def _init_chroma(self) -> None:
        """Initialize ChromaDB PersistentClient."""
        self.client = chromadb.PersistentClient(
            path=str(self.db_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Knowledge base document embeddings for Simple RAG Chatbot"},
        )

    def _load_metadata(self) -> None:
        """Load document fingerprint metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.indexed_meta: dict[str, Any] = json.load(f)
            except Exception as e:
                logger.warning("Failed to parse %s: %s. Rebuilding index metadata.", self.metadata_file, e)
                self.indexed_meta = {}
        else:
            self.indexed_meta = {}

    def _save_metadata(self) -> None:
        """Persist document fingerprint metadata to JSON file."""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.indexed_meta, f, indent=2)
        except Exception as e:
            logger.error("Failed to save index metadata to %s: %s", self.metadata_file, e)

    def check_document_status(self, file_name: str, doc_hash: str) -> str:
        """Determine if a document is 'new', 'unchanged', or 'modified'.

        Returns:
            One of 'new', 'unchanged', or 'modified'.
        """
        if file_name not in self.indexed_meta:
            return "new"
        existing_hash = self.indexed_meta[file_name].get("hash")
        if existing_hash == doc_hash:
            return "unchanged"
        return "modified"

    def remove_document(self, file_name: str) -> None:
        """Delete all chunks belonging to a document from ChromaDB."""
        try:
            # Query and delete where file_name matches
            self.collection.delete(where={"file_name": file_name})
            if file_name in self.indexed_meta:
                del self.indexed_meta[file_name]
                self._save_metadata()
            logger.info("Removed document '%s' from vector store.", file_name)
        except Exception as e:
            logger.warning("Error removing '%s' from ChromaDB: %s", file_name, e)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Generate embeddings and insert document chunks into ChromaDB.

        Args:
            chunks: List of DocumentChunk instances to store.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        # Group chunks by file_name to update metadata registry
        files_in_batch: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            fname = chunk.file_name
            fhash = chunk.metadata.get("doc_hash", "")
            if fname not in files_in_batch:
                files_in_batch[fname] = {"hash": fhash, "count": 0}
            files_in_batch[fname]["count"] += 1

        # Extract contents and metadata for batch insertion
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [
            f"{chunk.file_name}_chunk_{chunk.chunk_id}_{int(time.time()*1000)}_{idx}"
            for idx, chunk in enumerate(chunks)
        ]

        logger.info("Generating embeddings for %d chunk(s)...", len(texts))
        embeddings = self.embedding_manager.get_embeddings(texts)

        logger.info("Storing %d chunk(s) in ChromaDB collection '%s'...", len(chunks), self.COLLECTION_NAME)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        # Update and persist metadata
        for fname, info in files_in_batch.items():
            self.indexed_meta[fname] = {
                "hash": info["hash"],
                "chunks": info["count"],
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        self._save_metadata()

        return len(chunks)

    def search_similar(
        self,
        query: str,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict[str, Any]]:
        """Perform semantic similarity search for a query string.

        Args:
            query: User's question or search phrase.
            top_k: Number of most similar chunks to retrieve.
            similarity_threshold: Minimum similarity score (0.0 to 1.0) to include.

        Returns:
            List of dictionaries containing:
                - 'content': Chunk text
                - 'metadata': Provenance dict (file_name, chunk_id, page, etc.)
                - 'score': Similarity score between 0.0 and 1.0 (higher = more similar)
                - 'distance': Raw vector distance
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        query_embedding = self.embedding_manager.get_query_embedding(query)

        # Perform query in ChromaDB (default metric is squared L2 distance or cosine)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            # ChromaDB default distance is L2 or cosine distance.
            # Convert distance to normalized similarity score (1 / (1 + distance))
            similarity_score = 1.0 / (1.0 + float(dist))

            if similarity_score >= similarity_threshold:
                retrieved.append(
                    {
                        "content": doc_text,
                        "metadata": meta,
                        "score": round(similarity_score, 4),
                        "distance": round(float(dist), 4),
                    }
                )

        logger.info(
            "Retrieved %d relevant chunk(s) (threshold=%.2f, top_k=%d) for query: '%s'",
            len(retrieved),
            similarity_threshold,
            top_k,
            query[:40],
        )
        return retrieved

    def get_stats(self) -> dict[str, Any]:
        """Return knowledge base statistics and list of indexed files."""
        total_chunks = self.collection.count()
        indexed_files = [
            {
                "file_name": name,
                "chunks": data.get("chunks", 0),
                "indexed_at": data.get("indexed_at", "N/A"),
            }
            for name, data in self.indexed_meta.items()
        ]
        return {
            "total_documents": len(self.indexed_meta),
            "total_chunks": total_chunks,
            "vector_store": "ChromaDB (Persistent)",
            "status": "Ready" if total_chunks > 0 else "Empty",
            "indexed_files": indexed_files,
        }

    def clear_database(self) -> bool:
        """Completely delete the vector collection and reset metadata."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self._init_chroma()
            self.indexed_meta = {}
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Cleared knowledge base and reset vector database.")
            return True
        except Exception as e:
            logger.error("Error clearing vector database: %s", e)
            return False
