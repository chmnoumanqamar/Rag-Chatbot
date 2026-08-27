"""Central RAG pipeline coordinator for Simple RAG Chatbot.

Orchestrates document loading, text chunking, embedding generation,
vector storage, retrieval, and LLM question answering.
"""

import shutil
from pathlib import Path
from typing import Any, Callable

from chatbot import Chatbot
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEBUG_MODE,
    DOCUMENTS_DIR,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_DB_DIR,
)
from document_loader import Document, DocumentLoader
from text_splitter import DocumentChunk, TextSplitter
from utils import calculate_file_hash, format_file_size, logger, validate_file_path
from vector_store import EmbeddingManager, VectorStoreManager


class RAGPipeline:
    """End-to-end coordinator for the Retrieval-Augmented Generation workflow."""

    def __init__(
        self,
        documents_dir: Path = DOCUMENTS_DIR,
        vector_db_dir: Path = VECTOR_DB_DIR,
        debug_mode: bool = DEBUG_MODE,
    ):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.debug_mode = debug_mode

        # Initialize pipeline components
        self.text_splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager(
            db_directory=vector_db_dir,
            embedding_manager=self.embedding_manager,
        )
        self.chatbot = Chatbot()

        logger.info("RAGPipeline initialized successfully.")

    def add_document(self, file_path: str | Path) -> dict[str, Any]:
        """Validate, stage, and register a user-provided document file.

        Args:
            file_path: File system path to the target document.

        Returns:
            Dictionary with status, file name, character count, and file size.
        """
        path = Path(file_path).expanduser().resolve()
        is_valid, msg = validate_file_path(path)
        if not is_valid:
            return {"success": False, "message": msg}

        target_dest = self.documents_dir / path.name

        # Copy document into documents directory if not already there
        if path.resolve() != target_dest.resolve():
            shutil.copy2(path, target_dest)
            logger.info("Copied '%s' to '%s'", path, target_dest)

        # Verify extractable content
        try:
            docs = DocumentLoader.load_file(target_dest)
            total_chars = sum(len(d.content) for d in docs)
            file_size = format_file_size(target_dest.stat().st_size)

            return {
                "success": True,
                "file_name": path.name,
                "pages_or_sections": len(docs),
                "total_chars": total_chars,
                "file_size": file_size,
                "message": f"Document '{path.name}' added successfully.",
            }
        except Exception as e:
            logger.error("Failed to load added document %s: %s", path.name, e)
            return {"success": False, "message": f"Error loading document: {e}"}

    def build_knowledge_base(
        self,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Ingest all staged documents, generate chunks and embeddings, and store them in ChromaDB.

        Args:
            progress_callback: Optional callable to receive progress update strings.

        Returns:
            Dictionary with indexing summary statistics.
        """

        def report(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        doc_files = [
            f for f in sorted(self.documents_dir.iterdir())
            if f.is_file() and f.suffix.lower() in (".txt", ".pdf", ".md", ".docx")
        ]

        if not doc_files:
            return {
                "success": False,
                "message": "No documents found in documents directory. Please add documents first.",
                "indexed_count": 0,
                "skipped_count": 0,
                "total_chunks": 0,
            }

        report("Scanning documents in knowledge repository...")

        indexed_docs = 0
        skipped_docs = 0
        total_new_chunks = 0
        all_chunks_to_add: list[DocumentChunk] = []

        for doc_path in doc_files:
            file_name = doc_path.name
            try:
                doc_hash = calculate_file_hash(doc_path)
                status = self.vector_store.check_document_status(file_name, doc_hash)

                if status == "unchanged":
                    report(f"  • '{file_name}': Already indexed (unchanged). Skipping.")
                    skipped_docs += 1
                    continue
                elif status == "modified":
                    report(f"  • '{file_name}': Modified since last index. Re-indexing...")
                    self.vector_store.remove_document(file_name)
                else:
                    report(f"  • '{file_name}': New document detected. Processing...")

                # Extract and chunk
                docs = DocumentLoader.load_file(doc_path)
                chunks = self.text_splitter.split_documents(docs)
                all_chunks_to_add.extend(chunks)
                indexed_docs += 1

            except Exception as e:
                report(f"  ⚠️ Error processing '{file_name}': {e}")
                logger.error("Error processing document %s: %s", file_name, e)

        if all_chunks_to_add:
            report(f"Generating embeddings for {len(all_chunks_to_add)} chunk(s)...")
            report("Saving chunks into ChromaDB vector database...")
            added = self.vector_store.add_chunks(all_chunks_to_add)
            total_new_chunks = added

        stats = self.vector_store.get_stats()
        report("Knowledge base build complete!")

        return {
            "success": True,
            "indexed_count": indexed_docs,
            "skipped_count": skipped_docs,
            "new_chunks": total_new_chunks,
            "total_documents": stats["total_documents"],
            "total_chunks": stats["total_chunks"],
            "message": "Knowledge base built successfully.",
        }

    def ask(self, question: str) -> dict[str, Any]:
        """Execute the end-to-end RAG query pipeline for a user question.

        1. Validates knowledge base status and input question.
        2. Contextualizes follow-up questions using conversation history.
        3. Retrieves top relevant chunks from ChromaDB.
        4. Synthesizes an anti-hallucination grounded response via LLM.

        Args:
            question: The user query string.

        Returns:
            Dictionary containing:
                - 'answer': Generated answer string
                - 'sources': List of cited source objects
                - 'retrieved_chunks': Raw retrieved chunks with scores
                - 'search_query': Standalone search query used for vector search
        """
        clean_q = question.strip()
        if not clean_q:
            return {
                "answer": "Please enter a non-empty question.",
                "sources": [],
                "retrieved_chunks": [],
                "search_query": "",
            }

        stats = self.vector_store.get_stats()
        if stats["total_chunks"] == 0:
            return {
                "answer": (
                    "Knowledge base is empty.\n\n"
                    "Please add documents and build the knowledge base first."
                ),
                "sources": [],
                "retrieved_chunks": [],
                "search_query": clean_q,
            }

        # Step 1: Contextualize question for semantic retrieval
        search_query = self.chatbot.reformulate_query(clean_q)

        # Step 2: Retrieve relevant chunks
        logger.info("Executing similarity search for: '%s'", search_query)
        retrieved_chunks = self.vector_store.search_similar(
            query=search_query,
            top_k=TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )

        # Step 3: LLM generation
        answer, sources = self.chatbot.generate_answer(
            question=clean_q,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "search_query": search_query,
        }

    def get_knowledge_base_info(self) -> dict[str, Any]:
        """Retrieve full knowledge base and storage metrics."""
        stats = self.vector_store.get_stats()
        # Also check files in documents directory
        staged_files = [
            f.name for f in self.documents_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".txt", ".pdf", ".md", ".docx")
        ]
        stats["staged_files"] = staged_files
        stats["embedding_provider"] = (
            "OpenAI (" + self.embedding_manager.model_name + ")"
            if not self.embedding_manager.is_fallback
            else "Deterministic Offline Embeddings"
        )
        return stats

    def clear_knowledge_base(self) -> bool:
        """Clear all stored embeddings and reset chat state."""
        success = self.vector_store.clear_database()
        self.chatbot.clear_history()
        return success

    def get_last_sources(self) -> list[dict[str, Any]]:
        """Return sources from the most recent answer."""
        return self.chatbot.last_sources

    def clear_chat_history(self) -> None:
        """Clear active conversation history."""
        self.chatbot.clear_history()
