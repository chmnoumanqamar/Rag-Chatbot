"""Text splitting and chunking module for the Simple RAG Chatbot.

Breaks down long documents into semantically coherent chunks with configurable
size and overlap while attaching comprehensive source metadata.
"""

from dataclasses import dataclass, field
from typing import Any

from config import CHUNK_OVERLAP, CHUNK_SIZE
from document_loader import Document
from utils import logger


@dataclass
class DocumentChunk:
    """Represents a text chunk extracted from a document with full provenance metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def file_name(self) -> str:
        return self.metadata.get("file_name", "unknown")

    @property
    def chunk_id(self) -> int:
        return self.metadata.get("chunk_id", 0)

    @property
    def page(self) -> int:
        return self.metadata.get("page", 1)


class TextSplitter:
    """Recursively splits text into overlapping chunks based on natural delimiters."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Delimiters in order of priority: Paragraphs, Newlines, Sentences, Words, Characters
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split a raw text string into a list of text chunk strings."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Core recursive splitting algorithm."""
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split using chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        # Merge splits up to chunk_size with chunk_overlap
        good_splits: list[str] = []
        for s in splits:
            if s.strip():
                good_splits.append(s)

        if not good_splits:
            return []

        current_chunk: list[str] = []
        current_length = 0

        for split in good_splits:
            split_len = len(split) + (len(separator) if current_chunk else 0)

            if current_length + split_len > self.chunk_size:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)

                    # Calculate overlap: backtrack elements to keep overlap length
                    overlap_chunk: list[str] = []
                    overlap_len = 0
                    for prev_s in reversed(current_chunk):
                        if overlap_len + len(prev_s) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_s)
                            overlap_len += len(prev_s) + len(separator)
                        else:
                            break

                    current_chunk = overlap_chunk
                    current_length = sum(len(x) for x in current_chunk) + (
                        len(separator) * max(0, len(current_chunk) - 1)
                    )

                # If a single split itself is larger than chunk_size, split it further
                if len(split) > self.chunk_size and new_separators:
                    sub_chunks = self._recursive_split(split, new_separators)
                    final_chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk.append(split)
                    current_length += len(split) + (len(separator) if len(current_chunk) > 1 else 0)
            else:
                current_chunk.append(split)
                current_length += split_len

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def split_documents(self, documents: list[Document]) -> list[DocumentChunk]:
        """Split a list of Document objects into indexed DocumentChunk objects with metadata.

        Args:
            documents: List of Document instances.

        Returns:
            List of DocumentChunk instances.
        """
        all_chunks: list[DocumentChunk] = []
        doc_chunk_counter: dict[str, int] = {}

        for doc in documents:
            file_name = doc.file_name
            if file_name not in doc_chunk_counter:
                doc_chunk_counter[file_name] = 1

            raw_chunks = self.split_text(doc.content)

            for chunk_str in raw_chunks:
                chunk_id = doc_chunk_counter[file_name]
                doc_chunk_counter[file_name] += 1

                # Combine parent document metadata with chunk-specific fields
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_char_count": len(chunk_str),
                    }
                )

                chunk = DocumentChunk(content=chunk_str, metadata=metadata)
                all_chunks.append(chunk)

        logger.info(
            "Split %d document page(s) into %d chunk(s) (chunk_size=%d, overlap=%d)",
            len(documents),
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks
