"""Command Line Interface (CLI) for Simple RAG Chatbot.

Provides an interactive terminal menu for managing documents, building
the vector knowledge base, querying the chatbot, and viewing storage stats.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure proper UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import DEBUG_MODE, DOCUMENTS_DIR, OPENAI_API_KEY
from rag_pipeline import RAGPipeline
from utils import logger


class RAGChatbotCLI:
    """Handles CLI presentation and user interaction for the RAG chatbot."""

    SEPARATOR = "=" * 50
    SUB_SEPARATOR = "-" * 50

    def __init__(self) -> None:
        logger.info("Starting Simple RAG Chatbot CLI application...")
        self.pipeline = RAGPipeline()

    def print_header(self, title: str) -> None:
        """Print a formatted visual header."""
        print(f"\n{self.SEPARATOR}")
        print(f"{title.center(50)}")
        print(f"{self.SEPARATOR}\n")

    def display_main_menu(self) -> None:
        """Render the primary application menu."""
        self.print_header("SIMPLE RAG CHATBOT")
        print("1. Add Documents")
        print("2. Build Knowledge Base")
        print("3. Ask Questions")
        print("4. View Knowledge Base Info")
        print("5. Clear Knowledge Base")
        print("6. Exit")
        print()

    def run(self) -> None:
        """Main execution loop."""
        # Initial greeting and status check
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("your_"):
            print("\n💡 NOTE: No OPENAI_API_KEY set in .env.")
            print("   The pipeline will run with deterministic offline embeddings and demo responses.")
            print("   Add your OpenAI API key in .env for full AI generation.\n")

        while True:
            try:
                self.display_main_menu()
                choice = input("Enter your choice (1-6): ").strip()

                if choice == "1":
                    self.handle_add_document()
                elif choice == "2":
                    self.handle_build_knowledge_base()
                elif choice == "3":
                    self.handle_chat_mode()
                elif choice == "4":
                    self.handle_view_info()
                elif choice == "5":
                    self.handle_clear_knowledge_base()
                elif choice == "6":
                    print("\nThank you for using Simple RAG Chatbot. Goodbye!\n")
                    logger.info("User requested exit. Shutting down...")
                    break
                else:
                    print("\n⚠️ Invalid choice. Please enter a number between 1 and 6.")

            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user. Returning to main menu...")
            except Exception as e:
                print(f"\n❌ An unexpected error occurred: {e}")
                logger.error("Unexpected error in main CLI loop: %s", e)

    def handle_add_document(self) -> None:
        """Prompt user for a file path, validate, and stage the document."""
        self.print_header("ADD DOCUMENT")
        print("Supported formats: .txt, .pdf, .md, .docx")
        print("Type 'back' to return to the main menu.\n")

        file_input = input("Enter document path: ").strip()

        # Strip surrounding quotes if user drag-and-dropped a path
        if (file_input.startswith('"') and file_input.endswith('"')) or (
            file_input.startswith("'") and file_input.endswith("'")
        ):
            file_input = file_input[1:-1]

        if not file_input:
            print("\n⚠️ Document path cannot be empty.")
            return

        if file_input.lower() == "back":
            return

        print("\nValidating and loading document...")
        result = self.pipeline.add_document(file_input)

        if result["success"]:
            print("\n✅ Document loaded successfully!")
            print(f"   File: {result['file_name']}")
            print(f"   Size: {result['file_size']}")
            print(f"   Total Characters: {result['total_chars']}")
            print(f"   Sections/Pages: {result['pages_or_sections']}")
            print(f"\nDocument is staged in '{DOCUMENTS_DIR.name}/'.")
            print("Select '2. Build Knowledge Base' from the main menu to index it.")
        else:
            print(f"\n❌ {result['message']}")

    def handle_build_knowledge_base(self) -> None:
        """Trigger chunking, embedding generation, and vector indexing."""
        self.print_header("BUILD KNOWLEDGE BASE")

        def on_progress(msg: str) -> None:
            print(f"  {msg}")

        print("Starting knowledge base compilation...\n")
        result = self.pipeline.build_knowledge_base(progress_callback=on_progress)

        if result["success"]:
            print(f"\n{self.SUB_SEPARATOR}")
            print("✅ Knowledge base built successfully!")
            print(f"   New/Updated Documents: {result['indexed_count']}")
            print(f"   Unchanged Documents:   {result['skipped_count']}")
            print(f"   New Chunks Indexed:    {result['new_chunks']}")
            print(f"   Total Stored Chunks:   {result['total_chunks']}")
            print(f"{self.SUB_SEPARATOR}")
        else:
            print(f"\n⚠️ {result['message']}")

    def handle_chat_mode(self) -> None:
        """Enter interactive Q&A chat mode."""
        self.print_header("RAG CHATBOT")
        print("Ask questions based on your indexed documents.")
        print("Special Commands:")
        print("  • 'exit' or 'quit' : Return to main menu")
        print("  • 'help'           : Show this help message")
        print("  • 'sources'        : Show sources of last answer")
        print("  • 'clear'          : Clear active conversation history\n")

        while True:
            try:
                question = input("\nYou: ").strip()

                if not question:
                    print("⚠️ Please enter a question.")
                    continue

                lower_q = question.lower()

                if lower_q in ("exit", "quit", "q"):
                    print("Returning to main menu...")
                    break

                if lower_q == "help":
                    print("\nAvailable Commands:")
                    print("  exit    - Return to main menu")
                    print("  help    - Show available commands")
                    print("  sources - Display detailed chunk sources for the last answer")
                    print("  clear   - Reset conversation memory for this chat session")
                    continue

                if lower_q == "clear":
                    self.pipeline.clear_chat_history()
                    print("\n🧹 Conversation history cleared.")
                    continue

                if lower_q == "sources":
                    self._display_last_sources()
                    continue

                # Process query through RAG pipeline
                print("\nSearching knowledge base and generating answer...")
                result = self.pipeline.ask(question)

                # Optional Debug Mode output
                if self.pipeline.debug_mode and result.get("retrieved_chunks"):
                    self._display_debug_chunks(result["retrieved_chunks"], result.get("search_query"))

                # Display Answer
                print(f"\n{self.SEPARATOR}")
                print("ANSWER")
                print(self.SEPARATOR)
                print(result["answer"])

                # Display Sources if available
                if result.get("sources"):
                    print(f"\n{self.SEPARATOR}")
                    print("SOURCES")
                    print(self.SEPARATOR)
                    for idx, chunk in enumerate(result["sources"], start=1):
                        meta = chunk.get("metadata", {})
                        fname = meta.get("file_name", "Unknown File")
                        cid = meta.get("chunk_id", idx)
                        page = meta.get("page")
                        page_str = f" (Page {page})" if meta.get("file_type") == ".pdf" and page else ""
                        score = chunk.get("score")
                        score_str = f" [Similarity: {score:.2f}]" if score is not None else ""
                        print(f"{idx}. {fname} — Chunk {cid}{page_str}{score_str}")

            except KeyboardInterrupt:
                print("\nReturning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Error answering question: {e}")
                logger.error("Error in chat loop: %s", e)

    def _display_last_sources(self) -> None:
        """Display extended chunk previews from the most recent answer."""
        sources = self.pipeline.get_last_sources()
        if not sources:
            print("\nℹ️ No sources available. (Either no question has been asked yet, or no relevant chunks were found).")
            return

        print(f"\n{self.SEPARATOR}")
        print("SOURCES FROM LAST ANSWER")
        print(self.SEPARATOR)
        for idx, chunk in enumerate(sources, start=1):
            meta = chunk.get("metadata", {})
            fname = meta.get("file_name", "Unknown File")
            cid = meta.get("chunk_id", idx)
            page = meta.get("page")
            page_info = f", Page {page}" if meta.get("file_type") == ".pdf" and page else ""
            print(f"\n[{idx}] {fname}{page_info} (Chunk {cid}):")
            print(f"    \"{chunk['content'][:250]}...\"")

    def _display_debug_chunks(self, chunks: list[dict[str, Any]], query_used: str | None) -> None:
        """Display internal retrieval details when debug mode is enabled."""
        print(f"\n🔍 [DEBUG MODE] Search Query: '{query_used}'")
        print(f"🔍 [DEBUG MODE] Retrieved {len(chunks)} Chunks:")
        for idx, c in enumerate(chunks, start=1):
            meta = c.get("metadata", {})
            print(
                f"   Chunk {idx}: {meta.get('file_name')} (ID: {meta.get('chunk_id')}) "
                f"| Score: {c.get('score')} | Distance: {c.get('distance')}"
            )

    def handle_view_info(self) -> None:
        """Display knowledge base statistics and indexed files."""
        self.print_header("KNOWLEDGE BASE INFORMATION")
        info = self.pipeline.get_knowledge_base_info()

        print(f"Total Indexed Documents : {info['total_documents']}")
        print(f"Total Vector Chunks      : {info['total_chunks']}")
        print(f"Vector Database Engine   : {info['vector_store']}")
        print(f"Embedding Provider       : {info['embedding_provider']}")
        print(f"System Status            : {info['status']}")

        print("\nStaged Files in 'data/documents/':")
        if info.get("staged_files"):
            for fname in info["staged_files"]:
                print(f"  📄 {fname}")
        else:
            print("  (No files currently staged)")

        print("\nIndexed Files in Vector Store:")
        if info.get("indexed_files"):
            for item in info["indexed_files"]:
                print(f"  • {item['file_name']} — {item['chunks']} chunks (Indexed: {item['indexed_at']})")
        else:
            print("  (No files indexed yet)")

    def handle_clear_knowledge_base(self) -> None:
        """Prompt confirmation and delete the vector database."""
        self.print_header("CLEAR KNOWLEDGE BASE")
        confirm = input("Are you sure you want to delete the knowledge base? (Y/N): ").strip().upper()

        if confirm in ("Y", "YES"):
            print("\nClearing vector database and metadata...")
            success = self.pipeline.clear_knowledge_base()
            if success:
                print("✅ Knowledge base cleared successfully.")
                print("   (Note: Your original staged document files in 'data/documents/' were kept safe).")
            else:
                print("❌ Failed to clear vector database. Check logs for details.")
        else:
            print("Action cancelled.")


def main() -> None:
    """Application entry point."""
    cli = RAGChatbotCLI()
    cli.run()


if __name__ == "__main__":
    main()
