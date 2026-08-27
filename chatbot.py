"""Chatbot engine module for Simple RAG Chatbot.

Coordinates LLM generation, conversational state management, prompt construction,
and graceful API exception handling.
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, MAX_HISTORY, OPENAI_API_KEY
from prompts import RAG_USER_PROMPT, STANDALONE_QUERY_PROMPT, SYSTEM_PROMPT
from utils import logger


class Chatbot:
    """Manages chat session history, prompt construction, and LLM communication."""

    def __init__(self, api_key: str = OPENAI_API_KEY, model_name: str = LLM_MODEL):
        self.api_key = api_key.strip() if api_key else ""
        self.model_name = model_name
        self.history: list[dict[str, str]] = []
        self.last_sources: list[dict[str, Any]] = []
        self.llm: ChatOpenAI | None = None

        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the ChatOpenAI client if API key is provided."""
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=0.0,
                    api_key=self.api_key,
                )
                logger.info("Initialized ChatOpenAI with model: %s", self.model_name)
            except Exception as e:
                logger.warning("Could not initialize ChatOpenAI: %s", e)
                self.llm = None
        else:
            self.llm = None
            logger.info("No valid OpenAI API key provided for Chatbot LLM.")

    def update_api_key(self, api_key: str) -> None:
        """Update API key dynamically at runtime."""
        self.api_key = api_key.strip()
        self._init_llm()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the in-memory conversation history."""
        self.history.append({"role": role, "content": content})
        # Keep within MAX_HISTORY turns (each turn = user + assistant = 2 messages)
        max_messages = MAX_HISTORY * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear_history(self) -> None:
        """Reset the active conversation history."""
        self.history.clear()
        self.last_sources.clear()
        logger.info("Conversation history cleared.")

    def get_formatted_history(self) -> str:
        """Format history into a human-readable text block for prompt inclusion."""
        if not self.history:
            return ""
        lines = ["Conversation History:"]
        for msg in self.history[-6:]:  # Use last 3 turns for context
            speaker = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")
        lines.append("")
        return "\n".join(lines)

    def reformulate_query(self, question: str) -> str:
        """Generate a standalone search query if the question references prior conversation."""
        if not self.history or self.llm is None:
            return question

        # If question contains pronouns or referential phrases, query reformulation is beneficial
        referential_cues = [" it ", " its ", " this ", " that ", " they ", " these ", " those ", " also ", " main types", " difference"]
        lower_q = f" {question.lower()} "
        has_cue = any(cue in lower_q for cue in referential_cues)

        if not has_cue:
            return question

        try:
            history_str = self.get_formatted_history()
            prompt = STANDALONE_QUERY_PROMPT.format(chat_history=history_str, question=question)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            reformulated = response.content.strip()
            if reformulated and len(reformulated) < len(question) * 3:
                logger.info("Reformulated query '%s' -> '%s'", question, reformulated)
                return reformulated
        except Exception as e:
            logger.warning("Failed to reformulate query: %s. Using original question.", e)

        return question

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Generate a grounded response using retrieved context chunks and LLM.

        Args:
            question: The user's question.
            retrieved_chunks: Relevant document chunks retrieved from vector store.

        Returns:
            Tuple of (answer_text, sources_list).
        """
        if not retrieved_chunks:
            msg = "I could not find enough relevant information in the uploaded documents to answer this question."
            self.last_sources = []
            return msg, []

        self.last_sources = retrieved_chunks

        # Format context with chunk citations
        context_parts = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            fname = chunk.get("metadata", {}).get("file_name", "Unknown File")
            cid = chunk.get("metadata", {}).get("chunk_id", idx)
            page = chunk.get("metadata", {}).get("page", 1)
            page_info = f", Page {page}" if chunk.get("metadata", {}).get("file_type") == ".pdf" else ""
            context_parts.append(
                f"[Source {idx}: {fname}{page_info}, Chunk {cid}]\n{chunk['content']}"
            )
        formatted_context = "\n\n".join(context_parts)

        # Check LLM availability
        if self.llm is None:
            # Informative fallback response if no API key is provided
            answer = (
                "[Demo Mode - No OpenAI API Key Set]\n\n"
                "Retrieved relevant context chunks successfully from your documents:\n\n"
                f"{formatted_context}\n\n"
                "To generate full natural language AI answers, add your OPENAI_API_KEY in the .env file."
            )
            self.add_message("user", question)
            self.add_message("assistant", answer)
            return answer, retrieved_chunks

        history_text = self.get_formatted_history()
        user_prompt = RAG_USER_PROMPT.format(
            context=formatted_context,
            conversation_history=history_text,
            question=question,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            logger.info("Sending prompt to LLM (%s)...", self.model_name)
            response = self.llm.invoke(messages)
            answer_text = response.content.strip()

            if not answer_text:
                answer_text = "I could not generate an answer from the provided documents."

            # Update conversation history
            self.add_message("user", question)
            self.add_message("assistant", answer_text)

            return answer_text, retrieved_chunks

        except Exception as e:
            err_str = str(e)
            logger.error("LLM Generation error: %s", err_str)

            if "api_key" in err_str.lower() or "authentication" in err_str.lower():
                return (
                    "Authentication error: Invalid or missing OpenAI API key.\n"
                    "Please check your OPENAI_API_KEY in the .env file.",
                    [],
                )
            elif "rate_limit" in err_str.lower() or "429" in err_str:
                return (
                    "OpenAI API rate limit exceeded or quota exhausted.\n"
                    "Please check your account balance or try again in a few moments.",
                    [],
                )
            elif "connection" in err_str.lower() or "timeout" in err_str.lower():
                return (
                    "Unable to connect to the AI service.\n"
                    "Please check your internet connection and proxy settings.",
                    [],
                )
            else:
                return (
                    f"AI service error: {err_str.splitlines()[0]}\n"
                    "Please try again or inspect rag_chatbot.log for details.",
                    [],
                )
