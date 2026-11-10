"""
Base classes for LLM and Embedding providers.

This module defines abstract base classes that all providers must implement,
ensuring consistent interfaces across different service providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model providers.
    
    All LLM providers must implement the methods defined in this class to ensure
    consistent behavior across different models and services.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LLM provider with configuration parameters."""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """
        Generate a response from the LLM given input messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated response string
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod 
    def extract_keywords(self, text: str, max_keywords: int = 8) -> List[str]:
        """
        Extract key concepts/keywords from text for AveFact strategy.
        
        This method is crucial for the AveFact retrieval strategy, which uses
        keyword-based vector averaging for memory retrieval.
        
        Args:
            text: Input text to extract keywords from
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of extracted keywords (up to max_keywords)
            
        Raises:
            Exception: If keyword extraction fails
        """
        pass
    
    def generate_script(self, trajectory: str, images: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a high-level script from a task trajectory.

        This is used by the Script and Proceduralization build strategies
        to create abstract representations of successful task completions.

        Args:
            trajectory: Detailed step-by-step trajectory
            images: Optional list of image dicts with 'url' key for multimodal input

        Returns:
            High-level script representation
        """
        prompt = f"""
        Analyze the following task trajectory and generate a concise, high-level script
        that captures the key steps and decision points:

        {trajectory}

        Generate a script with 3-5 high-level steps that could guide similar tasks.
        """
        if images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img["url"]}})
            messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]
        return self.generate(messages, temperature=0)


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding providers.
    
    Embedding providers are used to convert text into vector representations
    for similarity search and retrieval operations.
    """
    
    def __init__(self, max_text_len: int = 8196, max_token_len: int = 0, **kwargs: Any) -> None:
        """
        Initialize the embedding provider with configuration parameters.

        Args:
            max_text_len: Maximum characters allowed per query before chunking.
                Longer queries will be split into fixed-size chunks and the
                resulting embeddings will be averaged. Set to 0 or a negative
                value to disable chunking.
            max_token_len: Maximum tokens allowed per query. When > 0, texts
                exceeding this limit are truncated to max_token_len tokens
                (using tiktoken cl100k_base). Applied before character chunking.
        """
        self.max_text_len = max_text_len
        self.max_token_len = max_token_len
        self._tiktoken_enc = None
        if max_token_len and max_token_len > 0:
            try:
                import tiktoken
                self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (one per input text)
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector for the input text
        """
        return self.embed([text])[0]

    def _truncate_by_tokens(self, text: str) -> str:
        """Truncate text to max_token_len tokens if encoder is available."""
        if not self._tiktoken_enc or not self.max_token_len or self.max_token_len <= 0:
            return text
        tokens = self._tiktoken_enc.encode(text)
        if len(tokens) <= self.max_token_len:
            return text
        return self._tiktoken_enc.decode(tokens[: self.max_token_len])

    # ---- chunking helpers -------------------------------------------------
    def _chunk_texts(self, texts: List[str]) -> tuple[List[str], List[int]]:
        """
        Split texts into chunks according to max_text_len.
        If max_token_len is set, truncate each text by tokens first.

        Returns:
            A tuple of (chunked_texts, chunk_counts_per_text)
        """
        # Token-level truncation first
        if self.max_token_len and self.max_token_len > 0:
            texts = [self._truncate_by_tokens(t) for t in texts]

        if self.max_text_len and self.max_text_len > 0:
            chunk_size = self.max_text_len
        else:
            chunk_size = None

        chunked: List[str] = []
        counts: List[int] = []
        for text in texts:
            if chunk_size and len(text) > chunk_size:
                pieces = [
                    text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
                ]
            else:
                pieces = [text]
            chunked.extend(pieces)
            counts.append(len(pieces))
        return chunked, counts

    @staticmethod
    def _merge_chunk_embeddings(
        chunk_embeddings: List[List[float]], counts: List[int]
    ) -> List[List[float]]:
        """
        Merge chunk-level embeddings back to per-text embeddings by averaging.
        """
        merged: List[List[float]] = []
        idx = 0
        for count in counts:
            group = chunk_embeddings[idx : idx + count]
            idx += count
            if not group:
                merged.append([])
                continue
            if count == 1:
                merged.append(group[0])
            else:
                merged.append(BaseEmbedder._average_vectors(group))
        return merged

    @staticmethod
    def _average_vectors(vectors: List[List[float]]) -> List[float]:
        """Compute element-wise average of vectors without extra dependencies."""
        if not vectors:
            return []
        length = len(vectors[0])
        sums = [0.0] * length
        for vec in vectors:
            for i in range(length):
                sums[i] += float(vec[i])
        return [s / len(vectors) for s in sums]


class ProviderError(Exception):
    """Base exception class for provider-related errors."""
    pass


class LLMError(ProviderError):
    """Exception raised for LLM-specific errors."""
    pass


class EmbedderError(ProviderError):
    """Exception raised for embedding-specific errors."""
    pass


# ---------------------------------------------------------------------------
# Shared retry configuration for API providers
# ---------------------------------------------------------------------------

API_MAX_RETRIES = 5
API_RETRY_DELAYS = (10, 60, 180, 300, 600)  # Fixed incremental delays per attempt (seconds)

# Keep legacy names for any code that still references them (e.g., tenacity decorators)
API_RETRY_BASE_DELAY = API_RETRY_DELAYS[0]
API_RETRY_MAX_DELAY = API_RETRY_DELAYS[-1]


def get_retry_delay(attempt: int) -> float:
    """Return the retry delay for the given 1-based attempt number."""
    idx = min(attempt - 1, len(API_RETRY_DELAYS) - 1)
    return float(API_RETRY_DELAYS[idx])

_RETRYABLE_400_CODES = {"upstream_error", "unknown_model"}
_NON_RETRYABLE_400_CODES = {"content_filter", "content_management_policy"}


def is_retryable_api_error(error: Exception) -> bool:
    """Classify whether an API error is transient and worth retrying."""
    error_str = str(error).lower()
    status = None
    try:
        status = getattr(getattr(error, "response", None), "status_code", None)
    except Exception:
        pass

    if status is not None and status >= 500:
        return True
    if status == 429:
        return True
    if status == 400:
        if any(code in error_str for code in _NON_RETRYABLE_400_CODES):
            return False
        if any(code in error_str for code in _RETRYABLE_400_CODES):
            return True
        return False
    if status == 403:
        return False

    error_type = type(error).__name__
    if error_type in ("APIConnectionError", "APITimeoutError", "InternalServerError", "RemoteProtocolError"):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    if "incomplete chunked read" in error_str or "peer closed connection" in error_str:
        return True
    return False
