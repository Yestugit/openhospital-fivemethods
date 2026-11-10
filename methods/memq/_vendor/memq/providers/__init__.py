"""
Provider implementations for LLM and embedding services.

This package contains abstract base classes and concrete implementations
for various AI service providers used in the Memp system.
"""

from .base import BaseLLM, BaseEmbedder, ProviderError, LLMError, EmbedderError
from .llm import OpenAILLM, MockLLM
from .circuit_breaker import (
    CircuitBreakerState,
    LLMCircuitBreaker,
    LLMCircuitBreakerFatalError,
    build_shared_llm_circuit_breaker,
)
from .embedding import OpenAIEmbedder, LocalEmbedder, MockEmbedder, AverageEmbedder

__all__ = [
    # Base classes
    "BaseLLM",
    "BaseEmbedder", 
    "ProviderError",
    "LLMError",
    "EmbedderError",
    
    # LLM providers
    "OpenAILLM",
    "CircuitBreakerState",
    "LLMCircuitBreaker",
    "LLMCircuitBreakerFatalError",
    "build_shared_llm_circuit_breaker",
    "MockLLM",
    
    # Embedding providers
    "OpenAIEmbedder",
    "LocalEmbedder",
    "MockEmbedder",
    "AverageEmbedder",
]
