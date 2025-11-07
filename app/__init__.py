"""
Gemma Model API Package
"""
from app.generate import (
    start_ollama_service,
    load_model,
    generate_response,
    get_model_status,
    MODEL_NAME,
    OLLAMA_BASE_URL
)

__all__ = [
    "start_ollama_service",
    "load_model",
    "generate_response",
    "get_model_status",
    "MODEL_NAME",
    "OLLAMA_BASE_URL"
]
