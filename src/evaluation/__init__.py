"""Evaluation modules for comparing base and fine-tuned models."""

from .evaluate_finetuned import (
    load_finetuned_model,
    evaluate_model_on_dataset,
    calculate_average_improvements,
    display_comparison_report
)
from .evaluate_ollama import evaluate_ollama_model

__all__ = [
    'load_finetuned_model',
    'evaluate_model_on_dataset',
    'calculate_average_improvements',
    'display_comparison_report',
    'evaluate_ollama_model'
]
