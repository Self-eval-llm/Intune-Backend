"""Data generation modules for creating training datasets."""

from .teacher import LocalDatasetGenerator
from .student import OllamaInferencePipeline

__all__ = ['LocalDatasetGenerator', 'OllamaInferencePipeline']
