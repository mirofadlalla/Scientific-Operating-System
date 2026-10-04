"""Concrete embedding providers."""

from .e5 import E5InstructEmbedding
from .fallback import SentenceTransformerEmbedding, make_sentence_transformer
from .huggingface import build_huggingface
from .jina import JinaEmbedding, build_jina
from .openai import build_openai

__all__ = [
    "E5InstructEmbedding",
    "JinaEmbedding",
    "SentenceTransformerEmbedding",
    "build_huggingface",
    "build_jina",
    "build_openai",
    "make_sentence_transformer",
]
