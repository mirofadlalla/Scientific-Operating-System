"""Compatibility shims for llama-index / Weaviate version skew."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_weaviate_compat() -> None:
    """Restore get_doc_id() on llama-index nodes.

    llama-index-core >= 0.11 removed TextNode.get_doc_id(), but older
    versions of llama-index-vector-stores-weaviate still call it. Idempotent.
    Shim في البرمجة يعني طبقة صغيرة بتحطها بين حاجتين عشان تخليهم يشتغلوا مع بعض رغم إنهم مش متوافقين بشكل مباشر.
    """
    try:
        from llama_index.core.schema import BaseNode, TextNode

        for cls in (TextNode, BaseNode):
            if not hasattr(cls, "get_doc_id"):
                cls.get_doc_id = lambda self: self.ref_doc_id or self.node_id  # type: ignore[attr-defined]
                logger.debug("[Indexer] Patched %s.get_doc_id() for Weaviate compatibility.", cls.__name__)
    except Exception:
        logger.debug("[Indexer] Could not apply Weaviate compat shim.", exc_info=True)
