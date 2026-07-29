"""Docstring and comment embedding pipeline.

Embeds docstrings, comments, and README sections using sentence-transformers.
Each embedding links back to its parent code symbol for cross-reference.
"""

from collections.abc import Callable

from sentence_transformers import SentenceTransformer


class DocEmbedder:
    """Embeds natural language text (docstrings, comments) using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the doc embedder with a sentence-transformers model."""
        self.model = SentenceTransformer(model_name)

    def embed_batch(
        self,
        docs: list[dict[str, str]],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict]:
        """
        Embed a batch of docstrings.

        Args:
            docs (list[dict]): List of dictionaries with keys "text" and "symbol_id".
            batch_size (int, optional): Size of inference batches. Defaults to 32.
            progress_callback (Callable[[int, int], None], optional): Callback function
                receiving (processed_count, total_count).

        Returns:
            list[dict]: List of dictionaries containing "embedding" (numpy array) and "symbol_id".
        """
        if not docs:
            return []

        # Filter out empty texts and keep symbol links
        valid_docs = [doc for doc in docs if doc.get("text") and doc["text"].strip()]

        total_count = len(valid_docs)
        if total_count == 0:
            return []

        results = []
        processed_count = 0

        # Process in batches
        for i in range(0, total_count, batch_size):
            batch_docs = valid_docs[i : i + batch_size]
            batch_texts = [doc["text"] for doc in batch_docs]

            # encode returns a numpy array or torch tensor (numpy by default for sentence_transformers)
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                show_progress_bar=False,
            )

            # Link embeddings back to their symbol IDs
            for doc, emb in zip(batch_docs, batch_embeddings, strict=True):
                results.append(
                    {
                        "embedding": emb,
                        "symbol_id": doc.get("symbol_id"),
                    }
                )

            processed_count += len(batch_docs)
            if progress_callback:
                progress_callback(processed_count, total_count)

        return results
