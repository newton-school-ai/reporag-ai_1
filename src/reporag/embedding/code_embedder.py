"""Code embedding pipeline.

Embeds code chunks using CodeBERT or UniXcoder. Produces 768-dim L2-normalized
vectors. Supports batch embedding with GPU acceleration and CPU fallback.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class CodeEmbedder:
    """Code embedding class using CodeBERT for semantic code representation."""

    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize the embedder with a pre-trained model."""
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        self.cache = {}

    def embed_batch(self, code_strings: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a batch of code strings.

        Args:
            code_strings (list[str]): List of code snippets.
            batch_size (int, optional): Size of the batches for inference. Defaults to 32.

        Returns:
            np.ndarray: Matrix of embeddings of shape (N, 768).
        """
        if not code_strings:
            return np.array([])

        embeddings = []
        uncached_indices = []
        uncached_strings = []

        # Check cache
        for i, code in enumerate(code_strings):
            if code in self.cache:
                embeddings.append(self.cache[code])
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_strings.append(code)

        # Process uncached strings in batches
        if uncached_strings:
            with torch.no_grad():
                for i in range(0, len(uncached_strings), batch_size):
                    batch_texts = uncached_strings[i : i + batch_size]

                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    ).to(self.device)

                    outputs = self.model(**inputs)

                    # Use CLS token representation (first token)
                    token_embeddings = outputs.last_hidden_state
                    cls_embeddings = token_embeddings[:, 0, :]

                    # L2-normalize
                    normalized_embeddings = torch.nn.functional.normalize(
                        cls_embeddings, p=2, dim=1
                    )
                    batch_embeddings = normalized_embeddings.cpu().numpy()

                    # Update cache and embeddings list
                    for j, emb in enumerate(batch_embeddings):
                        idx = uncached_indices[i + j]
                        text = batch_texts[j]
                        embeddings[idx] = emb
                        self.cache[text] = emb

        return np.vstack(embeddings)
