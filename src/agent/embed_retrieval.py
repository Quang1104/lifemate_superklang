"""Sentence embeddings for choosing retrieved examples (LIFEMATE_EMBED_RETRIEVAL=1).

Word overlap misses requests that mean the same with other words (dialect, ASR spelling). With the flag on,
fewshot.Examples ranks the usable pool rows by cosine similarity of multilingual-e5-small embeddings instead; the
prompt format, filters, budget and K are unchanged, so an already trained planner can be evaluated with it.
The model is loaded from a local directory only (LIFEMATE_EMBEDDER_DIR, default /models/embedder).
"""

import os
from pathlib import Path

_PREFIX = "query: "  # multilingual-e5 was trained with this prefix for symmetric text similarity
_loaded = None


class Embedder:
    """Mean-pooled, L2-normalised sentence vectors from a local encoder checkpoint."""

    def __init__(self, directory: Path, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not Path(directory).is_dir():
            raise ValueError(f"Embedder must be supplied as a local directory: {directory}")
        local = {"local_files_only": True, "trust_remote_code": False}
        self._torch, self._device = torch, device
        self._tokenizer = AutoTokenizer.from_pretrained(directory, **local)
        self._model = AutoModel.from_pretrained(directory, **local).to(device).eval()
        self.parameter_count = sum(parameter.numel() for parameter in self._model.parameters())

    def __call__(self, texts: list[str]):
        """An (n, d) float32 numpy array of unit vectors, one row per text."""
        torch, rows = self._torch, []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                batch = self._tokenizer([_PREFIX + text for text in texts[start:start + 64]], padding=True,
                                        truncation=True, max_length=128, return_tensors="pt").to(self._device)
                hidden = self._model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                rows.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(rows).numpy()


def embedder(device: str | None = None) -> Embedder:
    """The process-wide embedder, loaded once (on `device`, else CUDA when available)."""
    global _loaded
    if _loaded is None:
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        _loaded = Embedder(Path(os.environ.get("LIFEMATE_EMBEDDER_DIR") or "/models/embedder"), device)
    return _loaded


def encode(texts: list[str]):
    return embedder()(texts)


def warm(device: str) -> int:
    """Load the embedder and embed the native example pool during model loading; returns its parameter count."""
    from .fewshot import native_examples

    model = embedder(device)
    native_examples().embed_pool()
    return model.parameter_count
