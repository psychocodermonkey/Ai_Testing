"""
 Program: Model registry for text / instruct models (classification, summarization, extraction).
    Name: Andrew Dixon            File: textModels.py
   Notes:
  Config keys used by loadTextModel: repo or modelId, maxNewTokens, temperature, doSample, topP,
  and optionally forceDevice ('cuda'|'mps'|'cpu'), isVerbose.
  Optional fields for future use / documentation:
    topK (int): nucleus/top-k sampling size
    repetitionPenalty (float): discourage repetition
    stopSequences (list[str]): stop generation at these strings
    maxLength (int): max context length
    seed (int): random seed for reproducibility
"""

from __future__ import annotations

from typing import Any

# Registry of text model configs. Pass a key to getTextModel() to load via textRuntime.loadTextModel().
MODELS: dict[str, dict[str, Any]] = {
  "qwen-default": {
    "repo": "Qwen/Qwen2.5-0.5B-Instruct",
    "maxNewTokens": 1024,
    "temperature": 0.3,
    "doSample": True,
    "topP": 0.9,
  },
  "qwen-1.5b": {
    "repo": "Qwen/Qwen2.5-1.5B-Instruct",
    "maxNewTokens": 1024,
    "temperature": 0.3,
    "doSample": True,
    "topP": 0.9,
  },
}


def getTextModels() -> dict[str, dict[str, Any]]:
  """Return the text model registry (read-only)."""
  return MODELS


def getTextModel(modelKey: str) -> "TextModelWrapper":
  """Load and return a TextModelWrapper for the given model key. Raises if key is unknown."""
  from .textRuntime import TextModelWrapper, loadTextModel

  config = MODELS.get(modelKey)
  if not config:
    raise ValueError(f"Unknown modelKey: {modelKey!r}. Choose from {list(MODELS)}")
  return loadTextModel(config)
