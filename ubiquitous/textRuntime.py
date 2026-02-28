"""
 Program: Run the downloaded model.
    Name: Andrew Dixon            File: textRuntime.py
    Date: 6 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hfAuth import hfLogin
from .paths import MODEL_DIR


def _isMpsAvailable() -> bool:
  if not hasattr(torch.backends, "mps"):
    return False
  return bool(torch.backends.mps.is_available())


def selectTorchDevice() -> tuple[str, torch.dtype]:
  if torch.cuda.is_available():
    return ("cuda", torch.float16)
  if _isMpsAvailable():
    return ("mps", torch.float16)
  return ("cpu", torch.float32)


def _selectForcedDevice(forceDevice: str) -> tuple[str, torch.dtype]:
  if forceDevice == "cuda":
    if not torch.cuda.is_available():
      raise ValueError("forceDevice='cuda' requested but CUDA is not available")
    return ("cuda", torch.float16)

  if forceDevice == "mps":
    if not _isMpsAvailable():
      raise ValueError("forceDevice='mps' requested but MPS is not available")
    return ("mps", torch.float16)

  if forceDevice == "cpu":
    return ("cpu", torch.float32)

  raise ValueError("forceDevice must be one of: 'cuda', 'mps', 'cpu'")


class TextModelWrapper:
  def __init__(
    self,
    model: Any,
    tokenizer: Any,
    device: str,
    defaultMaxNewTokens: int,
    defaultTemperature: float,
    defaultDoSample: bool,
    defaultTopP: float,
  ) -> None:
    self._model = model
    self._tokenizer = tokenizer
    self._device = device
    self._defaultMaxNewTokens = defaultMaxNewTokens
    self._defaultTemperature = defaultTemperature
    self._defaultDoSample = defaultDoSample
    self._defaultTopP = defaultTopP

  def generate(
    self,
    prompt: str,
    maxNewTokens: int | None = None,
    temperature: float | None = None,
    doSample: bool | None = None,
    topP: float | None = None,
  ) -> str:
    runMaxNewTokens = maxNewTokens if maxNewTokens is not None else self._defaultMaxNewTokens
    runTemperature = temperature if temperature is not None else self._defaultTemperature
    runDoSample = doSample if doSample is not None else self._defaultDoSample
    runTopP = topP if topP is not None else self._defaultTopP

    inputIds: Any
    attentionMask: Any
    if hasattr(self._tokenizer, "apply_chat_template"):
      try:
        chatTensor = self._tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=True,
          add_generation_prompt=True,
          return_tensors="pt",
        )
        inputIds = chatTensor.to(self._device)
        attentionMask = torch.ones_like(inputIds)
      except Exception:
        tokenized = self._tokenizer(prompt, return_tensors="pt")
        inputIds = tokenized["input_ids"].to(self._device)
        attentionMask = tokenized.get("attention_mask")
        if attentionMask is not None:
          attentionMask = attentionMask.to(self._device)
    else:
      tokenized = self._tokenizer(prompt, return_tensors="pt")
      inputIds = tokenized["input_ids"].to(self._device)
      attentionMask = tokenized.get("attention_mask")
      if attentionMask is not None:
        attentionMask = attentionMask.to(self._device)

    with torch.inference_mode():
      generatedIds = self._model.generate(
        input_ids=inputIds,
        attention_mask=attentionMask,
        max_new_tokens=runMaxNewTokens,
        temperature=runTemperature,
        do_sample=runDoSample,
        top_p=runTopP,
      )

    promptLength = inputIds.shape[-1]
    completionIds = generatedIds[0][promptLength:]
    return self._tokenizer.decode(completionIds, skip_special_tokens=True).strip()


def loadTextModel(modelConfig: dict[str, Any]) -> TextModelWrapper:
  modelId = modelConfig.get("repo") or modelConfig.get("modelId")
  if not modelId:
    raise ValueError("Model config must include 'repo' or 'modelId'")

  token = hfLogin()
  forceDevice = modelConfig.get("forceDevice")
  if forceDevice is None:
    (device, dtype) = selectTorchDevice()
  else:
    (device, dtype) = _selectForcedDevice(str(forceDevice))

  tokenizer = AutoTokenizer.from_pretrained(
    modelId,
    cache_dir=str(MODEL_DIR),
    token=token,
  )
  model = AutoModelForCausalLM.from_pretrained(
    modelId,
    cache_dir=str(MODEL_DIR),
    token=token,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
  )
  model = model.to(device)
  model.eval()

  return TextModelWrapper(
    model=model,
    tokenizer=tokenizer,
    device=device,
    defaultMaxNewTokens=int(modelConfig.get("maxNewTokens", 256)),
    defaultTemperature=float(modelConfig.get("temperature", 0.0)),
    defaultDoSample=bool(modelConfig.get("doSample", False)),
    defaultTopP=float(modelConfig.get("topP", 1.0)),
  )
