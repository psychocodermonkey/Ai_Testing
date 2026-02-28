"""
 Program: Run the downloaded model.
    Name: Andrew Dixon            File: hfAssets.py
    Date: 6 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from .hfAuth import hfLogin
from .paths import MODEL_DIR


def getModel(repoId: str, fileName: str, override: bool = False) -> Path | None:
  if not repoId or not fileName:
    return None

  targetPath = MODEL_DIR / fileName

  if targetPath.exists() and not override:
    return targetPath

  if targetPath.exists() and override:
    targetPath.unlink()

  token = hfLogin()

  modelPath = hf_hub_download(
    repo_id=repoId,
    filename=fileName,
    local_dir=MODEL_DIR,
    token=token,
  )

  return Path(modelPath)
