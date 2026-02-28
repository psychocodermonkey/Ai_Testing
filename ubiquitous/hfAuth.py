"""
 Program: Run the downloaded model.
    Name: Andrew Dixon            File: hfAuth.py
    Date: 6 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

import os
from pathlib import Path


def hfLogin() -> str | None:
  envPath = Path(__file__).resolve().parent.parent / ".env"
  if envPath.exists():
    try:
      from dotenv import load_dotenv

      load_dotenv(dotenv_path=envPath)
    except Exception:
      pass

  token = os.getenv("HF_TOKEN")
  if not token:
    return None

  token = token.strip()
  if not token:
    return None

  try:
    from huggingface_hub import HfFolder

    HfFolder.save_token(token)
  except Exception:
    pass

  return token
