"""
 Program: Quiet/noisy runtime configuration.
    Name: Andrew Dixon            File: quiet.py
    Date: 28 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

import logging
import warnings


def configureQuietMode(isVerbose: bool = False) -> None:
  if isVerbose:
    return

  warnings.filterwarnings("ignore", category=UserWarning)
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings(
    "ignore",
    message=".*upcast_vae.*deprecated.*",
    category=FutureWarning,
  )

  logging.getLogger("transformers").setLevel(logging.ERROR)
  logging.getLogger("diffusers").setLevel(logging.ERROR)
  logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
  logging.getLogger("httpx").setLevel(logging.WARNING)
  logging.getLogger("httpcore").setLevel(logging.WARNING)
  logging.getLogger("urllib3").setLevel(logging.WARNING)

  try:
    from transformers.utils import logging as hfLogging

    hfLogging.set_verbosity_error()
  except Exception:
    pass

  try:
    from diffusers.utils import logging as diffusersLogging

    diffusersLogging.set_verbosity_error()
    diffusersLogging.disable_progress_bar()
  except Exception:
    pass
