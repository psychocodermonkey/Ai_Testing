"""
 Program: Run the downloaded model.
    Name: Andrew Dixon            File: diagnostics.py
    Date: 6 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

import torch


def checkGpu() -> None:
  isCudaAvailable = torch.cuda.is_available()
  print(f"CUDA available: {isCudaAvailable}")

  if isCudaAvailable:
    deviceIndex = torch.cuda.current_device()
    deviceName = torch.cuda.get_device_name(deviceIndex)
    print(f"CUDA device index: {deviceIndex}")
    print(f"CUDA device name: {deviceName}")
