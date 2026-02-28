"""
 Program: Run the downloaded model.
    Name: Andrew Dixon            File: prompts.py
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


def loadPrompt(promptFile: Path) -> dict[str, list[str]]:
  promptData: dict[str, list[str]] = {
    "prompt": [],
    "exclude": [],
    "refine prompt": [],
    "refine exclude": [],
  }

  if not promptFile.exists():
    raise FileNotFoundError(f"Prompt file not found: {promptFile}")

  currentSection: str | None = None
  text = promptFile.read_text(encoding="utf-8")

  for line in text.splitlines():
    cleanLine = line.strip()

    if not cleanLine:
      continue
    if cleanLine.startswith("#") or cleanLine.startswith("//") or cleanLine.startswith(";"):
      continue

    sectionName = cleanLine.lower()
    if sectionName == "[prompt]":
      currentSection = "prompt"
      continue
    if sectionName == "[exclude]":
      currentSection = "exclude"
      continue
    if sectionName == "[refine prompt]":
      currentSection = "refine prompt"
      continue
    if sectionName == "[refine exclude]":
      currentSection = "refine exclude"
      continue

    if currentSection:
      promptData[currentSection].append(cleanLine)

  if not promptData["prompt"]:
    raise ValueError("Prompt file contains no [prompt] section or is empty")

  return promptData


def genPromptString(lines: list[str]) -> str:
  return ", ".join([line.strip() for line in lines if line.strip()])
