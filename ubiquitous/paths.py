"""
 Program: Run the downloaded model.
    Name: Andrew Dixon            File: paths.py
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
import platform
from pathlib import Path


_GITIGNORE_CONTENT = "*\n!.gitignore"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def getUserDataDir() -> Path:
  systemName = platform.system()

  if systemName == "Darwin":
    return Path.home() / "Library" / "Application Support" / "com.psychocodermonkey.dev"

  if systemName == "Windows":
    localAppData = os.getenv("LOCALAPPDATA")
    if localAppData:
      return Path(localAppData) / "com.psychocodermonkey.dev"
    return Path.home() / "AppData" / "Local" / "com.psychocodermonkey.dev"

  xdgDataHome = os.getenv("XDG_DATA_HOME")
  if xdgDataHome:
    return Path(xdgDataHome) / "com.psychocodermonkey.dev"
  return Path.home() / ".local" / "share" / "com.psychocodermonkey.dev"


def _enforceGitignore(dirPath: Path) -> None:
  gitignorePath = dirPath / ".gitignore"
  if gitignorePath.exists():
    existingContent = gitignorePath.read_text(encoding="utf-8")
    if existingContent == _GITIGNORE_CONTENT:
      return
  gitignorePath.write_text(_GITIGNORE_CONTENT, encoding="utf-8")


def _resolveDataDir() -> Path:
  envDataDir = os.getenv("UBIQUITOUS_DATA_DIR")
  if envDataDir:
    return Path(envDataDir).expanduser()
  return getUserDataDir()


def _resolveModelDir(dataDir: Path) -> Path:
  envModelDir = os.getenv("UBIQUITOUS_MODEL_DIR")
  if envModelDir:
    return Path(envModelDir).expanduser()
  return dataDir / "models"


def _resolveOutputDir() -> Path:
  envOutputDir = os.getenv("UBIQUITOUS_OUTPUT_DIR")
  if envOutputDir:
    return Path(envOutputDir).expanduser()
  return _REPO_ROOT / "output"


DATA_DIR = _resolveDataDir()
MODEL_DIR = _resolveModelDir(DATA_DIR)
OUTPUT_DIR = _resolveOutputDir()

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_enforceGitignore(MODEL_DIR)
_enforceGitignore(OUTPUT_DIR)


def getModelDir() -> Path:
  return MODEL_DIR


def getOutputDir() -> Path:
  return OUTPUT_DIR
