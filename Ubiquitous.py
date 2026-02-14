#! /usr/bin/env python3
"""
 Program: Ubiquitous tools for scripts
    Name: Andrew Dixon            File: Ubiquitous.py
    Date: 7 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

import os
from pathlib import Path


def getModelDir() -> Path:
  """
  Build and return the directory where models will be stored.

  Generates a .gitignore file to retrain the directory and ignore anything put within.

  :return: Model working directory path
  :rtype: Path
  """

  # Build the path for where we want to store models.
  base: Path = Path(__file__).parent
  modelDir: Path = base / 'models'

  modelDir.mkdir(parents=True, exist_ok=True)

  # Enforce a gitignore file in the models directory.
  gitignore = '*\n!.gitignore'
  with (modelDir / '.gitignore').open('w') as f:
    f.write(gitignore)

  # Return the path after we have made sure it exists and enforced the gitignore file.
  return Path(modelDir)


def getOutputDir() -> Path:
  """
  Build and return the directory for output of the scripts.

  Generates a .gitignore file to retain the directory and ignore anything put within.

  :return: Direcotry used for easy collection of output from scripts.
  :rtype: Path
  """
  # Build the path for where to save output.
  base: Path = Path(__file__).parent
  outputDir: Path = base / 'output'

  outputDir.mkdir(parents=True, exist_ok=True)
  gitignore = '*\n!.gitignore'
  with (outputDir / '.gitignore').open('w') as f:
    f.write(gitignore)

  # Return the path after we have made sure it exists and enforced the gitignore file.
  return Path(outputDir)


def getModel(repoID: str = None, fileName: str = None, override: bool = False) -> Path | None:
  """
  Download a model from hugging face repo and save it to a local directory.
  """
  from huggingface_hub import hf_hub_download

  if repoID is None or fileName is None:
    return None

  targetPath = MODEL_DIR / fileName

  if targetPath.exists() and not override:
    return targetPath

  if targetPath.exists() and override:
    targetPath.unlink()

  hfToken = hfLogin()

  modelPath = hf_hub_download(
    repo_id=repoID,
    filename=fileName,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
    token=hfToken,
  )

  return Path(modelPath)


def hfLogin() -> str | None:
  """
  Load HF_TOKEN (optionally from .env) and make it available for Hugging Face Hub.

  Returns the token string if present, else None.
  """
  envPath: Path = Path(__file__).parent / '.env'
  if envPath.exists():
    try:
      from dotenv import load_dotenv

      load_dotenv(dotenv_path=envPath)
    except Exception:
      # dotenv isn't required if HF_TOKEN is already in the environment
      pass

  token = os.getenv('HF_TOKEN')
  if not token:
    return None

  token: str = token.strip()

  # Cache token for huggingface_hub so downstream libraries can pick it up silently.
  try:
    from huggingface_hub import HfFolder

    HfFolder.save_token(token)
  except Exception:
    # Not fatal; we can still pass token directly.
    pass

  return token


def loadPrompt(promptFile: Path) -> dict[str, list[str]]:
  """
  Parse the prompt file for prompt and exclusion prompting headers.

  :param promptFile: External prompt file.
  :type promptFile: Path
  :return: Dictionary of the prompt and what will be use for exclusion / negatitive prompt.
  :rtype: dict
  """

  promptData = {
    'prompt': [],
    'exclude': [],
    'refine prompt': [],
    'refine exclude': []
  }

  if not promptFile.exists():
    raise FileNotFoundError(f'Prompt file not found: {promptFile}')

  currentSection = None

  # Get the text body so we can look for our appropriat esections.
  text = promptFile.read_text(encoding='utf-8').strip()
  for line in text.splitlines():
    line = line.strip()

    # Skip common comment lines.
    if not line or line.startswith('#') or line.startswith('//') or line.startswith(';'):
      continue

    # Set what the current section is and then skip to the next line.
    if line.lower() == '[prompt]':
      currentSection = 'prompt'
      continue

    elif line.lower() == '[exclude]':
      currentSection = 'exclude'
      continue

    elif line.lower() == '[refine prompt]':
      currentSection = 'refine prompt'
      continue

    elif line.lower() == '[refine exclude]':
      currentSection = 'refine exclude'
      continue

    # Append the line to the appropriate section.
    if currentSection:
      promptData[currentSection].append(line.strip())

  # validate that we have prompt text. If exclude is missing we don't care we just won't use it.
  if not promptData['prompt']:
    raise ValueError('Prompt file contains no [prompt] section or is empty')

  return promptData


def genPromptString(lines: list[str]) -> str:
  """Return a normalized comma separated line to use as a prompt."""
  return ', '.join([line.strip() for line in lines if line.strip()])


# Globals
MODEL_DIR = getModelDir()
OUTPUT_DIR = getOutputDir()
