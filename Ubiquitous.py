#! /usr/bin/env python3
'''
 Program: Ubiquitous tools for scripts
    Name: Andrew Dixon            File: Ubiquitous.py
    Date: 7 Feb 2026
   Notes:

........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
'''

import os
# import contextlib
from pathlib import Path


def getModelDir() -> Path:
  """
  Build and return the directory where models will be stored.

  Generates a .gitignore file to retrain the directory and ignore anything put within.

  :return: Model working directory path
  :rtype: Path
  """

  # Build the path for where we want to store models.
  base = Path(__file__).parent
  modelDir = base / 'models'

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
  base = Path(__file__).parent
  outputDir = base / 'output'

  outputDir.mkdir(parents=True, exist_ok=True)
  gitignore = '*\n!.gitignore'
  with (outputDir / '.gitignore').open('w') as f:
    f.write(gitignore)

  # Return the path after we have made sure it exists and enforced the gitignore file.
  return Path(outputDir)


def getModel(repoID: str = None, fileName: str = None, override: bool = False) -> bool:
  """
  Download a model from hugging face repo and save it to a local directory.

  :param repoID: Repository ID of model to download from hugging face
  :type repoID: str
  :param fileName: Filename of model to download from hugging face repo.
  :type fileName: str
  :param override: Force delete and redownload of a model (true/false)
  :type override: bool
  :return: Path to downloaded / existing model or none if there was a problem.
  :rtype: Path | None
  """
  from huggingface_hub import hf_hub_download

  # Dump out if we didn't get the info we need to download the model.
  if repoID is None or fileName is None:
    return False

  # See if the model has already been downloaded, if so just return true and move on.
  if Path(MODEL_DIR / fileName).exists() and not override:
    return True

  # Delete the file if override is set and it exists.
  if Path(MODEL_DIR / fileName).exists() and override:
    Path(MODEL_DIR / fileName).unlink()

  # Download the model if it does not exist or we are overriding.
  modelPath = hf_hub_download(
    repo_id=repoID, filename=fileName, local_dir=MODEL_DIR, local_dir_use_symlinks=False
  )

  return modelPath is not None


def hfLogin() -> bool:
  """Log in to hugging face from .env token."""
  env = Path(__file__).parent / '.env'
  if env.exists():
    from dotenv import load_dotenv
    # from huggingface_hub import login
    # Bring in secrets from environment file
    load_dotenv()
    # login(token=os.getenv('HF_TOKEN'))
  return "HF_TOKEN" in os.environ


def loadPrompt(promptFile: Path) -> dict:
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
