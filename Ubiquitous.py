#! /usr/bin/env python3
'''
 Program: Ubiquitous tools for scripts
    Name: Andrew Dixon            File: Ubiquitous.py
    Date: 7 Feb 2026
   Notes:

........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
'''

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
  :return: Was script successful?
  :rtype: bool
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


# Globals
MODEL_DIR = getModelDir()
OUTPUT_DIR = getOutputDir()
