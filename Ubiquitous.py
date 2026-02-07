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

MODEL_DIR = getModelDir()


def getModel(repoID: str = None, fileName: str = None, override: bool = False) -> bool:

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
