#! /usr/bin/env python3

"""
 Program: A diffusers-only image generation shell with model configs centralized in MODELS.
    Name: Andrew Dixon            File: genImage.py
    Date: 13 Feb 2026
   Notes:

    Local Diffusers Image Generation Shell
    =======================================

    This script is a local, dict-driven image generation framework built on Hugging Face Diffusers.
    It is designed for rapid experimentation with multiple text-to-image and image-to-image models
    while remaining hardware-aware, disk-aware, and easy to extend.

    Core design goals:
    - One script, many models: all model behavior is defined in a single MODELS dictionary
        (pipelines, repos, parameters, constraints).
    - Explicit control over prompt stages:
        • Stage 1: text → image (optional if input image is supplied)
        • Stage 2: image → image refinement (optional, model-dependent)
    - Models may share or mix pipelines (e.g. PixArt txt2img + SDXL refiner).
    - Long-prompt support where available (beyond CLIP's 77-token limit).
    - Guardrails are model-driven, not hardcoded (no forced safety logic).
    - Designed for Apple Silicon (MPS) but portable to CUDA/CPU systems.

    Key features:
    - Dict-driven model registry:
        • repo / imgRepo
        • txtPipeline / imgPipeline
        • dtype, CPU offload control
        • per-stage inference settings
        • prompt handling rules (negative prompt modes, token limits)
    - Optional external image input (--input-image) to:
        • skip txt2img
        • or drive refinement with fallback prompt logic
    - Automatic handling of gated Hugging Face models.
    - Model disk-awareness:
        • can list configured models
        • can indicate whether required model artifacts are already downloaded
    - Clean output naming:
        <promptName>-YYYYMMDD-HHMMSS.png
    - Minimal CLI surface; tuning lives in the dict, not flags.

    Intended use:
    - Fast iteration on local image generation.
    - Comparing model behavior (quality, speed, bias, guardrails).
    - Exploring prompt structure and refinement strategies.
    - Experimenting with mixed-model pipelines without rewriting code.

    This script is intentionally extensible:
    To add a model, import its pipeline(s), add a dict entry, and run.

    The architecture favors clarity, experimentation, and control over maximum automation.

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

import sys
import torch
import logging
import warnings
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from ubiquitous import MODEL_DIR, OUTPUT_DIR
from ubiquitous import generateImageFromPromptData, getImageModels, hfLogin, loadPrompt


def main() -> None:
  """Main"""
  if len(sys.argv) < 2:
    printUsageAndExit()

  args: Args = parseArgs(sys.argv[1:])

  # Set baseline logging (quiet mode handled below)
  logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
  configureQuietMode(isVerbose=args.isVerbose)

  # Import noisy libs after quiet-mode config
  from diffusers.utils import logging as diffusersLogging
  from transformers.utils import logging as hfLogging
  from huggingface_hub.errors import GatedRepoError

  if not args.isVerbose:
    hfLogging.set_verbosity_error()
    diffusersLogging.set_verbosity_error()
    diffusersLogging.disable_progress_bar()

  hfToken: str | None = hfLogin()
  print(f'[+] HF Login Success: {bool(hfToken)}')

  # Model registry: everything tweakable lives here.
  # Add new models by adding new keys; select with --model <name>.
  models: dict[str, dict[str, Any]] = getImageModels()

  # Print out the models thi is configured to utilize. (Dump MODELS dict)
  if args.listModels:
    printUsageAndExit(exitCode=0, models=models)

  # Ensure the model is in the dict.
  if args.modelName not in models:
    print(f"[!] Unknown model '{args.modelName}'. Available models:")

    for keyName in sorted(models.keys()):
      extra = ''

      if models[keyName].get('disabledReason'):
        extra = f' (disabled: {models[keyName]["disabledReason"]})'

      print(f'  - {keyName}{extra}')

    sys.exit(2)

  cfg: dict[str, Any] = models[args.modelName]
  if cfg.get('disabledReason'):
    print(f"[!] Model '{args.modelName}' is currently disabled: {cfg['disabledReason']}")
    sys.exit(2)

  # Validate that the prompt file exists.
  promptFile: Path = args.promptFile.resolve()
  if not promptFile.exists():
    print(f'[!] Prompt file not found: {promptFile}')
    sys.exit(2)

  promptData: dict[str, list[str]] = loadPrompt(promptFile)

  inputImage = None
  if args.inputImage is not None:
    inputPath: Path = args.inputImage.resolve()
    if not inputPath.exists():
      print(f'[!] Input image not found: {inputPath}')
      sys.exit(2)

    try:
      from PIL import Image

      inputImage: Image.Image = Image.open(inputPath).convert('RGB')
      print(f'[+] Using input image: {inputPath}')

    except Exception as ex:
      print(f'[!] Failed to load input image: {ex}')
      sys.exit(2)

  timeStamp: str = datetime.now().strftime('%Y%m%d-%H%M%S')
  outputFile: Path = OUTPUT_DIR / f'{promptFile.stem}-{timeStamp}.png'

  deviceType: Literal["mps", "cpu"] = 'mps' if torch.backends.mps.is_available() else 'cpu'
  print(f'\n[+] Using device: {deviceType}')
  print(f'[+] Using model: {args.modelName}')
  print(f'[+] Output file: {outputFile}\n')

  txtPipelineClass = cfg['txtPipeline']
  if txtPipelineClass is None:
    print(f"[!] Model '{args.modelName}' has no txtPipeline configured.")
    sys.exit(2)

  try:
    generateImageFromPromptData(
      modelCfg=cfg,
      promptData=promptData,
      outputFile=outputFile,
      inputImage=inputImage,
      hfToken=hfToken,
    )
  except GatedRepoError as ex:
    gatedRepoId: str = getattr(ex, 'repoId', cfg['repo'])
    print(f'[!] Access denied for gated model: {gatedRepoId}')
    print('[!] Visit the model page, request/accept access, then rerun.')
    sys.exit(2)

  print(f'\n[+] Image saved to: {outputFile}\n')

# Dataclass for passing command line arguments around.
@dataclass(frozen=True, slots=True)
class Args:
  promptFile: Path
  modelName: str
  isVerbose: bool
  listModels: bool
  inputImage: Path | None

def parseArgs(rawArgs: list[str]) -> Args:
  # Positional: promptFile
  # Optional:
  #   --model <name>  (defaults to sd3.5-medium)
  #   --verbose
  if not rawArgs:
    printUsageAndExit()

  isVerbose: bool = '--verbose' in rawArgs
  listModels: bool = '--list-models' in rawArgs

  modelName = 'pixart-sigma-512'
  if '--model' in rawArgs:
    modelIndex: int = rawArgs.index('--model')

    try:
      modelName: str = rawArgs[modelIndex + 1].strip()

    except IndexError:
      print('[!] --model requires a value.')
      printUsageAndExit(exitCode=2)

  inputImage: Path | None = None
  if '--input-image' in rawArgs:
    imageIndex: int = rawArgs.index('--input-image')

    try:
      inputImage: Path = Path(rawArgs[imageIndex + 1]).expanduser()

    except IndexError:
      print('[!] --input-image requires a value.')
      printUsageAndExit(exitCode=2)

  promptPath = ''
  skipNext = False
  for item in rawArgs:
    if skipNext:
      skipNext = False
      continue

    if item in ('--model', '--input-image'):
      skipNext = True
      continue

    if item.startswith('--'):
      continue

    promptPath = item
    break

  if (not promptPath or promptPath is None) and not listModels:
    print('[!] Missing prompt file.')
    printUsageAndExit(exitCode=2)
  elif listModels:
    promptPath = 'empty'

  return Args(
    promptFile=Path(promptPath),
    modelName=modelName,
    isVerbose=isVerbose,
    listModels=listModels,
    inputImage=inputImage,
  )


def repoIdToCachePrefix(repoId: str) -> str:
  # HuggingFace cache folder naming: models--{org}--{repo}
  parts: list[str] = repoId.strip().split('/')
  if len(parts) != 2:
    return ''
  return f'models--{parts[0]}--{parts[1]}'


def isRepoCached(repoId: str, modelDir: Path) -> bool:
  prefix: str = repoIdToCachePrefix(repoId)
  if not prefix:
    return False

  # Most common: a directory with that exact prefix exists
  if (modelDir / prefix).exists():
    return True

  # Some setups nest differently; allow "starts with" match as fallback
  try:
    for item in modelDir.iterdir():
      if item.is_dir() and item.name.startswith(prefix):
        return True

  except FileNotFoundError:
    return False

  return False


def supportsColor() -> bool:
  return sys.stdout.isatty()


def color(text: str, code: str) -> str:
  if not supportsColor():
    return text
  return f'\033[{code}m{text}\033[0m'


def red(text: str) -> str:
  return color(text, '31')


def green(text: str) -> str:
  return color(text, '32')


def yellow(text: str) -> str:
  return color(text, '33')


def printUsageAndExit(exitCode: int = 1, models: dict[str, dict[str, Any]] | None = None,) -> None:
  print(
    'Usage:\n'
    '  To show models available:\n\n'
    '  \tpython genImage-diff.py --list-models\n'
    '\n'
    '  To generate or process images:\n\n'
    '  \tpython genImage-diff.py <promptFile.prompt> [--model <name>] [--verbose]\n'
    '  \tpython generateImage.py <promptFile.prompt> [--model <name>] [--input-image <img.png>] [--verbose]\n'
    '\n'
    'Examples:\n'
    '  python genImage-diff.py description.prompt\n'
    '  python genImage-diff.py description.prompt --model sd3.5-large\n'
    '  python genImage-diff.py description.prompt --model sd3.5-medium --verbose\n'
    '  python genImage-diff.py refine.prompt --input-image output/previous.png --model sd3.5-medium\n'
    '\n'
  )

  if models:
    print('Available models:')

    needsDownload: bool = False

    for modelName in sorted(models.keys()):
      cfg: dict[str, Any] = models[modelName]

      repoId = cfg.get('repo')
      imgRepoId = cfg.get('imgRepo')

      missing = False

      if repoId and not isRepoCached(repoId, MODEL_DIR):
        missing = True

      if imgRepoId and not isRepoCached(imgRepoId, MODEL_DIR):
        missing = True

      line = f'  - {modelName}'

      if cfg.get('disabledReason'):
        line += f' (disabled: {cfg["disabledReason"]})'

      if missing:
        print(yellow(line))
        needsDownload = True
      else:
        print(line)

    if needsDownload:
      print()
      print(yellow('One or more models above must be downloaded before use.\n'))

  sys.exit(exitCode)


def configureQuietMode(isVerbose: bool = False) -> None:
  # If verbose let everything come through.
  if isVerbose:
    return

  # Silence depreation warnings from Python
  warnings.filterwarnings(
    action='ignore',
    message='.*upcast_vae.*deprecated.*',
    category=FutureWarning,
  )

  # Ignore model warnings issuesd.
  warnings.filterwarnings('ignore', category=UserWarning)

  # Set error logging for all others to ERROR only
  logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
  logging.getLogger('transformers').setLevel(logging.ERROR)
  logging.getLogger('diffusers').setLevel(logging.ERROR)
  logging.getLogger('httpx').setLevel(logging.WARNING)
  logging.getLogger('httpcore').setLevel(logging.WARNING)
  logging.getLogger('hpack').setLevel(logging.WARNING)
  logging.getLogger('urllib3').setLevel(logging.WARNING)


if __name__ == '__main__':
  main()
