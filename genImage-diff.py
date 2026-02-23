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
from Ubiquitous import MODEL_DIR, OUTPUT_DIR
from Ubiquitous import genPromptString, hfLogin, loadPrompt


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
  from diffusers import (
    PixArtAlphaPipeline,
    PixArtSigmaPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
  )

  if not args.isVerbose:
    hfLogging.set_verbosity_error()
    diffusersLogging.set_verbosity_error()
    diffusersLogging.disable_progress_bar()

  hfToken: str | None = hfLogin()
  print(f'[+] HF Login Success: {bool(hfToken)}')

  # Model registry: everything tweakable lives here.
  # Add new models by adding new keys; select with --model <name>.
  MODELS: dict[str, dict[str, Any]] = {
    # # SD1.5
    'sd-1.5': {
      'repo': 'runwayml/stable-diffusion-v1-5',
      'txtPipeline': StableDiffusionPipeline,
      'imgPipeline': StableDiffusionImg2ImgPipeline,
      'dtype': torch.float16,
      'useCpuOffload': False,
      'txt2img': {
        'numInferenceSteps': 30,
        'guidanceScale': 8.0,
        'width': 512,
        'height': 512,
        'maxSequenceLength': 77,  # CLIP hard limit
        'negativePromptMode': 'normal',
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 30,
        'guidanceScale': 6.0,
        'maxSequenceLength': 77,
      },
    },
    # SDXL Base
    'sd-xl-base': {
      'repo': 'stabilityai/stable-diffusion-xl-base-1.0',
      'txtPipeline': StableDiffusionXLPipeline,
      'imgPipeline': StableDiffusionXLImg2ImgPipeline,
      'dtype': torch.float32,  # matches your original script
      'useCpuOffload': False,
      'txt2img': {
        'numInferenceSteps': 30,
        'guidanceScale': 8.0,
        'width': 1024,
        'height': 1024,
        'negativePromptMode': 'normal',
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 30,
        'guidanceScale': 6.0,
      },
    },
    # # SD3.5 Medium
    'sd3.5-medium': {
      'repo': 'stabilityai/stable-diffusion-3.5-medium',
      'txtPipeline': StableDiffusion3Pipeline,
      'imgPipeline': StableDiffusion3Img2ImgPipeline,
      'dtype': torch.float16,
      'useCpuOffload': True,  # safer default on MPS.
      'txt2img': {
        'numInferenceSteps': 30,
        'guidanceScale': 7.0,
        'width': 512,
        'height': 512,
        'maxSequenceLength': 128,  # Set higher to allow longer prompts.
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 30,
        'guidanceScale': 6.0,
        'maxSequenceLength': 128,
      },
    },
    # # SD3.5 Large
    'sd3.5-large': {
      'repo': 'stabilityai/stable-diffusion-3.5-large',
      'txtPipeline': StableDiffusion3Pipeline,
      'imgPipeline': StableDiffusion3Img2ImgPipeline,
      'dtype': torch.float16,
      'useCpuOffload': False,  # safer default on MPS.
      'txt2img': {
        'numInferenceSteps': 30,
        'guidanceScale': 7.0,
        'width': 512,
        'height': 512,
        'maxSequenceLength': 128,  # Set higher to allow longer prompts.
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 30,
        'guidanceScale': 6.0,
        'maxSequenceLength': 128,
      },
    },
    # PixArt-Σ XL model
    'pixart-sigma': {
      'repo': 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
      'txtPipeline': PixArtSigmaPipeline,
      'imgPipeline': StableDiffusionXLImg2ImgPipeline,
      'imgRepo': 'stabilityai/stable-diffusion-xl-refiner-1.0',
      'dtype': torch.float16,
      'useCpuOffload': True,
      'txt2img': {
        'numInferenceSteps': 30,
        'guidanceScale': 7.0,
        'width': 1024,
        'height': 1024,
        'maxSequenceLength': 256,
        'negativePromptMode': 'empty',  # PixArt-Sigma expects ""
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 30,
        'guidanceScale': 6.0,
      },
    },
    # PixArt-Σ 512
    'pixart-sigma-512': {
      'repo': 'Jingya/pixart_sigma_pipe_xl_2_512_ms',
      'txtPipeline': PixArtSigmaPipeline,
      'imgPipeline': StableDiffusionXLImg2ImgPipeline,
      'imgRepo': 'stabilityai/stable-diffusion-xl-refiner-1.0',
      'dtype': torch.float16,
      'useCpuOffload': True,
      'txt2img': {
        'numInferenceSteps': 25,
        'guidanceScale': 6.0,
        'width': 512,
        'height': 512,
        'maxSequenceLength': 256,
        'negativePromptMode': 'empty',  # PixArt-Sigma expects ""
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 20,
        'guidanceScale': 5.5,
      },
    },
    # # PixArt-α 512
    'pixart-alpha-512': {
      'repo': 'PixArt-alpha/PixArt-XL-2-512x512',
      'txtPipeline': PixArtAlphaPipeline,
      'imgPipeline': None,
      'imgRepo': '',
      'dtype': torch.float16,
      'useCpuOffload': True,
      'txt2img': {
        'numInferenceSteps': 25,
        'guidanceScale': 6.0,
        'width': 512,
        'height': 512,
        'maxSequenceLength': 256,
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 20,
        'guidanceScale': 5.5,
      },
    },
  }

  # Print out the models thi is configured to utilize. (Dump MODELS dict)
  if args.listModels:
    printUsageAndExit(exitCode=0, models=MODELS)

  # Ensure the model is in the dict.
  if args.modelName not in MODELS:
    print(f"[!] Unknown model '{args.modelName}'. Available models:")

    for keyName in sorted(MODELS.keys()):
      extra = ''

      if MODELS[keyName].get('disabledReason'):
        extra = f' (disabled: {MODELS[keyName]["disabledReason"]})'

      print(f'  - {keyName}{extra}')

    sys.exit(2)

  cfg: dict[str, Any] = MODELS[args.modelName]
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

  # Build text to image pipelines.
  txtPipelineClass = cfg['txtPipeline']
  if txtPipelineClass is None:
    print(f"[!] Model '{args.modelName}' has no txtPipeline configured.")
    sys.exit(2)

  try:
    txtPipe = txtPipelineClass.from_pretrained(
      cfg['repo'],
      cache_dir=MODEL_DIR,
      torch_dtype=cfg['dtype'],
      token=hfToken,
    )

  # If it is a gated repo and authentication fails report and exit.
  except GatedRepoError:
    print(f"[!] Access denied for gated model: {cfg['repo']}")
    print("[!] Visit the model page, request/accept access, then rerun.")
    sys.exit(2)

  # SD3 docs strongly suggest offloading for commodity hardware; keep it dict-controlled.
  # For MPS, offload can help memory but may trade speed and requires accelerate.
  if cfg.get('useCpuOffload'):
    try:
      txtPipe.enable_model_cpu_offload()
      print('[+] Enabled model CPU offload')

    except Exception as ex:
      print(f'[!] Failed to enable CPU offload: {ex}')

  txtPipe.safety_checker = None
  txtPipe.set_progress_bar_config(disable=False)

  # If we didn't enable CPU offload, move to device normally.
  if not cfg.get('useCpuOffload'):
    txtPipe = txtPipe.to(deviceType)

  # Create SD3 img2img pipeline from the same checkpoint (avoids AutoPipeline import issues)
  # Build img2img pipeline (dict-driven; may be None)
  imgPipelineClass = cfg.get('imgPipeline')
  imgPipe = None

  if imgPipelineClass is not None:
    # If using a model that doesn't have it's own image to image we can sub in another pipeline
    imgRepo: str = cfg.get('imgRepo', cfg['repo'])

    try:
      imgPipe = imgPipelineClass.from_pretrained(
        imgRepo,
        cache_dir=MODEL_DIR,
        torch_dtype=cfg['dtype'],
        token=hfToken,
      )

    # If we are a gated repo and authentication fails, report and exit.
    except GatedRepoError:
      print(f'[!] Access denied for gated model: {imgRepo}')
      print('[!] Visit the model page, request/accept access, then rerun.')
      sys.exit(2)

    # Set CPU offload if requested and available.
    if cfg.get('useCpuOffload'):
      try:
        imgPipe.enable_model_cpu_offload()
        print('[+] Enabled img2img model CPU offload')

      except Exception as ex:
        print(f'[!] Failed to enable img2img CPU offload: {ex}')

    # Disable safety and be sure we get progress bars for progress tracking
    imgPipe.safety_checker = None
    imgPipe.set_progress_bar_config(disable=False)

    if not cfg.get('useCpuOffload'):
      imgPipe = imgPipe.to(deviceType)

  # Stage 1: Text -> Image
  promptText: str = genPromptString(promptData.get('prompt', []))
  negativePromptText: str = genPromptString(promptData.get('exclude', []))

  # Set up configurations for text -> image generation.
  txtCfg: dict[str, Any] = cfg['txt2img']
  txtArgs: dict[str, Any] = {
    'prompt': promptText,
    'num_inference_steps': txtCfg['numInferenceSteps'],
    'guidance_scale': txtCfg['guidanceScale'],
    'width': txtCfg['width'],
    'height': txtCfg['height'],
  }

  # Set max token length if specified in the configruation.
  if 'maxSequenceLength' in txtCfg:
    txtArgs['max_sequence_length'] = txtCfg['maxSequenceLength']

  # Default negativePromptMode to normal if it is not in the dict.
  negativePromptMode: str = txtCfg.get('negativePromptMode', 'normal')

  # Empty value for the negative prompt form models where it is recommended.
  if negativePromptMode == 'empty':
    txtArgs['negative_prompt'] = ''

  # Leave the negativePrompt out alltogether for models that do not support it.
  elif negativePromptMode == 'omit':
    pass

  # Bring in the negative prompt if provided and allowed.
  else:
    if negativePromptText:
      txtArgs['negative_prompt'] = negativePromptText

  image: Image.Image | None = None
  if inputImage is not None:
    image: Image.Image = inputImage
    print('[+] Skipping txt2img stage (input image provided).')

  else:
    image: Image.Image = txtPipe(**txtArgs).images[0]

  # Process second stage image to image prompt
  refinePrompt: str = genPromptString(promptData.get('refine prompt', []))
  refineNegativePrompt: str = genPromptString(promptData.get('refine exclude', []))

  # If an external input image is supplied and no refine prompt exists,
  # fall back to the base prompt so img2img still runs.
  if inputImage is not None and not refinePrompt:
    refinePrompt: str = promptText

  # Same logic for negative prompts
  if inputImage is not None and not refineNegativePrompt:
    refineNegativePrompt: str = negativePromptText

  refinedImage = None
  if not refinePrompt:
    print('[+] No refine prompt found; skipping refine stage.')

  elif imgPipe is None:
    print('[+] No img2img pipeline configured for this model; skipping refine stage.')

  else:
    print('[+] Refining image...')
    imgCfg: dict[str, Any] = cfg['img2img']

    imgArgs: dict[str, Any] = {
      'prompt': refinePrompt,
      'image': image,
      'strength': imgCfg['strength'],
      'num_inference_steps': imgCfg['numInferenceSteps'],
      'guidance_scale': imgCfg['guidanceScale'],
    }

    if 'maxSequenceLength' in imgCfg:
      imgArgs['max_sequence_length'] = imgCfg['maxSequenceLength']

    # Prefer refine negatives, else fall back to base negatives
    if refineNegativePrompt:
      imgArgs['negative_prompt'] = refineNegativePrompt

    elif negativePromptText:
      imgArgs['negative_prompt'] = negativePromptText

    refinedImage: Image.Image = imgPipe(**imgArgs).images[0]

  # Save final output only
  finalImage: Image.Image = refinedImage if refinedImage is not None else image
  finalImage.save(outputFile)

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
