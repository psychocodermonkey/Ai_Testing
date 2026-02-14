#! /usr/bin/env python3

"""
 Program: A diffusers-only image generation shell with model configs centralized in MODELS.
    Name: Andrew Dixon            File: genImage.py
    Date: 13 Feb 2026
   Notes:

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
    # SD3.5 Medium
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
    # SD3.5 Large
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
        'negativePromptMode': 'empty',   # PixArt-Sigma expects ""
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
        'width': 1024,
        'height': 1024,
        'maxSequenceLength': 256,
        'negativePromptMode': 'empty',  # PixArt-Sigma expects ""
      },
      'img2img': {
        'strength': 0.20,
        'numInferenceSteps': 20,
        'guidanceScale': 5.5,
      },
    },
    # PixArt-α 512
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
        'width': 1024,
        'height': 1024,
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
    imgRepo = cfg.get('imgRepo', cfg['repo'])

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
  negativePromptMode = txtCfg.get('negativePromptMode', 'normal')

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

  image = txtPipe(**txtArgs).images[0]

  # Process second stage image to image prompt
  refinePrompt: str = genPromptString(promptData.get('refine prompt', []))
  refineNegativePrompt: str = genPromptString(promptData.get('refine exclude', []))

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

    refinedImage = imgPipe(**imgArgs).images[0]

  # Save final output only
  finalImage = refinedImage if refinedImage is not None else image
  finalImage.save(outputFile)

  print(f'\n[+] Image saved to: {outputFile}\n')

# Dataclass for passing command line arguments around.
@dataclass(frozen=True, slots=True)
class Args:
  promptFile: Path
  modelName: str
  isVerbose: bool
  listModels: bool


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

  # First non-flag argument is the prompt file
  promptPath = ''
  skipNext = False
  for idx, item in enumerate(rawArgs):
    if skipNext:
      skipNext = False
      continue
    if item == '--model':
      skipNext = True
      continue
    if item.startswith('--'):
      continue
    promptPath = item
    break

  if promptPath is None and not listModels:
    print('[!] Missing prompt file.')
    printUsageAndExit(exitCode=2)
  elif listModels:
    promptPath = 'empty'

  return Args(
    promptFile=Path(promptPath),
    modelName=modelName,
    isVerbose=isVerbose,
    listModels=listModels,
  )


def printUsageAndExit(exitCode: int = 1, models: dict[str, dict[str, Any]] | None = None,) -> None:
  print(
    'Usage:\n'
    '  python generateImage.py <promptFile.prompt> [--model <name>] [--verbose]\n\n'
    'Examples:\n'
    '  python generateImage.py description.prompt\n'
    '  python generateImage.py description.prompt --model sd3.5-large\n'
    '  python generateImage.py description.prompt --model sd3.5-medium --verbose\n'
  )

  if models:
    print('Available models:')
    for modelName in sorted(models.keys()):
      cfg = models[modelName]
      extra = ''
      if cfg.get('disabledReason'):
        extra = f' (disabled: {cfg["disabledReason"]})'
      print(f'  - {modelName}{extra}')

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
