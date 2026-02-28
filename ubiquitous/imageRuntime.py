"""
 Program: Runtime functions for image generation.
    Name: Andrew Dixon            File: imageRuntime.py
    Date: 28 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .paths import MODEL_DIR
from .prompts import genPromptString


def _attachRepoIdAndRaise(gatedError: Exception, repoId: str) -> None:
  setattr(gatedError, 'repoId', repoId)
  raise gatedError


def buildTxtPipeline(modelCfg: dict[str, Any], hfToken: str | None) -> Any:
  from huggingface_hub.errors import GatedRepoError

  txtPipelineClass = modelCfg['txtPipeline']
  txtPipe = None

  try:
    txtPipe = txtPipelineClass.from_pretrained(
      modelCfg['repo'],
      cache_dir=MODEL_DIR,
      torch_dtype=modelCfg['dtype'],
      token=hfToken,
    )
  except GatedRepoError as gatedError:
    _attachRepoIdAndRaise(gatedError, modelCfg['repo'])

  if modelCfg.get('useCpuOffload'):
    try:
      txtPipe.enable_model_cpu_offload()
      print('[+] Enabled model CPU offload')
    except Exception as ex:
      print(f'[!] Failed to enable CPU offload: {ex}')

  txtPipe.safety_checker = None
  txtPipe.set_progress_bar_config(disable=False)

  deviceType = 'mps' if torch.backends.mps.is_available() else 'cpu'
  if not modelCfg.get('useCpuOffload'):
    txtPipe = txtPipe.to(deviceType)

  return txtPipe


def buildImgPipeline(modelCfg: dict[str, Any], hfToken: str | None) -> Any | None:
  from huggingface_hub.errors import GatedRepoError

  imgPipelineClass = modelCfg.get('imgPipeline')
  imgPipe = None

  if imgPipelineClass is None:
    return None

  imgRepo = modelCfg.get('imgRepo') or modelCfg['repo']

  try:
    imgPipe = imgPipelineClass.from_pretrained(
      imgRepo,
      cache_dir=MODEL_DIR,
      torch_dtype=modelCfg['dtype'],
      token=hfToken,
    )
  except GatedRepoError as gatedError:
    _attachRepoIdAndRaise(gatedError, imgRepo)

  if modelCfg.get('useCpuOffload'):
    try:
      imgPipe.enable_model_cpu_offload()
      print('[+] Enabled img2img model CPU offload')
    except Exception as ex:
      print(f'[!] Failed to enable img2img CPU offload: {ex}')

  imgPipe.safety_checker = None
  imgPipe.set_progress_bar_config(disable=False)

  deviceType = 'mps' if torch.backends.mps.is_available() else 'cpu'
  if not modelCfg.get('useCpuOffload'):
    imgPipe = imgPipe.to(deviceType)

  return imgPipe


def runTxt2Img(
  txtPipe: Any,
  modelCfg: dict[str, Any],
  promptText: str,
  negativePromptText: str,
  inputImage: Any | None,
) -> Any:
  if inputImage is not None:
    print('[+] Skipping txt2img stage (input image provided).')
    return inputImage

  txtCfg = modelCfg['txt2img']
  txtArgs: dict[str, Any] = {
    'prompt': promptText,
    'num_inference_steps': txtCfg['numInferenceSteps'],
    'guidance_scale': txtCfg['guidanceScale'],
    'width': txtCfg['width'],
    'height': txtCfg['height'],
  }

  if 'maxSequenceLength' in txtCfg:
    txtArgs['max_sequence_length'] = txtCfg['maxSequenceLength']

  negativePromptMode = txtCfg.get('negativePromptMode', 'normal')
  if negativePromptMode == 'empty':
    txtArgs['negative_prompt'] = ''
  elif negativePromptMode == 'omit':
    pass
  else:
    if negativePromptText:
      txtArgs['negative_prompt'] = negativePromptText

  return txtPipe(**txtArgs).images[0]


def runImg2Img(
  imgPipe: Any | None,
  modelCfg: dict[str, Any],
  baseImage: Any,
  promptText: str,
  negativePromptText: str,
  refinePrompt: str,
  refineNegativePrompt: str,
  inputImageProvided: bool,
) -> Any | None:
  if inputImageProvided and not refinePrompt:
    refinePrompt = promptText

  if inputImageProvided and not refineNegativePrompt:
    refineNegativePrompt = negativePromptText

  if not refinePrompt:
    print('[+] No refine prompt found; skipping refine stage.')
    return None

  if imgPipe is None:
    print('[+] No img2img pipeline configured for this model; skipping refine stage.')
    return None

  print('[+] Refining image...')
  imgCfg = modelCfg['img2img']
  imgArgs: dict[str, Any] = {
    'prompt': refinePrompt,
    'image': baseImage,
    'strength': imgCfg['strength'],
    'num_inference_steps': imgCfg['numInferenceSteps'],
    'guidance_scale': imgCfg['guidanceScale'],
  }

  if 'maxSequenceLength' in imgCfg:
    imgArgs['max_sequence_length'] = imgCfg['maxSequenceLength']

  if refineNegativePrompt:
    imgArgs['negative_prompt'] = refineNegativePrompt
  elif negativePromptText:
    imgArgs['negative_prompt'] = negativePromptText

  return imgPipe(**imgArgs).images[0]


def generateImageFromPromptData(
  modelCfg: dict[str, Any],
  promptData: dict[str, list[str]],
  outputFile: Path,
  inputImage: Any | None,
  hfToken: str | None,
) -> Path:
  txtPipe = buildTxtPipeline(modelCfg=modelCfg, hfToken=hfToken)
  imgPipe = buildImgPipeline(modelCfg=modelCfg, hfToken=hfToken)

  promptText = genPromptString(promptData.get('prompt', []))
  negativePromptText = genPromptString(promptData.get('exclude', []))
  image = runTxt2Img(
    txtPipe=txtPipe,
    modelCfg=modelCfg,
    promptText=promptText,
    negativePromptText=negativePromptText,
    inputImage=inputImage,
  )

  refinePrompt = genPromptString(promptData.get('refine prompt', []))
  refineNegativePrompt = genPromptString(promptData.get('refine exclude', []))
  refinedImage = runImg2Img(
    imgPipe=imgPipe,
    modelCfg=modelCfg,
    baseImage=image,
    promptText=promptText,
    negativePromptText=negativePromptText,
    refinePrompt=refinePrompt,
    refineNegativePrompt=refineNegativePrompt,
    inputImageProvided=(inputImage is not None),
  )

  finalImage = refinedImage if refinedImage is not None else image
  finalImage.save(outputFile)
  return outputFile
