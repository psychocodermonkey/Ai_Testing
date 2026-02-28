"""
 Program: Model registry for image generation.
    Name: Andrew Dixon            File: imageModels.py
    Date: 28 Feb 2026
   Notes:

  Copyright (c) 2026 Andrew Dixon

  This file is part of AI_Testing.
  Licensed under the GNU Lesser General Public License v2.1.
  See the LICENSE file at the project root for details.
........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
"""

from __future__ import annotations

from typing import Any


def _buildModels() -> dict[str, dict[str, Any]]:
  import torch
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

  return {
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


_modelsCache: dict[str, dict[str, Any]] | None = None


def getImageModels() -> dict[str, dict[str, Any]]:
  global _modelsCache

  if _modelsCache is None:
    _modelsCache = _buildModels()

  return _modelsCache
