#! /usr/bin/env python3
'''
 Program: Example of generating an image from text.
    Name: Andrew Dixon            File: genImage.py
    Date: 7 Feb 2026
   Notes:

........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
'''

import sys
from pathlib import Path
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from Ubiquitous import MODEL_DIR, OUTPUT_DIR

XL = True


def main() -> None:
  if len(sys.argv) < 2:
    print('Usage: python generateImage.py <promptFile.txt>')
    sys.exit(1)

  promptFile = Path(sys.argv[1]).resolve()
  promptText = loadPrompt(promptFile)

  # Where the script lives — output image goes here
  timeStamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  outputFile = OUTPUT_DIR / f'IMG{timeStamp}.png'

  repo = {
    'sdModel': 'runwayml/stable-diffusion-v1-5',
    'xlModel': 'stabilityai/stable-diffusion-xl-base-1.0',
  }

  # Load pipeline from Hugging Face into local models directory
  if XL:
    pipe = StableDiffusionXLPipeline.from_pretrained(
      repo['xlModel'],
      cache_dir=MODEL_DIR,
      torch_dtype=torch.float16
    )

  else:
    pipe = StableDiffusionPipeline.from_pretrained(
      repo['sdModel'],
      cache_dir=MODEL_DIR,
      torch_dtype=torch.float32
    )

  # deviceType = 'cuda' if torch.cuda.is_available() else 'cpu'
  deviceType = 'mps' if torch.backends.mps.is_available() else 'cpu'
  pipe = pipe.to(deviceType)
  print(f'\n ---- Using device: {deviceType} -----\n')

  # Generate image
  image = pipe(promptText, num_inference_steps=30, guidance_scale=7.5).images[0]

  # Save output
  image.save(outputFile)

  print(f'\n ---- Image saved to: {outputFile} ----\n')


def loadPrompt(promptFile: Path) -> str:
  if not promptFile.exists():
    raise FileNotFoundError(f'Prompt file not found: {promptFile}')

  promptText = promptFile.read_text(encoding='utf-8').strip()

  if not promptText:
    raise ValueError('Prompt file is empty')

  return promptText


if __name__ == '__main__':
  main()
