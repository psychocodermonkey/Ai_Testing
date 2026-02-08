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

XL = '--xl' in sys.argv
MODELS = {
  'sd' : {
    'repo' : 'runwayml/stable-diffusion-v1-5',
    'pipeline': StableDiffusionPipeline,
    'dtype' : torch.float16
  },
  'sd-xl' : {
    'repo' : 'stabilityai/stable-diffusion-xl-base-1.0',
    'pipeline' : StableDiffusionXLPipeline,
    'dtype' : torch.float32
  }
}

def main() -> None:
  """
  Main function to generate an image from a external prompting file.
  """

  if len(sys.argv) < 2:
    print('Usage: python generateImage.py <promptFile.txt>')
    sys.exit(1)

  promptFile = Path(sys.argv[1]).resolve()
  promptData = loadPrompt(promptFile)

  # Where the script lives — output image goes here
  timeStamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  outputFile = OUTPUT_DIR / f'IMG{timeStamp}.png'

  # Load the local model as defined in the configuration.
  # Simple example of pipeline call. Functions and names stored in dict to allow for easy switching.
  # What ends up being executed based on XL flag.
  # pipe = StableDiffusionXLPipeline.from_pretrained(
  #     'stabilityai/stable-diffusion-xl-base-1.0',
  #     cache_dir=MODEL_DIR,
  #     torch_dtype=torch.float16
  # )

  cfg = MODELS['sd-xl' if XL else 'sd']
  pipe = cfg['pipeline'].from_pretrained(
    cfg['repo'],
    cache_dir=MODEL_DIR,
    torch_dtype=cfg['dtype']
  )

  # deviceType = 'cuda' if torch.cuda.is_available() else 'cpu'
  deviceType = 'mps' if torch.backends.mps.is_available() else 'cpu'
  pipe = pipe.to(deviceType)
  print(f'\n ---- Using device: {deviceType} -----\n')

  # Generate image
  promptText = ', '.join(promptData['prompt'])
  negativePromptText = ', '.join(promptData['exclude'])
  pipeArgs = {
    'prompt': promptText,
    'num_inference_steps': 30,
    'guidance_scale' : 7.5
  }

  # Append negative prompt argument if exists.
  if negativePromptText:
    pipeArgs['negative_prompt'] = negativePromptText

  image = pipe(**pipeArgs).images[0]

  # Save output
  image.save(outputFile)

  print(f'\n ---- Image saved to: {outputFile} ----\n')


def loadPrompt(promptFile: Path) -> dict:
  """
  Parse the prompt file for prompt and exclusion prompting headers.

  :param promptFile: External prompt file.
  :type promptFile: Path
  :return: Dictionary of the prompt and what will be use for exclusion / negatitive prompt.
  :rtype: dict
  """

  promptText = {
    'prompt': [],
    'exclude': []
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

    # Append the line to the appropriate section.
    if currentSection:
      promptText[currentSection].append(line.strip())

  # validate that we have prompt text. If exclude is missing we don't care we just won't use it.
  if not promptText['prompt']:
    raise ValueError('Prompt file contains no [prompt] section or is empty')

  return promptText


if __name__ == '__main__':
  main()
