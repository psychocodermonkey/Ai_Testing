#! /usr/bin/env python3
'''
 Program: Example of generating an image from text.
    Name: Andrew Dixon            File: genImage.py
    Date: 7 Feb 2026
   Notes:

........1.........2.........3.........4.........5.........6.........7.........8.........9.........0.........1.........2.........3..
'''

from __future__ import annotations
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
import torch
from Ubiquitous import MODEL_DIR, OUTPUT_DIR
from Ubiquitous import loadPrompt, hfLogin, genPromptString
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from transformers import CLIPTokenizer

XL = '--xl' in sys.argv
MAX_TOKENS = 77

# Silence HTTP info messages
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('hpack').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

print(f'[+] HF Login Success: {hfLogin()}')

def main() -> None:
  """
  Main function to generate an image from a external prompting file.
  """

  if len(sys.argv) < 2:
    print('Usage: python generateImage.py <promptFile.txt>')
    sys.exit(1)

  # Set quiet mode for normal operation
  logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

  configureQuietMode(isQuiet=True)
  from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
  from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
  from diffusers.utils import logging as diffusersLogging
  from transformers import CLIPTokenizer
  from transformers.utils import logging as hfLogging

  hfLogging.set_verbosity_error()
  diffusersLogging.set_verbosity_error()
  diffusersLogging.disable_progress_bar()

  MODELS = {
    'sd' : {
      'repo' : 'runwayml/stable-diffusion-v1-5',
      'txtPipeline': StableDiffusionPipeline,
      'imgPipeline': StableDiffusionImg2ImgPipeline,
      'dtype' : torch.float16
    },
    'sd-xl' : {
      'repo' : 'stabilityai/stable-diffusion-xl-base-1.0',
      'txtPipeline' : StableDiffusionXLPipeline,
      'imgPipeline' : StableDiffusionXLImg2ImgPipeline,
      'dtype' : torch.float32
    }
  }
  TOKENIZER = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

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
  pipe = cfg['txtPipeline'].from_pretrained(
    cfg['repo'],
    cache_dir=MODEL_DIR,
    torch_dtype=cfg['dtype']
  )

  imgPipe = cfg['imgPipeline'](**pipe.components)

  # deviceType = 'cuda' if torch.cuda.is_available() else 'cpu'
  deviceType = 'mps' if torch.backends.mps.is_available() else 'cpu'

  # Text to image pipeline
  pipe.safety_checker = None
  pipe.set_progress_bar_config(disable=False)

  # Imate to Image refiner pipeline
  imgPipe.safety_checker = None
  imgPipe.set_progress_bar_config(disable=False)

  pipe = pipe.to(deviceType)
  imgPipe = imgPipe.to(deviceType)
  print(f'\n[+] Using device: {deviceType}\n')

  # Check for oversized prompts and alert the user.
  for key in promptData:
    tokens = tokenCount(genPromptString(promptData[key]), TOKENIZER)
    if tokens > MAX_TOKENS:
      print(f'\n\n[!] WARNING: {key.upper()} is {tokens} tokens;'
            f'\n[!] Maximum allowed is {MAX_TOKENS}.\n\n')
    else:
      print(f'[+] Tokens for {key.upper()}: {tokens}')

  # Generate image
  promptText = genPromptString(promptData['prompt'])
  negativePromptText = genPromptString(promptData['exclude'])
  pipeArgs = {
    'prompt': promptText,
    'num_inference_steps': 30,
    'guidance_scale' : 7.5
  }

  # Append negative prompt argument if exists.
  if negativePromptText:
    pipeArgs['negative_prompt'] = negativePromptText

  image = pipe(**pipeArgs).images[0]

  # Start second pass configuration (image 2 image)
  refinePrompt = genPromptString(promptData['refine prompt'])
  refineNegativePrompt = genPromptString(promptData['refine exclude'])
  imgPipeArgs = {
    'prompt' : refinePrompt,
    'image' : image,
    'strength' : 0.20,
    'num_inference_steps': 30,
    'guidance_scale' : 6.0
  }

  # Bring in our negative prompt listing, default to base negatives if present.
  if refineNegativePrompt:
    imgPipeArgs['negative_prompt'] = refineNegativePrompt

  elif negativePromptText:
    imgPipeArgs['negative_prompt'] = negativePromptText

  refinedImage = None
  if refinePrompt:
    print('[+] Refining image...')
    refinedImage = imgPipe(**imgPipeArgs).images[0]

  # Save output
  if refinedImage:
    refinedImage.save(outputFile)
  else:
    image.save(outputFile)

  print(f'\n ---- Image saved to: {outputFile} ----\n')


def tokenCount(text: str, tokenizer: CLIPTokenizer) -> int:
  """
  Parse and return the number of tokens from a given string.

  :param text: Prompt text to be checked.
  :type text: str
  :param tokenizer: Tokenizer to be leveraged to determine tokens used.
  :type tokenizer: CLIPTokenizer
  :return: Number of tokens used in the prompt.
  :rtype: int
  """

  tokens = tokenizer(
    text,
    truncation=False,
    add_special_tokens=True
  )

  return len(tokens["input_ids"])


def configureQuietMode(isQuiet: bool = True) -> None:
  if not isQuiet:
    return

  warnings.filterwarnings('ignore', category=UserWarning)

  logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
  logging.getLogger('transformers').setLevel(logging.ERROR)
  logging.getLogger('diffusers').setLevel(logging.ERROR)


if __name__ == '__main__':
  main()
