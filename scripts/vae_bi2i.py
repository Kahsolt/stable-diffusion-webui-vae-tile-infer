#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/05

import shutil
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage

import torch
from torch import Tensor
import numpy as np
import gradio as gr
from tqdm import tqdm

from modules.scripts import Script, AlwaysVisible
from modules.shared import sd_model
from modules.devices import autocast
from modules.processing import decode_first_stage

if 'global const':
    DEFAULT_OPEN = False

if 'global var':
    vae    = sd_model.first_stage_model
    dtype  = vae.dtype
    device = vae.device


def image_to_latent(img:PILImage) -> Tensor:
    ''' => StableDiffusionProcessingImg2Img.init() '''

    im = np.array(img).astype(np.float32) / 255.0
    im = np.moveaxis(im, 2, 0)

    X = torch.from_numpy(im).to(device)
    X = 2.0 * X - 1.0   # [-1, 1]
    X = X.unsqueeze_(dim=0)
    with autocast():
        return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(X))

def latent_to_image(X:Tensor) -> PILImage:
    ''' => process_images_inner() '''

    X = decode_first_stage(sd_model, X)
    X = X.squeeze_(dim=0)
    X = (X + 1.0) / 2.0
    X = X.clamp_(min=0.0, max=1.0)

    X = 255. * np.moveaxis(X.cpu().numpy(), 0, 2)
    X = X.astype(np.uint8)
    return Image.fromarray(X)


def process(dp_in:str, dp_out:str):
    dp_in:  Path = Path(dp_in)
    if not dp_in.exists() or not dp_in.is_dir():
        print(f'>> invalid input path: {dp_in}')
        return

    dp_out: Path = Path(dp_out)
    if dp_out.exists():
        shutil.rmtree(str(dp_out))
    dp_out.mkdir(parents=True)

    fps = list(dp_in.iterdir())
    if len(fps) <= 2:
        print(f'>> too few image to blend: {len(fps)}')
        return

    x = image_to_latent(Image.open(fps[0]).convert('RGB'))
    for fp in tqdm(fps[1:]):
        y = image_to_latent(Image.open(fp).convert('RGB'))
        z = (x + y) / 2
        latent_to_image(z).save(dp_out / f'{fp.stem}.png')

        x = y


class Script(Script):

    def title(self):
        return 'VAE batch img2img'

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('VAE batch img2img', open=DEFAULT_OPEN):
            with gr.Row(variant='compact').style(equal_height=True):
                dp_in  = gr.Text(label='Input directory',  max_lines=1)
            with gr.Row(variant='compact').style(equal_height=True):
                dp_out = gr.Text(label='Output directory', max_lines=1)

            with gr.Row(variant='compact').style(equal_height=True):
                btn = gr.Button(value='Process!')
                btn.click(fn=process, inputs=[dp_in, dp_out])

        return []
