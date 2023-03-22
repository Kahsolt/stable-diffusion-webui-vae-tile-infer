#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/05

import os
from pathlib import Path
from PIL import Image
from PIL.Image import Image as PILImage

import torch
from torch import Tensor
import numpy as np
import gradio as gr

from modules.scripts import Script, AlwaysVisible
from modules.shared import sd_model
from modules.devices import autocast
from modules.script_callbacks import on_before_image_saved, ImageSaveParams, remove_current_script_callbacks
from modules.processing import StableDiffusionProcessing, decode_first_stage

from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion

if 'global const':
    DEFAULT_OPEN    = False
    DEFAULT_ENABLED = False
    DEFAULT_REPEATS = 10

if 'global var':
    # .encode_first_stage(x:Tensor) -> Union[DiagonalGaussianDistribution, Tensor]
    # .get_first_stage_encoding(encoder_posterior:Union[DiagonalGaussianDistribution, Tensor]) -> Tensor
    # .decode_first_stage(z:Tensor, predict_cids=False, force_not_quantize=False) -> Tensor
    sd_model: LatentDiffusion = sd_model
    sd_model.encode_first_stage
    sd_model.get_first_stage_encoding
    sd_model.decode_first_stage
    # .encode(x:Tensor) -> DiagonalGaussianDistribution
    # .decode(x:Tensor) -> Tensor
    # .forward(input:Tensor, sample_posterior=True) -> Tuple[Tensor, DiagonalGaussianDistribution]
    vae: AutoencoderKL = sd_model.first_stage_model
    vae.encode
    vae.decode
    vae.forward
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
        if False:
            return sd_model.get_first_stage_encoding(sd_model.encode_first_stage(X))
        else:
            return vae.encode(X).mode()

def latent_to_image(X:Tensor) -> PILImage:
    ''' => process_images_inner() '''

    if False:
        X = decode_first_stage(sd_model, X)
    else:
        X = vae.decode(X)

    X = X.squeeze_(dim=0)
    X = (X + 1.0) / 2.0
    X = X.clamp_(min=0.0, max=1.0)

    X = 255. * np.moveaxis(X.cpu().numpy(), 0, 2)
    X = X.astype(np.uint8)
    return Image.fromarray(X)

def diff_image(A:PILImage, B:PILImage):
    imA = np.asarray(A, dtype=np.float32) / 255.    # [H, W, C]
    imB = np.asarray(B, dtype=np.float32) / 255.
    L1 = np.abs(imB - imA)
    L1 = torch.from_numpy(L1)
    print(f'  image: {L1.mean().item()}')
    print(f'  image (per channel): {L1.mean(dim=[0, 1]).cpu().numpy().tolist()}')

def diff_latent(A:Tensor, B:Tensor):
    L1 = (B - A).abs()              # [B, C=1, H, W]
    print(f'  latent: {L1.mean().item()}')
    print(f'  latent (per channel): {L1.mean(dim=[0, 2, 3]).cpu().numpy().tolist()}')


def test_img(image:PILImage, repeats:int):
    if image is None: return

    last_image  = image
    last_latent = image_to_latent(last_image)

    breakpoint()

    for i in range(repeats):
        print(f'[{i}/{repeats}]')

        cur_image = latent_to_image(last_latent)     # decode
        diff_image(last_image, cur_image)
        cur_latent = image_to_latent(cur_image)      # decode
        diff_latent(last_latent, cur_latent)

        cur_image.save(Path(os.environ['TEMP']) / f'{i}.png')

        last_image  = cur_image
        last_latent = cur_latent


class Script(Script):

    def title(self):
        return 'VAE idempotence test'

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('VAE idempotence test', open=DEFAULT_OPEN):
            with gr.Row(variant='compact').style(equal_height=True):
                enabled = gr.Checkbox(label='Enabled', value=lambda: DEFAULT_ENABLED)
                repeats = gr.Number(label='Repeat decode/encode times', value=DEFAULT_REPEATS, precision=0)

            with gr.Row(variant='compact').style(equal_height=True):
                img = gr.Image(label='Test an existing image', mode='RGB', type='pil')
                img.change(fn=test_img, inputs=[img, repeats])

        return [ enabled, repeats ]

    def process(self, p:StableDiffusionProcessing, enabled:bool, repeats:int):
        if not enabled: return

        on_before_image_saved(lambda params: test_img(params.image, repeats))

    def postprocess(self, p, processed, *args):
        remove_current_script_callbacks()
