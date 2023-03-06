#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/05

import math
from time import time
from collections import defaultdict
from traceback import print_exc
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
import gradio as gr

import modules.devices as devices
from modules.scripts import Script, AlwaysVisible
from modules.shared import state
from modules.processing import opt_f
from modules.ui import gr_show

from typing import Tuple, List, Union, Generator
from torch import Tensor
from torch.nn import GroupNorm
from modules.processing import StableDiffusionProcessing
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.model import Encoder, Decoder, ResnetBlock, AttnBlock

Net = Union[Encoder, Decoder]
Tile = Var = Mean = Tensor
TaskRet = Union[Tuple[GroupNorm, Tile, Tuple[Var, Mean]], Tile]
TaskGen = Generator[TaskRet, None, None]

# ↓↓↓ copied from https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111 ↓↓↓

def get_default_encoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if total_memory > 16*1000:
            ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000:
            ENCODER_TILE_SIZE = 2048
        elif total_memory > 8*1000:
            ENCODER_TILE_SIZE = 1536
        else:
            ENCODER_TILE_SIZE = 960
    else:
        ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE

def get_default_decoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   total_memory > 30*1000:
            DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000:
            DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000:
            DECODER_TILE_SIZE = 128
        elif total_memory >  8*1000:
            DECODER_TILE_SIZE = 96
        else:
            DECODER_TILE_SIZE = 64
    else:
        DECODER_TILE_SIZE = 64
    return DECODER_TILE_SIZE

def get_var_mean(input, num_groups, eps=1e-6):
    """
    Get mean and var for group norm
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])
    var, mean = torch.var_mean(input_reshaped, dim=[0, 2, 3, 4], unbiased=False)
    return var, mean

def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    """
    Custom group norm with fixed mean and var

    @param input: input tensor
    @param num_groups: number of groups. by default, num_groups = 32
    @param mean: mean, must be pre-calculated by get_var_mean
    @param var: var, must be pre-calculated by get_var_mean
    @param weight: weight, should be fetched from the original group norm
    @param bias: bias, should be fetched from the original group norm
    @param eps: epsilon, by default, eps = 1e-6 to match the original group norm

    @return: normalized tensor
    """
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c/num_groups)
    input_reshaped = input.contiguous().view(1, int(b * num_groups), channel_in_group, *input.size()[2:])

    out = F.batch_norm(input_reshaped, mean, var, weight=None, bias=None, training=False, momentum=0, eps=eps)
    out = out.view(b, c, *input.size()[2:])

    # post affine transform
    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out

# ↑↑↑ copied from https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111 ↑↑↑


if 'global const':
    DEFAULT_OPEN = False
    DEFAULT_ENABLED = True
    DEFAULT_ENCODER_PAD_SIZE = 16
    DEFAULT_DECODER_PAD_SIZE = 2
    DEFAULT_ENCODER_TILE_SIZE = get_default_encoder_tile_size()
    DEFAULT_DECODER_TILE_SIZE = get_default_decoder_tile_size()
    
    DEFAULT_AUTO_SHRINK = True
    DEFAULT_ZIGZAG_PROCESS = True
    DEFAULT_FORCE_UNSYNC = False
    DEFAULT_SKIP_INFER = False

    DEBUG_SHAPE = False
    DEBUG_STAGE = False

if 'global var':
    auto_shrink:    bool = None
    zigzag_process: bool = None
    force_unsync:   bool = None
    skip_infer:     bool = None

    zigzag_dir:     bool = True     # False: ->, True: <-
    zigzag_to_cpu:  bool = True     # to 'cpu'
    sync_gn:        bool = False
    skip_infer_plan = {
        Encoder: {
            'down0.block0': False,
            'down0.block1': False,
            'down1.block0': False,
            'down1.block1': False,
            'down2.block0': False,
            'down2.block1': False,
            'down3.block0': False,
            'down3.block1': False,
            'mid.block_1': False,
            'mid.attn_1':  False,
            'mid.block_2': False,
        },
        Decoder: {
            'mid.block_1': False,
            'mid.attn_1':  False,
            'mid.block_2': False,
            'up3.block0': False,
            'up3.block1': False,
            'up3.block2': False,
            'up2.block0': False,
            'up2.block1': False,
            'up2.block2': False,
            'up1.block0': False,
            'up1.block1': False,
            'up1.block2': False,
            'up0.block0': False,
            'up0.block1': False,
            'up0.block2': False,
        }
    }


# ↓↓↓ modified from 'ldm/modules/diffusionmodules/model.py' ↓↓↓

def nonlinearity(x:Tensor) -> Tensor:
    return F.silu(x, inplace=True)

def Resblock_forward(self:ResnetBlock, x:Tensor) -> TaskGen:
    if zigzag_process and not zigzag_to_cpu:
        h = x.clone()
    else:
        h = x

    if sync_gn:
        var, mean = get_var_mean(h, self.norm1.num_groups, self.norm1.eps)
        if zigzag_to_cpu: h = h.cpu()
        yield self.norm1, h, (var, mean)
        h = h.to(devices.device)
    else:
        h = self.norm1(h)

    h = nonlinearity(h)
    h: Tensor = self.conv1(h)

    if sync_gn:
        var, mean = get_var_mean(h, self.norm2.num_groups, self.norm2.eps)
        if zigzag_to_cpu: h = h.cpu()
        yield self.norm2, h, (var, mean)
        h = h.to(devices.device)
    else:
        h = self.norm2(h)

    h = nonlinearity(h)
    #h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)
    yield x + h

def AttnBlock_forward(self:AttnBlock, x:Tensor) -> TaskGen:
    if zigzag_process and not zigzag_to_cpu:
        h = x.clone()
    else:
        h = x

    if sync_gn:
        var, mean = get_var_mean(h, self.norm.num_groups, self.norm.eps)
        if zigzag_to_cpu: h = h.cpu()
        yield self.norm, h, (var, mean)
        h = h.to(devices.device)
    else:
        h = self.norm(h)

    q = self.q(h)
    k = self.k(h)
    v = self.v(h)

    # compute attention
    B, C, H, W = q.shape
    q = q.reshape(B, C, H * W)
    q = q.permute(0, 2, 1)         # b,hw,c
    k = k.reshape(B, C, H * W)     # b,c,hw
    w = torch.bmm(q, k)            # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w = w * (int(C)**(-0.5))
    w = torch.nn.functional.softmax(w, dim=2)

    # attend to values
    v = v.reshape(B, C, H * W)
    w = w.permute(0,2,1)          # b,hw,hw (first hw of k, second of q)
    h = torch.bmm(v, w)           # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h = h.reshape(B, C, H, W)

    h: Tensor = self.proj_out(h)
    yield x + h

def Encoder_forward(self:Encoder, x:Tensor) -> TaskGen:
    # prenet
    x = self.conv_in(x)
    if DEBUG_SHAPE: print('conv_in:', x.shape)

    skip_enc = skip_infer_plan[Encoder] if skip_infer else defaultdict(lambda: False)

    # downsampling
    for i_level in range(self.num_resolutions):
        for i_block in range(self.num_res_blocks):
            if not skip_enc[f'down{i_level}.block{i_block}']:
                for item in Resblock_forward(self.down[i_level].block[i_block], x):
                    if isinstance(item, Tensor): x = item
                    else: yield item
                if DEBUG_SHAPE: print(f'down[{i_level}].block[{i_block}]:', x.shape)
            assert not len(self.down[i_level].attn)
        if i_level != self.num_resolutions-1:
            x = self.down[i_level].downsample(x)

    # middle
    if not skip_enc['mid.block_1']:
        for item in Resblock_forward(self.mid.block_1, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('block_1:', x.shape)
    if not skip_enc['mid.attn_1']:
        for item in AttnBlock_forward(self.mid.attn_1, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('attn_1:', x.shape)
    if not skip_enc['mid.block_2']:
        for item in Resblock_forward(self.mid.block_2, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('block_2:', x.shape)

    # end
    if sync_gn:
        var, mean = get_var_mean(x, self.norm_out.num_groups, self.norm_out.eps)
        if zigzag_to_cpu: x = x.cpu()
        yield self.norm_out, x, (var, mean)
        x = x.to(devices.device)
    else:
        x = self.norm_out(x)

    x = nonlinearity(x)
    x = self.conv_out(x)
    yield x.cpu()

def Decoder_forward(self:Decoder, x:Tensor) -> TaskGen:
    # prenet
    x = self.conv_in(x)     # [B, C=4, H, W] => [B, C=512, H, W]
    if DEBUG_SHAPE: print('conv_in:', x.shape)

    skip_dec = skip_infer_plan[Decoder] if skip_infer else defaultdict(lambda: False)

    # middle
    if not skip_dec['mid.block_1']:
        for item in Resblock_forward(self.mid.block_1, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('block_1:', x.shape)
    if not skip_dec['mid.attn_1']:
        for item in AttnBlock_forward(self.mid.attn_1, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('attn_1:', x.shape)
    if not skip_dec['mid.block_2']:
        for item in Resblock_forward(self.mid.block_2, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('block_2:', x.shape)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
        for i_block in range(self.num_res_blocks+1):
            if not skip_dec[f'up{i_level}.block{i_block}']:
                for item in Resblock_forward(self.up[i_level].block[i_block], x):
                    if isinstance(item, Tensor): x = item
                    else: yield item
                if DEBUG_SHAPE: print(f'up[{i_level}].block[{i_block}]:', x.shape)
            assert not len(self.up[i_level].attn)
        if i_level != 0:
            x = self.up[i_level].upsample(x)
            if DEBUG_SHAPE: print(f'up[{i_level}].upsample:', x.shape)

    # end
    if self.give_pre_end: yield x.cpu()

    if sync_gn:
        var, mean = get_var_mean(x, self.norm_out.num_groups, self.norm_out.eps)
        if zigzag_to_cpu: x = x.cpu()
        yield self.norm_out, x, (var, mean)
        x = x.to(devices.device)
    else:
        x = self.norm_out(x)

    x = nonlinearity(x)
    x = self.conv_out(x)
    if DEBUG_SHAPE: print(f'conv_out:', x.shape)
    if self.tanh_out: x = torch.tanh(x)
    yield x.cpu()

# ↑↑↑ modified from 'ldm/modules/diffusionmodules/model.py' ↑↑↑


def perfcount(fn):
    def wrapper(*args, **kwargs):
        ts = time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(devices.device)
        devices.torch_gc()
        gc.collect()
        
        try:
            return fn(*args, **kwargs)
        except:
            raise
        finally:
            devices.torch_gc()
            gc.collect()
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated(devices.device) / 2**20
                torch.cuda.reset_peak_memory_stats(devices.device)
                print(f'Done in {time() - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
            else:
                print(f'Done in {time() - ts:.3f}s')
    
    return wrapper

@perfcount
@torch.inference_mode()
def VAE_forward_tile(self:Net, z:Tensor, tile_size:int, pad_size:int):
    global sync_gn, zigzag_dir, zigzag_to_cpu

    B, C, H, W = z.shape
    is_decoder = isinstance(self, Decoder)
    scaler = opt_f if is_decoder else 1/opt_f
    ch = 3 if is_decoder else 8
    steps = 31 if is_decoder else 23
    #result = z[:, :ch, :, :] if is_decoder else z[:, :ch, :H//opt_f, :W//opt_f]    # very cheap tmp result
    result = None

    if auto_shrink:
        def auto_tile_size(low:int, high:int) -> int:
            ''' VRAM saving when close to low, GPU warp friendy when close to high '''
            align_size = 64 if is_decoder else 512
            while low < high:
                r = low % align_size
                if low + r > high:
                    align_size //= 2
                else:
                    return low + r
            return high

        n_tiles_H = math.ceil(H / tile_size)
        n_tiles_W = math.ceil(W / tile_size)
        ts_low  = math.ceil(max(H / n_tiles_H, W / n_tiles_W))
        ts_high = math.ceil(max(H / (n_tiles_H - 0.15), W / (n_tiles_W - 0.15)))    # assure last tile fill 72.25%
        tile_size = auto_tile_size(ts_low, ts_high)

    if 'estimate max tensor shape':
        if is_decoder:
            shape = torch.Size((B, 256, (H+2*pad_size)*opt_f, (W+2*pad_size)*opt_f))
        else:
            shape = torch.Size((B, 128, tile_size+2*pad_size, tile_size+2*pad_size))
        size_t = 2 if self.conv_in.weight.dtype == torch.float16 else 4
        print(f'>> max tensor shape: {tuple(shape)}, memsize: {shape.numel() * size_t / 2**20:.3f} MB')

    n_tiles_H = math.ceil(H / tile_size)
    n_tiles_W = math.ceil(W / tile_size)
    n_tiles = n_tiles_H * n_tiles_W
    sync_gn = False if force_unsync else n_tiles > 1
    fill_ratio = H * W / (n_tiles * tile_size**2)
    print(f'>> dtype: {z.dtype}')
    print(f'>> input_size: {z.shape}')
    print(f'>> tile_size: {tile_size}')
    print(f'>> split to {n_tiles_H}x{n_tiles_W} = {n_tiles} tiles (fill ratio: {fill_ratio:.3%})')

    if not sync_gn:
        steps = 1
    elif skip_infer:
        for block, skip in skip_infer_plan[type(self)].items():
            if not skip: continue
            if   'attn'  in block: steps -= 1
            elif 'block' in block: steps -= 2

    if pad_size != 0: z = F.pad(z, (pad_size, pad_size, pad_size, pad_size), mode='reflect')     # [B, C, H+2*pad, W+2*pad]

    ''' split tiles '''
    bbox_inputs  = []
    bbox_outputs = []
    x = 0
    for _ in range(n_tiles_H):
        y = 0
        for _ in range(n_tiles_W):
            bbox_inputs.append((
                (x, min(x + tile_size, H) + 2 * pad_size),
                (y, min(y + tile_size, W) + 2 * pad_size),
            ))
            bbox_outputs.append((
                (int(x * scaler), int(min(x + tile_size, H) * scaler)),
                (int(y * scaler), int(min(y + tile_size, W) * scaler)),
            ))
            y += tile_size
        x += tile_size
    if DEBUG_STAGE:
        print('bbox_inputs:')
        print(bbox_inputs)
        print('bbox_outputs:')
        print(bbox_outputs)

    ''' start workers '''
    workers: List[TaskGen] = []
    for bbox in bbox_inputs:
        (Hs, He), (Ws, We) = bbox
        tile = z[:, :, Hs:He, Ws:We]
        workers.append(Decoder_forward(self, tile) if is_decoder else Encoder_forward(self, tile))
    n_workers = len(workers)

    interrupted = False
    pbar = tqdm(total=steps, desc=f'VAE tile {"decoding" if is_decoder else "encoding"}')
    while True:
        if state.interrupted or interrupted: break
        pbar.update()

        # run one round
        try:
            if zigzag_process:
                outputs: List[TaskRet] = [None] * n_workers
                for i in (reversed if zigzag_dir else iter)(range(n_workers)):
                    zigzag_to_cpu = (i != 0) if zigzag_dir else (i != n_workers - 1)
                    outputs[i] = next(workers[i])
                zigzag_dir = not zigzag_dir
            else:
                zigzag_to_cpu = True
                outputs = [next(worker) for worker in workers]

            if not 'check outputs type consistency':
                ret_type = type(outputs[0])
            else:
                ret_types = { type(o) for o in outputs }
                assert len(ret_types) == 1
                ret_type = ret_types.pop()
        except StopIteration:
            print_exc()
            raise ValueError('Error: workers stopped early !!')

        # handle intermediates
        if ret_type == tuple:      # GroupNorm sync barrier
            if not 'check gn object identity':
                gn: GroupNorm = outputs[0][0]
            else:
                gns = { gn for gn, _, _ in outputs }
                if len(gns) > 1:
                    print(f'group_norms: {gns}')
                    raise ValueError('Error: workers progressing states not synchronized !!')
                gn: GroupNorm = list(gns)[0]

            if DEBUG_STAGE:
                print('n_tiles:',         len(outputs))
                print('tile.shape:',      outputs[0][1].shape)
                print('tile[0].device:',  outputs[0][1].device)    # 'cpu'
                print('tile[-1].device:', outputs[-1][1].device)   # 'cuda'

            var  = torch.stack([var  for _, _, (var, _)  in outputs], dim=-1).mean(dim=-1)     # [NG=32], float32, 'cuda'
            mean = torch.stack([mean for _, _, (_, mean) in outputs], dim=-1).mean(dim=-1)
            for _, tile, _ in outputs:
                if state.interrupted: interrupted = True ; break

                tile_n = custom_group_norm(tile.to(mean.device), gn.num_groups, mean, var, gn.weight, gn.bias, gn.eps)
                tile.data = tile_n.to(tile.device)

        elif ret_type == Tensor:   # final Tensor splits
            if DEBUG_STAGE:
              print('n_outputs:',     len(outputs))
              print('output.shape:',  outputs[0].shape)
              print('output.device:', outputs[0].device)      # 'cpu'
            assert len(bbox_outputs) == len(outputs), 'n_tiles != n_bbox_outputs'

            result = torch.zeros([B, ch, int(H*scaler), int(W*scaler)], dtype=outputs[0].dtype)
            count  = torch.zeros([B, 1,  int(H*scaler), int(W*scaler)], dtype=torch.uint8)

            def crop_pad(x:Tensor, size:int):
                if size == 0: return x
                return x[:, :, size:-size, size:-size]

            for i, bbox in enumerate(bbox_outputs):
                (Hs, He), (Ws, We) = bbox
                result[:, :, Hs:He, Ws:We] += crop_pad(outputs[i], int(pad_size * scaler))
                count [:, :, Hs:He, Ws:We] += 1

            count = count.clamp_(min=1)
            result /= count
            pbar.close()
            break               # we're done!

        else:
            raise ValueError(f'Error: unkown ret_type: {ret_type} !!')

    ''' finish '''
    if not is_decoder: result = result.to(devices.device)
    return result


class Script(Script):

    def title(self):
        return "Yet Another VAE Tiling"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Yet Another VAE Tiling', open=DEFAULT_OPEN):
            with gr.Row():
                enabled = gr.Checkbox(label='Enabled', value=lambda: DEFAULT_ENABLED)
                reset = gr.Button(value='Reset defaults')

            with gr.Row():
                encoder_tile_size = gr.Slider(label='Encoder tile size', minimum=512, maximum=4096, step=32, value=lambda: DEFAULT_ENCODER_TILE_SIZE)
                decoder_tile_size = gr.Slider(label='Decoder tile size', minimum=32,  maximum=256,  step=8,  value=lambda: DEFAULT_DECODER_TILE_SIZE)

            with gr.Row():
                encoder_pad_size  = gr.Slider(label='Encoder pad size',  minimum=0,  maximum=64, step=8, value=lambda: DEFAULT_ENCODER_PAD_SIZE)
                decoder_pad_size  = gr.Slider(label='Decoder pad size',  minimum=0,  maximum=8,  step=1, value=lambda: DEFAULT_DECODER_PAD_SIZE)

            reset.click(fn=lambda: [DEFAULT_ENCODER_TILE_SIZE, DEFAULT_ENCODER_PAD_SIZE, DEFAULT_DECODER_TILE_SIZE, DEFAULT_DECODER_PAD_SIZE], 
                        outputs=[encoder_tile_size, encoder_pad_size, decoder_tile_size, decoder_pad_size])

            with gr.Row():
                ext_auto_shrink    = gr.Checkbox(label='Auto adjust real tile size', value=lambda: DEFAULT_AUTO_SHRINK)
                ext_zigzag_process = gr.Checkbox(label='Zigzag processing',          value=lambda: DEFAULT_ZIGZAG_PROCESS)
                ext_force_unsync   = gr.Checkbox(label='Force not sync GroupNorm',   value=lambda: DEFAULT_FORCE_UNSYNC)
                ext_skip_infer     = gr.Checkbox(label='Skip infer (experimental)',  value=lambda: DEFAULT_SKIP_INFER)

            with gr.Group(visible=DEFAULT_SKIP_INFER) as tab_skip_infer:
                gr.HTML('<p> => see "img/VAE_arch.md" for model arch reference </p>')

                with gr.Tab(label='Encoder skip infer'):
                    with gr.Row(variant='compact'):
                        skip_enc_down0_block0 = gr.Checkbox(label='down0.block0')
                        skip_enc_down0_block1 = gr.Checkbox(label='down0.block1')
                        skip_enc_down1_block0 = gr.Checkbox(label='down1.block0')
                        skip_enc_down1_block1 = gr.Checkbox(label='down1.block1')
                    with gr.Row(variant='compact'):
                        skip_enc_down2_block0 = gr.Checkbox(label='down2.block0')
                        skip_enc_down2_block1 = gr.Checkbox(label='down2.block1')
                        skip_enc_down3_block0 = gr.Checkbox(label='down3.block0')
                        skip_enc_down3_block1 = gr.Checkbox(label='down3.block1')
                    with gr.Row(variant='compact'):
                        skip_enc_mid_block_1  = gr.Checkbox(label='mid.block_1')
                        skip_enc_mid_attn_1   = gr.Checkbox(label='mid.attn_1')
                        skip_enc_mid_block_2  = gr.Checkbox(label='mid.block_2')

                with gr.Tab(label='Decoder skip infer'):
                    with gr.Row(variant='compact'):
                        skip_dec_mid_block_1 = gr.Checkbox(label='mid.block_1')
                        skip_dec_mid_attn_1  = gr.Checkbox(label='mid.attn_1')
                        skip_dec_mid_block_2 = gr.Checkbox(label='mid.block_2')
                    with gr.Row(variant='compact'):
                        skip_dec_up3_block0  = gr.Checkbox(label='up3.block0')
                        skip_dec_up3_block1  = gr.Checkbox(label='up3.block1')
                        skip_dec_up3_block2  = gr.Checkbox(label='up3.block2')
                        skip_dec_up2_block0  = gr.Checkbox(label='up2.block0')
                        skip_dec_up2_block1  = gr.Checkbox(label='up2.block1')
                        skip_dec_up2_block2  = gr.Checkbox(label='up2.block2')
                    with gr.Row(variant='compact'):
                        skip_dec_up1_block0  = gr.Checkbox(label='up1.block0 (no skip)', value=False, interactive=False)
                        skip_dec_up1_block1  = gr.Checkbox(label='up1.block1')
                        skip_dec_up1_block2  = gr.Checkbox(label='up1.block2')
                        skip_dec_up0_block0  = gr.Checkbox(label='up0.block0 (no skip)', value=False, interactive=False)
                        skip_dec_up0_block1  = gr.Checkbox(label='up0.block1')
                        skip_dec_up0_block2  = gr.Checkbox(label='up0.block2')

            ext_skip_infer.change(fn=lambda x: gr_show(x), inputs=ext_skip_infer, outputs=tab_skip_infer, show_progress=False)

        return [
            enabled, 
            encoder_tile_size, encoder_pad_size, 
            decoder_tile_size, decoder_pad_size,
            ext_auto_shrink, ext_zigzag_process, ext_force_unsync, ext_skip_infer,
            skip_enc_down0_block0,
            skip_enc_down0_block1,
            skip_enc_down1_block0,
            skip_enc_down1_block1,
            skip_enc_down2_block0,
            skip_enc_down2_block1,
            skip_enc_down3_block0,
            skip_enc_down3_block1,
            skip_enc_mid_block_1,
            skip_enc_mid_attn_1,
            skip_enc_mid_block_2,
            skip_dec_mid_block_1,
            skip_dec_mid_attn_1,
            skip_dec_mid_block_2,
            skip_dec_up3_block0,
            skip_dec_up3_block1,
            skip_dec_up3_block2,
            skip_dec_up2_block0,
            skip_dec_up2_block1,
            skip_dec_up2_block2,
            skip_dec_up1_block1,
            skip_dec_up1_block2,
            skip_dec_up0_block1,
            skip_dec_up0_block2,
        ]

    def process(self, p:StableDiffusionProcessing, 
            enabled:bool, 
            encoder_tile_size:int, encoder_pad_size:int, 
            decoder_tile_size:int, decoder_pad_size:int,
            ext_auto_shrink:bool, ext_zigzag_process:bool, ext_force_unsync:bool, ext_skip_infer:bool,
            skip_enc_down0_block0:bool,
            skip_enc_down0_block1:bool,
            skip_enc_down1_block0:bool,
            skip_enc_down1_block1:bool,
            skip_enc_down2_block0:bool,
            skip_enc_down2_block1:bool,
            skip_enc_down3_block0:bool,
            skip_enc_down3_block1:bool,
            skip_enc_mid_block_1:bool,
            skip_enc_mid_attn_1:bool,
            skip_enc_mid_block_2:bool,
            skip_dec_mid_block_1:bool,
            skip_dec_mid_attn_1:bool,
            skip_dec_mid_block_2:bool,
            skip_dec_up3_block0:bool,
            skip_dec_up3_block1:bool,
            skip_dec_up3_block2:bool,
            skip_dec_up2_block0:bool,
            skip_dec_up2_block1:bool,
            skip_dec_up2_block2:bool,
            skip_dec_up1_block1:bool,
            skip_dec_up1_block2:bool,
            skip_dec_up0_block1:bool,
            skip_dec_up0_block2:bool,
        ):

        vae: AutoencoderKL = p.sd_model.first_stage_model
        if vae.device == torch.device('cpu'): return
        
        encoder: Encoder = vae.encoder
        decoder: Decoder = vae.decoder

        # save original forward (only once)
        if not hasattr(encoder, 'original_forward'): encoder.original_forward = encoder.forward
        if not hasattr(decoder, 'original_forward'): decoder.original_forward = decoder.forward

        # undo hijack
        if not enabled:
            encoder.forward = encoder.original_forward
            decoder.forward = decoder.original_forward
            return

        global auto_shrink, zigzag_process, force_unsync, skip_infer, skip_infer_plan

        # store setting to globals
        auto_shrink = ext_auto_shrink
        zigzag_process = ext_zigzag_process
        force_unsync = ext_force_unsync
        skip_infer = ext_skip_infer
        if ext_skip_infer:
            skip_infer_plan[Encoder].update({
                'down0.block0': skip_enc_down0_block0,
                'down0.block1': skip_enc_down0_block1,
                'down1.block0': skip_enc_down1_block0,
                'down1.block1': skip_enc_down1_block1,
                'down2.block0': skip_enc_down2_block0,
                'down2.block1': skip_enc_down2_block1,
                'down3.block0': skip_enc_down3_block0,
                'down3.block1': skip_enc_down3_block1,
                'mid.block_1':  skip_enc_mid_block_1,
                'mid.attn_1':   skip_enc_mid_attn_1,
                'mid.block_2':  skip_enc_mid_block_2,
            })
            skip_infer_plan[Decoder].update({
                'mid.block_1': skip_dec_mid_block_1,
                'mid.attn_1':  skip_dec_mid_attn_1,
                'mid.block_2': skip_dec_mid_block_2,
                'up3.block0':  skip_dec_up3_block0,
                'up3.block1':  skip_dec_up3_block1,
                'up3.block2':  skip_dec_up3_block2,
                'up2.block0':  skip_dec_up2_block0,
                'up2.block1':  skip_dec_up2_block1,
                'up2.block2':  skip_dec_up2_block2,
                'up1.block1':  skip_dec_up1_block1,
                'up1.block2':  skip_dec_up1_block2,
                'up0.block1':  skip_dec_up0_block1,
                'up0.block2':  skip_dec_up0_block2,
            })

        # apply hijack
        encoder.forward = lambda x: VAE_forward_tile(encoder, x, encoder_tile_size, encoder_pad_size)
        decoder.forward = lambda x: VAE_forward_tile(decoder, x, decoder_tile_size, decoder_pad_size)
