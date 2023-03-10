#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/03/05

import os
import math
from pathlib import Path
from time import time
from collections import defaultdict
from enum import Enum
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
from modules.sd_vae_approx import cheap_approximation
from modules.ui import gr_show

from typing import Tuple, List, Dict, Union, Generator
from torch import Tensor
from torch.nn import GroupNorm
from modules.processing import StableDiffusionProcessing
from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.diffusionmodules.model import Encoder, Decoder, ResnetBlock, AttnBlock

Net = Union[Encoder, Decoder]
Tile = Var = Mean = Tensor
TaskRet = Union[Tuple[GroupNorm, Tile, Tuple[Var, Mean]], Tile]
TaskGen = Generator[TaskRet, None, None]
BBox = Tuple[int, int, int, int]


# ↓↓↓ copied from https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111 ↓↓↓

def get_default_encoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   total_memory > 16*1000: ENCODER_TILE_SIZE = 3072
        elif total_memory > 12*1000: ENCODER_TILE_SIZE = 2048
        elif total_memory > 10*1000: ENCODER_TILE_SIZE = 1536
        elif total_memory >  8*1000: ENCODER_TILE_SIZE = 1536
        elif total_memory >  6*1000: ENCODER_TILE_SIZE = 1024
        elif total_memory >  4*1000: ENCODER_TILE_SIZE = 768
        else:                        ENCODER_TILE_SIZE = 512
    else:
        ENCODER_TILE_SIZE = 512
    return ENCODER_TILE_SIZE

def get_default_decoder_tile_size():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(devices.device).total_memory // 2**20
        if   total_memory > 30*1000: DECODER_TILE_SIZE = 256
        elif total_memory > 16*1000: DECODER_TILE_SIZE = 192
        elif total_memory > 12*1000: DECODER_TILE_SIZE = 128
        elif total_memory > 10*1000: DECODER_TILE_SIZE = 96
        elif total_memory >  8*1000: DECODER_TILE_SIZE = 96
        elif total_memory >  6*1000: DECODER_TILE_SIZE = 80
        elif total_memory >  4*1000: DECODER_TILE_SIZE = 64
        else:                        DECODER_TILE_SIZE = 64
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
    if torch.isinf(var).any():
        var, mean = torch.var_mean(input_reshaped.float(), dim=[0, 2, 3, 4], unbiased=False)
        var, mean = var.to(input_reshaped.dtype), mean.to(input_reshaped.dtype)
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
    if weight is not None: out *= weight.view(1, -1, 1, 1)
    if bias   is not None: out += bias.view(1, -1, 1, 1)
    return out

# ↑↑↑ copied from https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111 ↑↑↑


class GroupNormSync(Enum):
    APPROX = 'Approximated'
    SYNC   = 'Full sync'
    UNSYNC = 'No sync'

if 'global const':
    DEFAULT_OPEN              = False
    DEFAULT_ENABLED           = True
    DEFAULT_SMART_IGNORE      = True
    DEFAULT_ENCODER_PAD_SIZE  = 16
    DEFAULT_DECODER_PAD_SIZE  = 2
    DEFAULT_ENCODER_TILE_SIZE = get_default_encoder_tile_size()
    DEFAULT_DECODER_TILE_SIZE = get_default_decoder_tile_size()
    
    DEFAULT_AUTO_SHRINK    = True
    DEFAULT_ZIGZAG_PROCESS = True
    DEFAULT_GN_SYNC        = GroupNormSync.APPROX.value
    DEFAULT_SKIP_INFER     = False

    DEBUG_SHAPE  = False
    DEBUG_STAGE  = False
    DEBUG_APPROX = False

if 'global var':
    smart_ignore:   bool = None
    auto_shrink:    bool = None
    gn_sync:        GroupNormSync = None
    skip_infer:     bool = None

    zigzag_dir:     bool = True     # False: ->, True: <-
    zigzag_to_cpu:  bool = False    # stash to 'cpu' and wait for apply `custom_group_norm`
    sync_approx:    bool = False    # True: apply, False: collect
    sync_approx_pc: int  = 0        # program cpunter of 'sync_approx_plan' execution
    sync_approx_plan: List[Tuple[Var, Mean]] = []
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
            'mid.block_1':  False,
            'mid.attn_1':   False,
            'mid.block_2':  False,
        },
        Decoder: {
            'mid.block_1': False,
            'mid.attn_1':  False,
            'mid.block_2': False,
            'up3.block0':  False,
            'up3.block1':  False,
            'up3.block2':  False,
            'up2.block0':  False,
            'up2.block1':  False,
            'up2.block2':  False,
            'up1.block0':  False,
            'up1.block1':  False,
            'up1.block2':  False,
            'up0.block0':  False,
            'up0.block1':  False,
            'up0.block2':  False,
        }
    }
    skip_infer_plan_dummy = defaultdict(lambda: False)


def _dbg_tensor(X:Tensor, name:str) -> None:
    var, mean = torch.var_mean(X)
    print(f'{name}: {list(X.shape)}, {X.max().item():.4f}, {X.min().item():.4f}, {mean.item():.4f}, {var.item():.4f}')

def _dbg_to_image(X:Tensor, name:str) -> None:
    import numpy as np
    from PIL import Image

    im = X.permute([1, 2, 0])
    im = (im + 1) / 2
    im = im.clamp_(0, 1)
    im = im.cpu().numpy()
    im = (im * 255).astype(np.uint8)
    img = Image.fromarray(im)
    img.save(Path(os.environ['TEMP']) / (name + '.png'))


# ↓↓↓ modified from 'ldm/modules/diffusionmodules/model.py' ↓↓↓

def nonlinearity(x:Tensor) -> Tensor:
    return F.silu(x, inplace=True)

def GroupNorm_forward(gn:GroupNorm, h:Tensor) -> TaskGen:
    if   gn_sync == GroupNormSync.SYNC:
        var, mean = get_var_mean(h, gn.num_groups, gn.eps)
        if zigzag_to_cpu: h = h.cpu()
        yield gn, h, (var, mean)
        h = h.to(devices.device)
    elif gn_sync == GroupNormSync.APPROX:
        if sync_approx:     # apply
            global sync_approx_pc
            var, mean = sync_approx_plan[sync_approx_pc]
            h = custom_group_norm(h, gn.num_groups, mean, var, gn.weight, gn.bias, gn.eps)
            sync_approx_pc = (sync_approx_pc + 1) % len(sync_approx_plan)
        else:               # collect
            var, mean = get_var_mean(h, gn.num_groups, gn.eps)
            sync_approx_plan.append((var, mean))
            h = gn(h)
    elif gn_sync == GroupNormSync.UNSYNC:
        h = gn(h)
    yield h

def Resblock_forward(self:ResnetBlock, x:Tensor) -> TaskGen:
    h = x.clone() if (gn_sync == GroupNormSync.SYNC and not zigzag_to_cpu) else x

    for item in GroupNorm_forward(self.norm1, h):
        if isinstance(item, Tensor): h = item
        else: yield item

    h = nonlinearity(h)
    h: Tensor = self.conv1(h)

    for item in GroupNorm_forward(self.norm2, h):
        if isinstance(item, Tensor): h = item
        else: yield item

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
    h = x.clone() if (gn_sync == GroupNormSync.SYNC and not zigzag_to_cpu) else x

    for item in GroupNorm_forward(self.norm, h):
        if isinstance(item, Tensor): h = item
        else: yield item

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

def _mid_forward(self:Net, x:Tensor, skip_plan:Dict[str, bool]) -> TaskGen:
    if not skip_plan['mid.block_1']:
        for item in Resblock_forward(self.mid.block_1, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('block_1:', x.shape)
    if not skip_plan['mid.attn_1']:
        for item in AttnBlock_forward(self.mid.attn_1, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('attn_1:', x.shape)
    if not skip_plan['mid.block_2']:
        for item in Resblock_forward(self.mid.block_2, x):
            if isinstance(item, Tensor): x = item
            else: yield item
        if DEBUG_SHAPE: print('block_2:', x.shape)
    yield x

def Encoder_forward(self:Encoder, x:Tensor) -> TaskGen:
    # prenet
    x = self.conv_in(x)
    if DEBUG_SHAPE: print('conv_in:', x.shape)

    skip_enc = skip_infer_plan[Encoder] if skip_infer else skip_infer_plan_dummy

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
    for item in _mid_forward(self, x, skip_enc):
        if isinstance(item, Tensor): x = item
        else: yield item

    # end
    for item in GroupNorm_forward(self.norm_out, x):
        if isinstance(item, Tensor): x = item
        else: yield item
    
    x = nonlinearity(x)
    x = self.conv_out(x)
    yield x.cpu()

def Decoder_forward(self:Decoder, x:Tensor) -> TaskGen:
    # prenet
    x = self.conv_in(x)     # [B, C=4, H, W] => [B, C=512, H, W]
    if DEBUG_SHAPE: print('conv_in:', x.shape)

    skip_dec = skip_infer_plan[Decoder] if skip_infer else skip_infer_plan_dummy

    # middle
    for item in _mid_forward(self, x, skip_dec):
        if isinstance(item, Tensor): x = item
        else: yield item

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

    for item in GroupNorm_forward(self.norm_out, x):
        if isinstance(item, Tensor): x = item
        else: yield item

    x = nonlinearity(x)
    x = self.conv_out(x)
    if DEBUG_SHAPE: print(f'conv_out:', x.shape)
    if self.tanh_out: x = torch.tanh(x)
    yield x.cpu()

# ↑↑↑ modified from 'ldm/modules/diffusionmodules/model.py' ↑↑↑


def get_real_tile_config(z:Tensor, tile_size:int, is_decoder:bool) -> Tuple[int, int, int, int]:
    global gn_sync
    
    B, C, H, W = z.shape

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
        tile_size_H = auto_tile_size(math.ceil(H / n_tiles_H), math.ceil(H / (n_tiles_H - 0.15)))   # assure last tile fill 72.25%
        tile_size_W = auto_tile_size(math.ceil(W / n_tiles_W), math.ceil(W / (n_tiles_W - 0.15)))
    else:
        tile_size_H = tile_size_W = tile_size

    n_tiles_H = math.ceil(H / tile_size_H)
    n_tiles_W = math.ceil(W / tile_size_W)
    n_tiles = n_tiles_H * n_tiles_W
    if n_tiles <= 1: gn_sync == GroupNormSync.UNSYNC        # trick: force unsync when signle tile
    fill_ratio = H * W / (n_tiles * tile_size_H * tile_size_W)

    suffix = ''
    if gn_sync == GroupNormSync.APPROX: suffix = '(apply)' if sync_approx else '(collect)' 
    print(f'>> sync group norm: {gn_sync.value} {suffix}')
    print(f'>> input: {list(z.shape)}, {str(z.dtype)[len("torch."):]} on {z.device}')
    print(f'>> real tile size: {tile_size_H} x {tile_size_W}')
    print(f'>> split to {n_tiles_H} x {n_tiles_W} = {n_tiles} tiles (fill ratio: {fill_ratio:.3%})')

    return tile_size_H, tile_size_W, n_tiles_H, n_tiles_W

def make_bbox(n_tiles_H:int, n_tiles_W:int, tile_size_H:int, tile_size_W:int, H:int, W:int, P:int, scaler:Union[int, float]) -> Tuple[List[BBox], List[BBox]]:
    bbox_inputs:  List[BBox] = []
    bbox_outputs: List[BBox] = []

    x = 0
    for _ in range(n_tiles_H):
        y = 0
        for _ in range(n_tiles_W):
            bbox_inputs.append((
                x, min(x + tile_size_H, H) + 2 * P,
                y, min(y + tile_size_W, W) + 2 * P,
            ))
            bbox_outputs.append((
                int(x * scaler), int(min(x + tile_size_H, H) * scaler),
                int(y * scaler), int(min(y + tile_size_W, W) * scaler),
            ))
            y += tile_size_W
        x += tile_size_H

    if DEBUG_STAGE:
        print('bbox_inputs:')
        print(bbox_inputs)
        print('bbox_outputs:')
        print(bbox_outputs)

    return bbox_inputs, bbox_outputs

def get_n_sync(net:Net, is_decoder:bool) -> int:
    if gn_sync != GroupNormSync.SYNC: return 1

    n_sync = 31 if is_decoder else 23
    if skip_infer:
        for block, skip in skip_infer_plan[type(net)].items():
            if not skip: continue
            if   'attn'  in block: n_sync -= 1
            elif 'block' in block: n_sync -= 2
    return n_sync

def perfcount(fn):
    def wrapper(*args, **kwargs):
        device = devices.device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        devices.torch_gc()
        gc.collect()
        
        ts = time()
        try: return fn(*args, **kwargs)
        except: raise
        finally:
            te = time()
            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated(device) / 2**20
                torch.cuda.reset_peak_memory_stats(device)
                print(f'Done in {te - ts:.3f}s, max VRAM alloc {vram:.3f} MB')
            else:
                print(f'Done in {te - ts:.3f}s')
            devices.torch_gc()
            gc.collect()
    return wrapper


@torch.inference_mode()
def VAE_forward_tile(self:Net, z:Tensor, tile_size:int, pad_size:int) -> Tensor:
    global gn_sync, zigzag_dir, zigzag_to_cpu, sync_approx_plan

    B, C, H, W = z.shape
    P = pad_size
    is_decoder = isinstance(self, Decoder)
    scaler     = opt_f if is_decoder else 1/opt_f
    ch         = 3     if is_decoder else 8
    result     = None       # z[:, :ch, :, :] if is_decoder else z[:, :ch, :H//opt_f, :W//opt_f]    # very cheap tmp result

    # modified: gn_sync
    tile_size_H, tile_size_W, n_tiles_H, n_tiles_W = get_real_tile_config(z, tile_size, is_decoder)

    if 'estimate max tensor shape':
        if is_decoder: shape = torch.Size((B, 256, (tile_size_H+2*P)*opt_f, (tile_size_W+2*P)*opt_f))
        else:          shape = torch.Size((B, 128, tile_size_H+2*P, tile_size_W+2*P))
        size_t = 2 if self.conv_in.weight.dtype == torch.float16 else 4
        print(f'>> max tensor shape: {list(shape)}, estimated vram size: {shape.numel() * size_t / 2**20:.3f} MB')
    
    ''' split tiles '''
    if P != 0: z = F.pad(z, (P, P, P, P), mode='reflect')     # [B, C, H+2*pad, W+2*pad]

    bbox_inputs, bbox_outputs = make_bbox(n_tiles_H, n_tiles_W, tile_size_H, tile_size_W, H, W, P, scaler)
    workers: List[TaskGen] = []
    for bbox in bbox_inputs:
        Hs, He, Ws, We = bbox
        tile = z[:, :, Hs:He, Ws:We]
        workers.append(Decoder_forward(self, tile) if is_decoder else Encoder_forward(self, tile))
    del z
    n_workers = len(workers)
    if n_workers >= 3: workers = workers[1:] + [workers[0]]     # trick: put two largest tiles at end for full_sync zigzagging

    ''' start workers '''
    steps = get_n_sync(self, is_decoder) * n_workers
    pbar = tqdm(total=steps, desc=f'VAE tile {"decoding" if is_decoder else "encoding"}')
    while True:
        if state.interrupted: return

        # run one round
        try:
            outputs: List[TaskRet] = [None] * n_workers
            for i in (reversed if zigzag_dir else iter)(range(n_workers)):
                if state.interrupted: return

                zigzag_to_cpu = (i != 0) if zigzag_dir else (i != n_workers - 1)
                outputs[i] = next(workers[i])
                pbar.update()
                if isinstance(outputs[i], Tile): workers[i] = None    # trick: release resource when done
            zigzag_dir = not zigzag_dir

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
        if   ret_type == tuple:     # GroupNorm full sync barrier
            assert gn_sync == GroupNormSync.SYNC

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
                if state.interrupted: return

                tile_n = custom_group_norm(tile.to(mean.device), gn.num_groups, mean, var, gn.weight, gn.bias, gn.eps)
                tile.data = tile_n.to(tile.device)

        elif ret_type == Tile:      # final Tensor splits
            if DEBUG_STAGE:
              print('n_outputs:',     len(outputs))
              print('output.shape:',  outputs[0].shape)
              print('output.device:', outputs[0].device)      # 'cpu'
            assert len(bbox_outputs) == len(outputs), 'n_tiles != n_bbox_outputs'

            if n_workers >= 3: outputs = [outputs[-1]] + outputs[:-1]   # trick: rev put two largest tiles at end for full_sync zigzagging

            result = torch.zeros([B, ch, int(H*scaler), int(W*scaler)], dtype=outputs[0].dtype)
            crop_pad = lambda x, P: x if P == 0 else x[:, :, P:-P, P:-P]
            for i, bbox in enumerate(bbox_outputs):
                Hs, He, Ws, We = bbox
                result[:, :, Hs:He, Ws:We] += crop_pad(outputs[i], int(P * scaler))

            pbar.close()
            break               # we're done!

        else:
            raise ValueError(f'Error: unkown ret_type: {ret_type} !!')

    ''' finish '''
    if not is_decoder: result = result.to(devices.device)
    return result

@perfcount
def VAE_hijack(enabled:bool, self:Net, z:Tensor, tile_size:int, pad_size:int) -> Tensor:
    if not enabled: return self.original_forward(z)

    global gn_sync, sync_approx, sync_approx_plan

    B, C, H, W = z.shape
    if max(H, W) <= tile_size:
        if smart_ignore:
            return self.original_forward(z)
        if gn_sync == GroupNormSync.APPROX:
            print('<< ignore gn_sync=APPROX due to tensor to small ;)')
            gn_sync = GroupNormSync.UNSYNC

    if gn_sync == GroupNormSync.APPROX:
        # collect
        sync_approx = False
        sync_approx_plan.clear()

        z_hat: Tensor = F.interpolate(z, size=(tile_size, tile_size), mode='nearest')    # NOTE: do NOT interp in order to keep stats
        if DEBUG_APPROX:
            _dbg_tensor(z, 'z')
            _dbg_tensor(z_hat, 'z_hat')
            _dbg_to_image(cheap_approximation(z_hat[0].float()), 'z_hat')

        if 'stats shift':
            std_src, mean_src = torch.std_mean(z_hat, dim=[0, 2, 3], keepdim=True)
            std_tgt, mean_tgt = torch.std_mean(z,     dim=[0, 2, 3], keepdim=True)
            z_hat = (z_hat - mean_src) / std_src
            z_hat = z_hat * std_tgt + mean_tgt
            z_hat = z_hat.clamp_(z.min(), z.max())
            if DEBUG_APPROX:
                _dbg_tensor(z_hat, 'z_hat_shift')
                _dbg_to_image(cheap_approximation(z_hat[0].float()), 'z_hat_shift')
            del std_src, mean_src, std_tgt, mean_tgt

        x_hat = VAE_forward_tile(self, z_hat, tile_size, pad_size)
        if DEBUG_APPROX:
            _dbg_to_image(x_hat[0].float(), 'z_approx')
        del z_hat, x_hat

        # apply
        sync_approx = True

    try:
        return VAE_forward_tile(self, z, tile_size, pad_size)
    except:
        print_exc()
        return torch.stack([cheap_approximation(sample.float()).to(sample) for sample in z], dim=0)
    finally:
        sync_approx_plan.clear()


class Script(Script):

    def title(self):
        return "Yet Another VAE Tiling"

    def show(self, is_img2img):
        return AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion('Yet Another VAE Tiling', open=DEFAULT_OPEN):
            with gr.Row(variant='compact').style(equal_height=True):
                enabled = gr.Checkbox(label='Enabled', value=lambda: DEFAULT_ENABLED)
                reset = gr.Button(value='↻', variant='tool')

            with gr.Row(variant='compact').style(equal_height=True):
                encoder_tile_size = gr.Slider(label='Encoder tile size', minimum=512, maximum=4096, step=32, value=lambda: DEFAULT_ENCODER_TILE_SIZE)
                decoder_tile_size = gr.Slider(label='Decoder tile size', minimum=32,  maximum=256,  step=8,  value=lambda: DEFAULT_DECODER_TILE_SIZE)

            with gr.Row(variant='compact').style(equal_height=True):
                encoder_pad_size  = gr.Slider(label='Encoder pad size',  minimum=0,  maximum=64, step=8, value=lambda: DEFAULT_ENCODER_PAD_SIZE)
                decoder_pad_size  = gr.Slider(label='Decoder pad size',  minimum=0,  maximum=8,  step=1, value=lambda: DEFAULT_DECODER_PAD_SIZE)

            reset.click(fn=lambda: [DEFAULT_ENCODER_TILE_SIZE, DEFAULT_ENCODER_PAD_SIZE, DEFAULT_DECODER_TILE_SIZE, DEFAULT_DECODER_PAD_SIZE], 
                        outputs=[encoder_tile_size, encoder_pad_size, decoder_tile_size, decoder_pad_size])

            with gr.Row(variant='compact').style(equal_height=True):
                ext_smart_ignore = gr.Checkbox(label='Do not process small images', value=lambda: DEFAULT_SMART_IGNORE)
                ext_auto_shrink  = gr.Checkbox(label='Auto adjust real tile size',  value=lambda: DEFAULT_AUTO_SHRINK)
                ext_gn_sync      = gr.Dropdown(label='GroupNorm sync',              value=lambda: DEFAULT_GN_SYNC, choices=[e.value for e in GroupNormSync])
                ext_skip_infer   = gr.Checkbox(label='Skip infer (experimental)',   value=lambda: DEFAULT_SKIP_INFER)

            with gr.Group(visible=DEFAULT_SKIP_INFER) as tab_skip_infer:
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

                with gr.Row():
                    gr.HTML('<p> => see "img/VAE_arch.md" for model arch reference </p>')

            ext_skip_infer.change(fn=lambda x: gr_show(x), inputs=ext_skip_infer, outputs=tab_skip_infer, show_progress=False)

        return [
            enabled, 
            encoder_tile_size, encoder_pad_size, 
            decoder_tile_size, decoder_pad_size,
            ext_smart_ignore,
            ext_auto_shrink,
            ext_gn_sync, 
            ext_skip_infer,
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
            ext_smart_ignore:bool, 
            ext_auto_shrink:bool, 
            ext_gn_sync:str, 
            ext_skip_infer:bool,
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
            from inspect import isfunction, getfullargspec
            if isfunction(encoder.forward) and getfullargspec(encoder.forward).args[0] == 'x':
                encoder.forward = encoder.original_forward
            if isfunction(decoder.forward) and getfullargspec(decoder.forward).args[0] == 'x':
                decoder.forward = decoder.original_forward
            return

        # extras parameters
        if enabled:
            global smart_ignore, auto_shrink, gn_sync, skip_infer, skip_infer_plan

            # store setting to globals
            smart_ignore   = ext_smart_ignore
            auto_shrink    = ext_auto_shrink
            gn_sync        = GroupNormSync(ext_gn_sync)
            skip_infer     = ext_skip_infer

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
        encoder.forward = lambda x: VAE_hijack(enabled, encoder, x, encoder_tile_size, encoder_pad_size)
        decoder.forward = lambda x: VAE_hijack(enabled, decoder, x, decoder_tile_size, decoder_pad_size)
