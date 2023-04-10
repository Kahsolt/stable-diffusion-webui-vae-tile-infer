# stable-diffusion-webui-vae-tile-infer

    Yet another vae tiling inferer extremely saving your VRAM, extension script for AUTOMATIC1111/stable-diffusion-webui.

----

⚠ This repo is more experimental than the production-ready [multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)'s implementation, please use that in production.  
⚠ But for developers want to see our idea, code in this repo is more clear & neat to understand.  

ℹ When processing with large images, please **turn off previews** to really save time and resoureces!!

⚠ 我们成立了插件反馈 QQ 群: 616795645 (赤狐屿)，欢迎出建议、意见、报告bug等 (w  
⚠ We have a QQ chat group (616795645) now, any suggestions, discussions and bug reports are highly wellllcome!!  

![ui](img/ui.png)


### Benchmark

```
device      = NVIDIA GeForce RTX 3060 (12G VRAM)
dtype       = float16
auto_adjust = True
gn_sync     = Approx
skip_infer  = None
```

⚪ Encoding is cheap

| Image Size | original | tile (tile_size=1024) |
| :-: | :-: | :-: |
|  512 x 512  | 0.009s /  2584.194MB | 0.417s / 2653.301MB / 1 tile  |
|  768 x 768  | 0.011s /  3227.944MB | 0.530s / 3332.989MB / 1 tile  |
| 1024 x 1024 | 0.012s /  4481.913MB | 0.758s / 4271.676MB / 1 tile  |
| 1600 x 1600 | 0.031s /  8512.850MB | 1.499s / 4301.680MB / 4 tiles |
| 2048 x 2048 | 0.034s / 10309.194MB | 2.368s / 4319.680MB / 4 tiles |

⚪ Decoding is heavy

- ablation on image size (tile_size=128)

| Image Size | original | tile |
| :-: | :-: | :-: |
|  512 x 512  | 0.020s /  2616.033MB |  0.202s / 2685.320MB /  1 tile  |
|  768 x 768  | 0.030s /  3296.306MB |  0.427s / 3399.634MB /  1 tile  |
| 1024 x 768  | 0.024s /  3704.470MB |  0.561s / 3824.823MB /  1 tile  |
| 1280 x 720  | 0.023s /  3985.083MB |  1.510s / 4386.115MB /  2 tiles |
| 1024 x 1024 | 0.017s /  4248.689MB |  0.747s / 4386.074MB /  1 tile  |
| 1920 x 1080 | 0.031s /  6375.797MB |  2.325s / 4387.078MB /  4 tiles |
| 2048 x 1024 | 0.032s /  6425.564MB |  2.307s / 4387.107MB /  2 tiles |
| 1600 x 1600 | 0.033s /  8373.138MB |  2.649s / 4387.482MB /  4 tiles |
| 2048 x 1536 | 2.252s /  8602.439MB |  3.041s / 4387.971MB /  4 tiles |
| 2560 x 1440 | 3.899s /  9725.989MB |  3.453s / 4389.521MB /  6 tiles |
| 2048 x 2048 | 2.582s / 10265.877MB |  3.814s / 4389.111MB /  4 tiles |
| 2560 x 4096 |       OOM            |  8.446s / 4397.221MB / 12 tiles |
| 4096 x 4096 |       OOM            | 12.998s / 4407.095MB / 16 tiles |
| 4096 x 8192 |       OOM            | 24.900s / 4428.142MB / 32 tiles |
| 8192 x 8192 |       OOM            | 49.069s / 4469.158MB / 64 tiles |

- ablation on tile size (image_size=2048)

ℹ VRAM peak usage is only related to the tile size, say goodbye to all OOMs :)

| Tile Size | tile |
| :-: | :-: |
|  32 | 3.630s, max VRAM alloc 2247.986 MB / 64 tiles |
|  48 | 3.500s, max VRAM alloc 2433.626 MB / 36 tiles |
|  64 | 3.347s, max VRAM alloc 2689.111 MB / 16 tiles |
|  96 | 3.636s, max VRAM alloc 3402.735 MB /  9 tiles |
| 128 | 3.803s, max VRAM alloc 4389.111 MB /  4 tiles |
| 160 | 4.273s, max VRAM alloc 5646.989 MB /  4 tiles |
| 192 | 5.809s, max VRAM alloc 7930.127 MB /  4 tiles |


### How it works?

- split RGB image / latent image to overlapped tiles (not always be square)
- normally VAE encode / decode each tile
- concatenate all tiles back

⚪ settings tuning

- `Encoder/Decoder tile size`: image tile as the actual processing unit; **set it as large as possible before gets OOM** :)
- `Encoder/Decoder pad size`: overlapped padding of each tile; larger value making more seamless
- `Auto adjust real tile size`: auto shrink real tile size to match tensor shape, avoding too small tailing tile
- `Zigzag processing`: always keep at least one tile on GPU, will be faster but requires a bit more VRAM (only effects when `gn_sync=FULL_SYNC`)
- `GroupNorm sync`: how to sync GroupNorm stats
  - `Approximated`: using stats from the pre-computed low-resolution image
  - `Full sync`: using accurate stats to sync globally
  - `No sync`: do not sync
- `Skip infer (experimental)`: skip calculation of certain network blocks, faster but results low quality


#### Acknowledgement

Thanks for the original idea from:

- multidiffusion-upscaler-for-automatic1111: [https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)

----
by Armit
2023/01/20 
