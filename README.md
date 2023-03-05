# stable-diffusion-webui-vae-tile-infer

    Yet another vae tiling inferer, extension script for AUTOMATIC1111/stable-diffusion-webui.

----

**VAE encoding / decoding is resource exhausting, often gets OOM (Out Of Memeory) errors or black images (NaNs).**
**You need this extension to draw large 2k/4k/8k images on low VRAM devices :)**

ℹ To draw really huge picture, you could use => [multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
⚠ When processing with large images, please **turn off previews** to really save time and resoureces!!

⚠ 我们成立了插件反馈 QQ 群: 616795645 (赤狐屿)，欢迎出建议、意见、报告bug等 (w  
⚠ We have a QQ chat group (616795645) now, any suggestions, discussions and bug reports are highly wellllcome!!  

![ui](img/ui.png)


### How it works?

- split RGB image / latent image to overlapped tiles (they might not always be square)
- normally VAE encode / decode each tile
- concatenate all tiles back, the overlapped areas are **averaged** to avoid seams

⚪ settings tuning

- `tile size`: image tile as the actual processing unit; larger value requires more real VRAM
- `pad size`: overlapped padding of each tile; larger value making more seamless

The `max tensor size` in a process:

- for encode: `[B, 128, tile_size + 2 * pad_size, tile_size + 2 * pad_size]`
- for decode: `[B, 256, (latent_hight + 2 * pad_size) * 8, (latent_width + 2 * pad_size) * 8]`

is related to actual `max VRAM alloc`, and could be your reference to tune `tile_size`.


#### Acknowledgement

Thanks for the original idea from:

- multidiffusion-upscaler-for-automatic1111: [https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)

----
by Armit
2023/01/20 
