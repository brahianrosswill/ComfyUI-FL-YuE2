import torch
from torch import nn
from comfy.ldm.audio.autoencoder import OobleckDecoder, DecoderBlock, ResidualUnit, SnakeBeta
from comfy.model_management import throw_exception_if_processing_interrupted


def _dependency_interval(module, low, high):
    """Inclusive input support of an output interval; no waveform blending."""
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in reversed(list(layers)):
            low, high = _dependency_interval(child, low, high)
        return low, high
    if isinstance(module, ResidualUnit):
        a, b = _dependency_interval(module.layers, low, high)
        return min(a, low), max(b, high)
    if isinstance(module, nn.ConvTranspose1d):
        s, p, d, k = (module.stride[0], module.padding[0],
                      module.dilation[0], module.kernel_size[0])
        return -(-(low + p - d * (k - 1)) // s), (high + p) // s
    if isinstance(module, nn.Conv1d):
        s, p, d, k = (module.stride[0], module.padding[0],
                      module.dilation[0], module.kernel_size[0])
        return low * s - p, high * s - p + d * (k - 1)
    if isinstance(module, (SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return low, high
    raise TypeError(f"No audited support rule for {type(module).__name__}")


def _output_length(module, length):
    if isinstance(module, (nn.Sequential, OobleckDecoder, DecoderBlock)):
        layers = module if isinstance(module, nn.Sequential) else module.layers
        for child in layers:
            length = _output_length(child, length)
        return length
    if isinstance(module, nn.ConvTranspose1d):
        return ((length - 1) * module.stride[0] - 2 * module.padding[0]
                + module.dilation[0] * (module.kernel_size[0] - 1)
                + module.output_padding[0] + 1)
    if isinstance(module, nn.Conv1d):
        return ((length + 2 * module.padding[0]
                 - module.dilation[0] * (module.kernel_size[0] - 1) - 1)
                // module.stride[0] + 1)
    if isinstance(module, (ResidualUnit, SnakeBeta, nn.ELU, nn.Identity, nn.Tanh)):
        return length
    raise TypeError(f"No audited length rule for {type(module).__name__}")


class YuE2VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        decoder = dict(config["decoder_config"])
        if decoder.pop("snake_type", "vanilla") != "vanilla" or decoder.pop("use_filter", False):
            raise ValueError("Unsupported YuE2 decoder format")
        self.decoder = OobleckDecoder(**decoder)
        self.ratio = config["downsampling_ratio"]
        self.sample_rate = config["sample_rate"]

    def decode(self, latent):
        return self.decoder(latent)

    def decode_tiled(self, latent, core_frames=1024, on_progress=None):
        frames = latent.shape[-1]
        total = _output_length(self.decoder, frames)
        low, high = _dependency_interval(self.decoder, 0, core_frames * self.ratio - 1)
        halo = max(16, -low, high - core_frames + 1)
        output = torch.empty((latent.shape[0], 2, total), dtype=torch.float32, device="cpu")
        tiles = (frames + core_frames - 1) // core_frames
        for index, start in enumerate(range(0, frames, core_frames)):
            throw_exception_if_processing_interrupted()
            end = min(frames, start + core_frames)
            left, right = max(0, start - halo), min(frames, end + halo)
            device = next(self.decoder.parameters()).device
            tile = self.decode(latent[..., left:right].to(device=device, dtype=torch.float32))
            out_start, out_end = start * self.ratio, min(end * self.ratio, total)
            crop_start = (start - left) * self.ratio
            output[..., out_start:out_end].copy_(tile[..., crop_start:crop_start + out_end - out_start])
            del tile
            if on_progress is not None:
                on_progress(index + 1, tiles)
        return output
