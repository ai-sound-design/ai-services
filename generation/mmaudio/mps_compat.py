"""MMAudio on the Apple GPU (MPS) with the pinned torch 2.5.1.

BigVGAN, the vocoder that turns MMAudio's mel spectrogram into audio, filters and
upsamples the whole sound at once with 1-D convolutions along time. PyTorch 2.5
runs a 1-D convolution on MPS as a 2-D one and refuses one whose time axis is too
long, with a misleading message ("Output channels > 65536 not supported at the MPS
device"); PYTORCH_ENABLE_MPS_FALLBACK does not cover that check. Measured on an M1
Ultra: conv1d fails above 65,536 output samples, conv_transpose1d above 65,536
input samples, so every sound longer than about 3.7 s failed at its last step.
Newer torch lifts the limit, but the service stays on the version its Docker image
pins.

A convolution is linear and local in time, so a long call is cut into stretches
that each stay under the limit and run on the GPU:

  conv1d            each block of outputs reads its own input stretch (with the
                    kernel's reach beyond it); the blocks are joined
  conv_transpose1d  each block of inputs scatters into the output, the overlapping
                    tails are added (overlap-add); the bias is added once

The result is the same as one call (within float rounding). Only calls on MPS
that exceed the limit take this path, and only with padding 0, which is how
BigVGAN calls them (it pads explicitly beforehand); everything else, and every
call on CUDA or the CPU, goes straight through. install() does nothing without MPS.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MPS_MAX_LENGTH = 65536          # longest time axis MPS takes (see above)
BLOCK = 32768                   # stretch per call, well under the limit
_installed = False


def _zero(value) -> bool:
    if isinstance(value, (tuple, list)):
        return all(v == 0 for v in value)
    return value == 0


def _one(value) -> int:
    return int(value[0]) if isinstance(value, (tuple, list)) else int(value)


def install() -> None:
    """Wrap F.conv1d and F.conv_transpose1d once (torch.nn.functional is what MMAudio's
    modules look the functions up in at every call)."""
    global _installed
    if _installed or not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return
    _installed = True
    conv1d, conv_transpose1d = F.conv1d, F.conv_transpose1d

    def conv1d_blocks(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.device.type == "mps" and input.dim() == 3:
            s, d, k = _one(stride), _one(dilation), weight.shape[-1]
            reach = d * (k - 1) + 1                       # input samples one output sample reads
            # Zero padding (nn.Conv1d's padding=p, or "same"/"valid") made explicit, so the
            # blocks below see the padded signal and need none of their own
            if padding == "valid":
                left = right = 0
            elif padding == "same":
                left, right = (reach - 1) // 2, (reach - 1) - (reach - 1) // 2
            else:
                left = right = _one(padding)
            length = input.shape[-1] + left + right
            out_len = (length - reach) // s + 1
            if out_len > MPS_MAX_LENGTH:
                if left or right:
                    input = F.pad(input, (left, right))
                parts = []
                for o0 in range(0, out_len, BLOCK):
                    o1 = min(out_len, o0 + BLOCK)
                    piece = input[..., o0 * s:(o1 - 1) * s + reach]
                    parts.append(conv1d(piece, weight, bias, s, 0, d, groups))
                return torch.cat(parts, dim=-1)
        return conv1d(input, weight, bias, stride, padding, dilation, groups)

    def conv_transpose1d_blocks(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        p, op = _one(padding), _one(output_padding)
        if (input.device.type == "mps" and input.dim() == 3 and input.shape[-1] > MPS_MAX_LENGTH
                and _one(dilation) == 1 and op <= p):
            s, k, length = _one(stride), weight.shape[-1], input.shape[-1]
            # The whole output without padding, by overlap-add of the blocks...
            full = None
            for i0 in range(0, length, BLOCK):
                i1 = min(length, i0 + BLOCK)
                y = conv_transpose1d(input[..., i0:i1], weight, None, s, 0, 0, groups, 1)
                if full is None:
                    full = y.new_zeros(y.shape[0], y.shape[1], (length - 1) * s + k)
                full[..., i0 * s:i0 * s + y.shape[-1]] += y
            # ...then the padding, which for a transposed convolution crops the output
            out_len = (length - 1) * s - 2 * p + (k - 1) + op + 1
            out = full[..., p:p + out_len]
            if bias is not None:
                out = out + bias.view(1, -1, 1)
            return out
        return conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)

    F.conv1d = conv1d_blocks
    F.conv_transpose1d = conv_transpose1d_blocks
