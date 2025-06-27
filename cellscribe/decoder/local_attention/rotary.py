# Copyright (c) Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn, einsum
from torch.nn import Module
from torch.amp import autocast

from einops import rearrange

def exists(val):
    return val is not None

class SinusoidalEmbeddings(Module):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    @autocast('cuda', enabled = False)
    def forward(self, x):
        seq_len, device = x.shape[-2], x.device

        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale ** -1

    if scale.ndim == 2:
        scale = scale[-q_len:, :]

    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k
