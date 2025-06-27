# MIT License
#
# Copyright (c) 2020 Phil Wang
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

# Modifications copyright (c) 2025 Humanome Lab Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn


from einops import rearrange, repeat


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


def rotate_every_two(x):
    # rotary positional embedding helpers
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(q, k, sinu_pos):
    # rotary positional embedding helpers
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


# sinusoidal positional embeddings

class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        gene2vec_weight = np.load('../data/gene2vec_16906.npy')
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        print("t", t)
        return self.emb(t)


class Gene2VecPositionalEmbeddingIdx(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        gene2vec_weight = np.load('/nfs_beijing/minsheng/scbig/data/biomap_gene2vec.npy')
        gene2vec_weight = np.concatenate((np.zeros((1, gene2vec_weight.shape[1])),gene2vec_weight),axis=0)
        gene2vec_weight = gene2vec_weight[:, :dim]
        gene2vec_weight = torch.Tensor(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze=False)

    def forward(self, x):
        return self.emb(x)


class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        gene2vec_weight = np.random.rand(max_seq_len - 1, dim)
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze=False)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class RandomPositionalEmbeddingIdx(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        gene2vec_weight = np.random.rand(max_seq_len, dim)
        gene2vec_weight = np.concatenate((np.zeros((1, gene2vec_weight.shape[1])),gene2vec_weight),axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight, freeze=False)

    def forward(self, x):
        return self.emb(x)
