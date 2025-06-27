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

import torch
import torch.nn as nn
from functools import partial

from .attention import FastAttention, SelfAttention
from .blocks import Chunk, FeedForward, ReZero, PreLayerNorm, PreScaleNorm
from .reversible import ReversibleSequence, SequentialSequence
from .utils import to_tuple_if_not, get_module_device, find_modules


class Performer(nn.Module):
    def __init__(
        self,
        d_model,
        n_layers,
        nheads,
        dim_head,
        ff_mult=4,
        ff_chunks=1,
        ff_glu=False,
        ff_dropout=0.0,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        nb_features=None,
        feature_redraw_interval=1000,
        reversible=False,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        attn_dropout=0.0,
        cross_attend=False,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
    ):
        """
            Reference:
                code:
                    scFoundation/Performer
                        which use scBERT/Performer(https://github.com/TencentAILabHealthcare/scBERT/blob/master/performer_pytorch/performer_pytorch.py)
                Components
                    Performer/FastAttention (FAVOR+)

                    local_attention:
                        https://github.com/lucidrains/local-attention

                    Reformer/ReversibleSequence
                    RevTorch/ReversibleSequence
                        https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py


            Parameters
            ----------
                d_model : int
                    Dimension of the model.
                n_layers : int
                    Number of layers.
                heads : int
                    Number of attention heads.
                dim_head : int
                    Dimension of each attention head.
                local_attn_heads : int, default=0
                    Number of local attention heads. The remaining heads are global performer heads.
                local_window_size : int, default=256
                    Window size for local attention.
                causal : bool, default=False
                    If True, the model is autoregressive.
                ff_mult : int, default=4
                    Multiplier for the dimension of intermediate features after attention relative to input features.
                nb_features : int, optional
                    Number of random features. If None, defaults to d * log(d), where d is the dimension of each head.
                feature_redraw_interval : int, default=1000
                    Frequency of redrawing the projection matrix. Higher frequency can slow down training.
                reversible : bool, default=False
                    If True, enables reversible layers, reducing memory usage (from Reformer model).
                ff_chunks : int, default=1
                    Number of chunks for feedforward layer, from Reformer.
                generalized_attention : bool, default=False
                    If True, uses generalized attention instead of softmax approximation.
                kernel_fn : nn.Module, default=nn.ReLU()
                    Kernel function used if generalized attention is enabled. Defaults to ReLU.
                use_scalenorm : bool, default=False
                    If True, applies ScaleNorm, as proposed in "Transformers without Tears." ScaleNorm can substitute LayerNorm.
                use_rezero : bool, default=False
                    If True, applies ReZero (from "ReZero is All You Need") as a LayerNorm substitute. Priority: ScaleNorm > ReZero > LayerNorm.
                ff_glu : bool, default=False
                    If True, uses Gated Linear Units (GLU) variant in the feedforward layer.
                ff_dropout : float, default=0.0
                    Dropout rate for the feedforward layer.
                attn_dropout : float, default=0.0
                    Dropout rate applied after attention.
                cross_attend : bool, default=False
                    If True, enables cross-attention.
                no_projection : bool, default=False
                    If True, skips projection in the attention layer.
                auto_check_redraw : bool, default=True
                    Automatically checks for redrawing projection matrices.
                qkv_bias : bool, default=True
                    If True, adds a bias term to the query, key, and value projections.

        """
        super().__init__()
        local_attn_heads = to_tuple_if_not(local_attn_heads)
        local_attn_heads = local_attn_heads * n_layers if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == n_layers, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= nheads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, d_model)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, d_model)

        module_list = list()
        for _, local_heads in zip(range(n_layers), local_attn_heads):
            module_list.append(nn.ModuleList([
                wrapper_fn(SelfAttention(
                    d_model, causal=causal, heads=nheads, dim_head=dim_head,
                    local_heads=local_heads, local_window_size=local_window_size,
                    nb_features=nb_features, generalized_attention=generalized_attention,
                    kernel_fn=kernel_fn, dropout=attn_dropout, no_projection=no_projection, qkv_bias=qkv_bias)
                ),
                wrapper_fn(Chunk(
                    ff_chunks,
                    FeedForward(d_model, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))
            # if no need cross_attend(decoder), begin next cycle
            if not cross_attend:
                continue
            module_list.append(nn.ModuleList([
                wrapper_fn(SelfAttention(
                    d_model, heads=nheads, dim_head=dim_head,
                    nb_features=nb_features,
                    generalized_attention=generalized_attention,
                    kernel_fn=kernel_fn, dropout=attn_dropout, no_projection=no_projection)
                ),
                wrapper_fn(Chunk(
                    ff_chunks,
                    FeedForward(d_model, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))
        layers = nn.ModuleList(module_list)
        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * n_layers * (2 if cross_attend else 1)  # ((True, False), (True, False), (True, False), (True, False), (True, False), (True, False))
        route_context = ((False, False), (True, False)) * n_layers
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        self.norm = nn.LayerNorm(d_model)
        self.expr_pred_head = nn.Linear(d_model, 1)

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if self.feature_redraw_interval is not None and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)

            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, output_attentions=False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()

        if output_attentions:
            x, attn_weights = self.net(x, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            x = self.expr_pred_head(x)
            return x, attn_weights
        else:
            # x = torch.utils.checkpoint.checkpoint(self.net, x)
            x = self.net(x, output_attentions=output_attentions, **kwargs)
            x = self.norm(x)
            x = self.expr_pred_head(x)
            return x
