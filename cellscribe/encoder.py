# Modifications copyright (c) 2025 Humanome Lab Inc.
# Copyright (c) 2023 BioMap (Beijing) Intelligence Technology Limited
#
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

import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        nheads: int,
        ff_mult: int
    ):
        """
            Parameter
            ---------
                d_model: int
                n_layers: int
                nheads: int
                ff_mult: int
                norm_first: bool = False
        """
        super().__init__()
        module_list = []
        for i in range(n_layers):
            module_list.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nheads,
                    dim_feedforward=d_model * ff_mult,
                    batch_first=True,
                    norm_first=False,
                    activation="gelu",
                    layer_norm_eps=1e-05
                )
            )

        self.transformer_layers = nn.ModuleList(module_list)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        x = self.norm(x)

        return x


