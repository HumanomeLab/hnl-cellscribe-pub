# Copyright 2025 Humanome Lab Inc.
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

import torch
import torch.nn as nn

from .decoder.performer import Performer
from .encoder import Transformer
from .embedding_module import EmbeddingModule


class CellScribe(nn.Module):
    """CellScribeモデル."""
    def __init__(
        self,
        arch_config: dict,
        mixed_precision: bool,
        local_rank: str,
    ) -> None:
        """初期化.

        Parameters
        ----------
        arch_config : dict
            モデルのアーキテクチャに関する設定情報. yamlでの設定方法.
            arch:
                max_seq_len: 20010
                pad_token: 20005
                encoder:
                    d_model: 1280
                    n_layers: 12
                    nheads: 16
                    ff_mult: 4
                decoder:
                    d_model: 1024
                    n_layers: 6
                    nheads: 16
                    dim_head: 32
        mixed_precision : bool
            混合精度を使う場合、EmbeddingModuleの中で精度を指定する必要があり、それを伝えるため
        local_rank : str
            デバイスの設定

        """
        super().__init__()
        self.embedding_module = EmbeddingModule(
            arch_config=arch_config,
            mixed_precision=mixed_precision,
            local_rank=local_rank
        )
        self.encoder = Transformer(**arch_config["encoder"])
        self.decoder = Performer(**arch_config["decoder"])

    def forward(self, data: dict) -> torch.Tensor:
        """順伝播.

        Parameters
        ----------
        data : dict
            詳細はpreprocess.pyのPreprocessModule
            data["decoder"] = {
                "expression": log_expression.to(self.gpu_id),
                "gene_ids": st_decoder_gene_ids.long().to(self.gpu_id),
                "encoder2full_mask": encoder2full_mask.long().to(self.gpu_id),
                "encoder_st_pa_mask": encoder_st_pa_mask.long().to(self.gpu_id),
                "masked_mask": masked_mask.long().to(self.gpu_id)
            }
            data["encoder"] = {
                "expression": st_pa_encoder_expr.to(self.gpu_id),
                "gene_ids": st_pa_encoder_gene_ids.long().to(self.gpu_id),
                "encoder_pad_mask": encoder_pad_mask.long().to(self.gpu_id)
            }

        Returns
        -------
        x : torch.Tensor
            予測結果.

        """
        x = self.embedding_module(mode="encoder", data=data["encoder"])
        x = self.encoder(x)

        x = self.embedding_module(mode="decoder", data=data["decoder"], encoder_output=x)
        x = self.decoder(x)

        return x

    def cell_embedding(
        self,
        encoder_data: dict,
        pool_type: str = "all",
    ) -> torch.Tensor:
        """細胞埋め込みの取得.

        Parameters
        ----------
        encoder_data : dict
            遺伝子発現量データ. データの後ろにSTがこの順番で結合されている.
            詳細はpreprocess.pyのPreprocessModule
            {
                "expression": st_pa_encoder_expr.to(self.gpu_id),
                "gene_ids": st_pa_encoder_gene_ids.long().to(self.gpu_id),
                "encoder_pad_mask": encoder_pad_mask.long().to(self.gpu_id)
            }

        pool_type : str
            'all' or 'max'

        Returns
        -------
        torch.Tensor

        Notes
        -----
        STは遺伝子発現量データの前に結合している

        Reference
        ---------
        https://github.com/biomap-research/scFoundation/blob/main/model/get_embedding.py

        """

        x = self.embedding_module(mode="encoder", data=encoder_data)
        x = self.encoder(x)

        if pool_type == 'all':
            emb1 = x[:, 0, :]  # S
            emb2 = x[:, 1, :]  # T
            emb3, _ = torch.max(x[:, 2:, :], dim=1)
            emb4 = torch.mean(x[:, 2:, :], dim=1)
            emb = torch.concat([emb1, emb2, emb3, emb4], axis=1)

        elif pool_type == 'max':
            emb, _ = torch.max(x, dim=1)

        else:
            raise NotImplementedError

        return emb

    def gene_embedding(self, data: dict) -> torch.Tensor:
        """遺伝子の埋め込み.

        Parameters
        ----------
        data : dict
            詳細はpreprocess.pyのPreprocessModule
            data["decoder"] = {
                "expression": log_expression.to(self.gpu_id),
                "gene_ids": st_decoder_gene_ids.long().to(self.gpu_id),
                "encoder2full_mask": encoder2full_mask.long().to(self.gpu_id),
                "encoder_st_pa_mask": encoder_st_pa_mask.long().to(self.gpu_id),
                "masked_mask": masked_mask.long().to(self.gpu_id)
            }
            data["encoder"] = {
                "expression": st_pa_encoder_expr.to(self.gpu_id),
                "gene_ids": st_pa_encoder_gene_ids.long().to(self.gpu_id),
                "encoder_pad_mask": encoder_pad_mask.long().to(self.gpu_id)
            }

        Returns
        -------
        x : torch.Tensor
            遺伝子の埋め込み.

        """
        x = self.embedding_module(mode="encoder", data=data["encoder"])
        x = self.encoder(x)

        x = self.embedding_module(mode="decoder", data=data["decoder"], encoder_output=x)
        x, attn_weights = self.decoder(x, output_attentions=True)
        print(attn_weights.shape)

        return attn_weights[:, 2:, :].contiguous()
        # x = self.forward(data=data, )
        # return x[:, 2:, :].contiguous()

    def gene_expression(self, data: dict, n_genes: int) -> torch.Tensor:
        """遺伝子発現量.

        Parameters
        ----------
        data : dict
            詳細はpreprocess.pyのPreprocessModule
            data["decoder"] = {
                "expression": log_expression.to(self.gpu_id),
                "gene_ids": st_decoder_gene_ids.long().to(self.gpu_id),
                "encoder2full_mask": encoder2full_mask.long().to(self.gpu_id),
                "encoder_st_pa_mask": encoder_st_pa_mask.long().to(self.gpu_id),
                "masked_mask": masked_mask.long().to(self.gpu_id)
            }
            data["encoder"] = {
                "expression": st_pa_encoder_expr.to(self.gpu_id),
                "gene_ids": st_pa_encoder_gene_ids.long().to(self.gpu_id),
                "encoder_pad_mask": encoder_pad_mask.long().to(self.gpu_id)
            }

        Returns
        -------
        x : torch.Tensor
            予測結果.
        """
        x = self.forward(data=data)
        return x[:, 2:].contiguous()
