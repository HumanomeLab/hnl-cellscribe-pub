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

from typing import Dict
import torch.nn as nn
import torch


class ExpressionAutoDiscretization(nn.Module):
    def __init__(
        self,
        d_model,
        bin_num,
        bin_alpha,
        mixed_precision,
        local_rank
    ):
        """
            reference :
                code : scFoundation/AutoDiscretizationEmbedding2
                paper :
                    scFoundation
                        The embedding module converted continuous gene
                        expression scalars into learnable high-dimensional vectors ensuring
                        full retention of raw expression values, which was a notable improve-
                        ment over the discretized values used in previous models
                    xTrimoGene
                        auto-discretization process involves a random look-up table T defined in R(d x b)
                        In this representation, d refers to the embedding dimension, while b
                        is the number of bins with a default value of 100.
                        The transformation starts by applying a linear layer to the expression value,
                        given by v1 = v · w1,
                        where w1 represents the weight vector. The resulting
                        v1 is then subjected to a leaky ReLU activation, resulting in v2 = Leaky_ReLU(v1). Subsequently, a
                        cross-layer projection is applied, represented by v3 = w2 · v2 + α · v2. Here, w2 denotes the weight
                        vector, and α is a scaling mixture factor. Next, the bin weights of v3 are normalized using the softmax
                        function, resulting in v4 = softmax(v3). Finally, the transformed value is represented as a weighted
                        combination of individual embeddings from the look-up table, given by e = T · v4. It’s important to
                        note that the weights in this combination serve as learnable parameters

        """
        super().__init__()
        self.d_model = d_model
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.local_rank = local_rank
        self.mixed_precision = mixed_precision

        # MLP for gene expression -> bin_weights
        self.layer1 = nn.Linear(1, self.bin_num)
        self.layer2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)

        # bin embedding
        self.emb = nn.Embedding(self.bin_num, self.d_model)
        self.bin_num_idx = torch.tensor(range(self.bin_num)).to(self.local_rank)

        # mask and pad token embedding
        if self.mixed_precision:
            self.token_emb_dtype = torch.bfloat16
        else:
            self.token_emb_dtype = torch.long

        self.tensor0 = torch.tensor(0, dtype=torch.long).to(self.local_rank)

        # mask token embedding
        self.emb_mask = nn.Embedding(1, self.d_model)

        # pad token embedding
        self.emb_pad = nn.Embedding(1, self.d_model)

    def forward(self, gene_expression, pad_mask=None, masked_mask=None):
        """
            Parameter
            ----------
                gene_expression: torch.Tensor

                pad_mask: torch.Tensor
                    mask of padded value in nonzero gene_expression input

                masked_gene_ids:
                    gene ids of masked value in full length of gene_expression

        """
        # weights shape  (batch_size, seq_len, n_bin)
        weights = self.get_bin_weights(gene_expression.unsqueeze(2))

        # bin_emb.shape (n_bin, d_model)
        bin_emb = self.emb(self.bin_num_idx)

        # embedding shape (batch_size, seq_len, d_model)
        embeddings = torch.matmul(weights, bin_emb)

        if pad_mask is not None:
            # https://pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero
            pad_idx = pad_mask.nonzero()

            pad_token_emb = self.emb_pad(self.tensor0).type(self.token_emb_dtype)
            # when mixed precision, embeddings: bfloat16, pad_token_emb: float32
            embeddings[pad_idx[:, 0], pad_idx[:, 1], :] = pad_token_emb.repeat(pad_idx.shape[0], 1)

        if masked_mask is not None:
            # https://pytorch.org/docs/stable/generated/torch.nonzero.html#torch.nonzero
            masked_idx = masked_mask.nonzero()
            masked_token_emb = self.emb_mask(self.tensor0).type(self.token_emb_dtype)
            # when mixed precision, embeddings: bfloat16, masked_token_emb: float32
            embeddings[masked_idx[:, 0], masked_idx[:, 1], :] = masked_token_emb.repeat(masked_idx.shape[0], 1)

        return embeddings

    def get_bin_weights(self, x):
        x = self.layer1(x)
        x = self.LeakyReLU(x)
        x_crosslayer = self.layer2(x)
        x = self.bin_alpha * x + x_crosslayer
        weights = self.Softmax(x)

        return weights


class EmbeddingModule(nn.Module):
    def __init__(self, arch_config, mixed_precision, local_rank):
        super().__init__()
        self.encoder_d_model = arch_config["encoder"]["d_model"]
        self.decoder_d_model = arch_config["decoder"]["d_model"]

        self.pad_token = arch_config["pad_token"]

        self.mixed_precision = mixed_precision
        self.local_rank = local_rank

        max_seq_len = arch_config["max_seq_len"]
        self.gene_emb = nn.Embedding(max_seq_len + 1, self.encoder_d_model, padding_idx=self.pad_token)
        self.expr_emb = ExpressionAutoDiscretization(
            d_model=self.encoder_d_model,
            bin_num=100,
            bin_alpha=1.0,
            mixed_precision=mixed_precision,
            local_rank=local_rank
        )

        self.decoder_dim_projection = nn.Linear(self.encoder_d_model, self.decoder_d_model, bias=True)

    def forward(
        self,
        mode=None,
        data=None,
        encoder_output=None
    ):
        """Forward pass of the EmbeddingModule.

        This method handles both the encoder and decoder modes for gene expression embedding.

        Parameters
        ----------
        mode : str
            "encoder" or "decoder"
        data : Dict
            Data for the embedding module, which includes:
            for "encoder" mode:
            - "expression": torch.Tensor (batch_size, ST + input_length)
                - ST + 非ゼロ遺伝子のlog2で対数変換した遺伝子発現量 + パディング
            - "gene_ids": torch.Tensor (batch_size, ST + input_length)
                - 「ST + 非ゼロ遺伝子のlog2で対数変換した遺伝子発現量 + パディング」に対する遺伝子ID
            - "encoder_pad_mask": torch.Tensor (batch_size, ST + input_length)
                - 「ST + 非ゼロ遺伝子の遺伝子発現量 + パディング」に対するパディングマスク

            for "decoder" mode:
            - "expression": torch.Tensor (batch_size, n_genes)
                - 全遺伝子のlog2で対数変換した遺伝子発現量
            - "gene_ids": torch.Tensor (batch_size, ST + n_genes)
                - STトークン+全遺伝子の遺伝子ID
            - "encoder_pad_mask": torch.Tensor (batch_size, ST + input_length)
                - 「ST + 非ゼロ遺伝子の遺伝子発現量 + パディング」に対するパディングマスク
            - "masked_mask": torch.Tensor (batch_size, n_genes)
                - 全遺伝子の遺伝子発現量のマスクされた位置に対するマスク
            - "encoder2full_mask": torch.Tensor (batch_size, n_genes)
                - 全遺伝子のうち、エンコーダーに入力された遺伝子の位置を示すマスク
            - "encoder_st_pa_mask": torch.Tensor (batch_size, ST + input_length)
                - 「ST + 非ゼロ遺伝子の遺伝子発現量 + パディング」の配列に対するSTとパディングのマスク

            for details on the data structure, refer to the docstring of PreprocessModule.

        Returns
        -------
        torch.Tensor
            Embedding output based on the mode:
            - In "encoder" mode, returns the gene expression embeddings.
            - In "decoder" mode, returns the full expression embeddings with encoder output filled in.

        Raises
        ------
        ValueError
            If an invalid mode is provided.

        Notes
        -----
        for decoder mode,
        - full_embedding[encoder2full_mask]
            - 全遺伝子の埋め込みのうち、エンコーダーに入力された遺伝子の位置に対してエンコーダー出力を埋め込む
        - encoder_output[~encoder_st_pa_mask]
            - エンコーダー出力のうち、STトークンとパディングを除いた位置に対して埋め込む

        """
        expression = data["expression"]
        gene_ids = data["gene_ids"]

        if mode == "encoder":
            nonzero_embedding = self.gene_emb(gene_ids)
            nonzero_embedding += self.expr_emb(
                gene_expression=expression,
                pad_mask=data["encoder_pad_mask"]
            )
            return nonzero_embedding

        elif mode == "decoder":
            masked_mask = data["masked_mask"]
            # extract full expression embedding from full expression data
            full_embedding = self.expr_emb(expression, masked_mask=masked_mask)

            encoder2full_mask = data["encoder2full_mask"].unsqueeze(-1).repeat(1, 1, self.encoder_d_model).bool()
            encoder_st_pa_mask = data["encoder_st_pa_mask"].unsqueeze(-1).repeat(1, 1, self.encoder_d_model).bool()

            if self.mixed_precision:
                # encoder_output : float32 , full_embedding dtype torch.bfloat16
                full_embedding[encoder2full_mask] = encoder_output[~encoder_st_pa_mask].type(torch.bfloat16)
            else:
                full_embedding[encoder2full_mask] = encoder_output[~encoder_st_pa_mask]

            encoded_st_embedding = encoder_output[:, :2, :]
            full_embedding = torch.concat([encoded_st_embedding, full_embedding], axis=1)

            full_embedding += self.gene_emb(gene_ids)

            # project encoder dimension(d_model) to decoder dimension(d_model)
            full_embedding = self.decoder_dim_projection(full_embedding)

            return full_embedding

        else:
            raise ValueError(f"Wrong mode {mode} is used in EmbeddingModule")
