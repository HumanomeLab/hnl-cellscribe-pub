# Modifications copyright (c) 2025 Humanome Lab Inc.
# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
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

import argparse
import yaml
import torch
import random
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pickle as pkl


def argparser() -> dict:
    parser = argparse.ArgumentParser(description="CellScribe")
    parser.add_argument('config', type=str, help='config file path')

    return parser.parse_args()


def to_cellscribe_gene_expr_df(
    raw_expr_df: pd.DataFrame,
    gene_dict: dict[str, int],
):
    """CellScribeの遺伝子リストに合わせたデータに整形.

    Parameters
    ----------
    raw_expr_df : pd.DataFrame
        細胞ごとの遺伝子発現量データ。
        列名は遺伝子名、行名は細胞名。
        例:
                gene1      gene2       gene3     gene4      gene5
        cell_1  10
        cell_2  10
        cell_3  10

    gene_dict : dict[str, int]
        CellScribeの遺伝子名とそのインデックスの辞書。

    Return
    -------
    processed_expr_df : pd.DataFrame
        CellScribeの事前学習に使った遺伝子リストに合わせた遺伝子発現量データ。
        例:
                0       1       2        3        4
        cell_1  10
        cell_2  0
        cell_3  1


    """
    ref_genes = list(gene_dict.keys())

    # CellScribeの遺伝子リストに含まれる遺伝子名を取得
    included_genes = [gene for gene in raw_expr_df.columns.tolist() if gene in ref_genes]

    # CellScribeの遺伝子リストに含まれる遺伝子の発現量を0で初期化し
    processed_expr_df = pd.DataFrame(
        np.zeros((len(raw_expr_df), len(ref_genes))),
        columns=ref_genes,
        index=raw_expr_df.index,
    )

    # raw_expr_dfの遺伝子発現量を割り当て
    for gene in included_genes:
        processed_expr_df[gene] = raw_expr_df[gene]

    # 遺伝子名をインデックスに変換
    return processed_expr_df.rename(columns=gene_dict)


def gather_encoder_data(
    expr: list[float],
    gene_ids: list[int],
    device: str
) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encoder用データを作成する.

    Parameters
    ----------
    expr : (ST + n_genes) list[float]
        ST + 全遺伝子の遺伝子発現量
    gene_ids : (ST + n_genes) list[int]
        ST + 全遺伝子のインデックス
    device : str
        デバイス名

    Returns
    -------
    expression : (1, ST + n_nonzero_genes + padding) torch.Tensor
        非ゼロ値の遺伝子発現量
    gene_ids  : (1, ST + n_nonzero_genes + padding) torch.Tensor
        非ゼロ値の遺伝子インデックス
    encoder_pad_mask : (1, ST + n_nonzero_genes + padding) torch.Tensor
        パディング位置がTrueのテンソル(パディングはしないため全てFalse)

    """
    nonzero_expr = [e for e in expr if e > 0]
    nonzero_expr_gene_ids = [id for i, id in enumerate(gene_ids) if expr[i] > 0]

    return {
        "expression": torch.tensor(nonzero_expr, dtype=torch.float32).unsqueeze(0).to(device),
        "gene_ids": torch.tensor(nonzero_expr_gene_ids).unsqueeze(0).to(device),
        "encoder_pad_mask": None,
    }


def gather_decoder_data(
    expr: list[float],
    gene_ids: list[int],
    device: str
) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decoder用データを作成する.

    Parameters
    ----------
    expr : (n_genes) list[float]
        ゼロ値を含む遺伝子発現量
    gene_ids : (n_genes) list[int]
        遺伝子のインデックス
    device : str
        デバイス名

    Returns
    -------
    expression : (batch_size, n_genes) torch.Tensor
        全遺伝子の遺伝子発現量(log2対数変換済み)
    gene_ids : (batch_size, ST + n_genes) torch.Tensor
        全遺伝子の遺伝子ID(各セルごとに [gene_id_1, ..., gene_id_n] の形で格納)
    encoder_pad_mask : (batch_size, ST + n_nonzero_genes + padding) torch.Tensor
        パディング位置がTrueのテンソル(パディングはしないため全てFalse)
    masked_mask : (batch_size, n_genes) torch.Tensor
        マスク対象となった遺伝子IDの位置が1、それ以外が0
    encoder2full_mask : (batch_size, n_genes) torch.Tensor
        全遺伝子のうち、エンコーダーに入力された遺伝子の位置を示すマスク
    encoder_st_pa_mask : (batch_size, ST + n_nonzero_genes + padding) torch.Tensor
        S_token, T_token, pad_token の位置がTrue、それ以外がFalse
        ただし、実質パディングはない

    """
    return {
        "expression": torch.tensor(expr, dtype=torch.float32).unsqueeze(0).to(device),
        "gene_ids": torch.tensor(gene_ids, dtype=torch.long).unsqueeze(0).to(device),
        "encoder_pad_mask": torch.tensor([0] * len(expr), dtype=torch.long).unsqueeze(0).to(device),
        "masked_mask": None,
        "encoder2full_mask": torch.tensor(np.array(expr) > 0, dtype=torch.long).unsqueeze(0).to(device),
        "encoder_st_pa_mask": torch.tensor([1, 1] + [0] * sum(np.array(expr) > 0), dtype=torch.long).unsqueeze(0).to(device),
    }


def main():
    args = argparser()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    random.seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルの読み込み
    with open(config["output"]["pkl_path"], "rb") as f:
        model = pkl.load(f)

    model.to(device)
    model.eval()

    # 遺伝子発現量の読み込み
    if config["input"]["data_path"][-7:] == 'parquet':
        raw_expr_df = pd.read_parquet(config["input"]["data_path"])
    elif config["input"]["data_path"][-3:] == 'csv':
        raw_expr_df = pd.read_csv(config["input"]["data_path"], index_col=0)
    else:
        raise ValueError('data_path must be parquet, csv, npz, h5ad or npy')

    # CellScribeの遺伝子リストで整形
    with open(config["preprocess"]["gene_dict_path"], "r") as f:
        gene_dict = json.load(f)

    expr_df = to_cellscribe_gene_expr_df(raw_expr_df, gene_dict)

    if config["demo"]:
        expr_df = expr_df.head(10)

    S_token = config["preprocess"]["S_token"]
    T_token = config["preprocess"]["T_token"]

    #Inference
    n_genes = len(gene_dict)   # 遺伝子数
    embeddings = []  # 細胞の埋め込みを格納するリスト
    for i in tqdm(range(expr_df.shape[0])):
        expr = expr_df.iloc[i, :] # pd.Series

        if config["input"]["pre_normalized"] == 'F':
            expr = np.log2(expr + 1)
            T = expr.sum()
        elif config["input"]["pre_normalized"] == 'T':
            T = expr.sum()
        elif config["input"]["pre_normalized"] == 'A':
            expr = expr[:-1]
            T = expr[-1]
        else:
            raise ValueError('pre_normalized must be T,F or A')

        # select resolution
        symbol = config["output"]["tgthighres"][0]
        resolution = float(config["output"]["tgthighres"][1:])
        if symbol == 'f':
            S = T * resolution
        elif symbol == 'a':
            S = T + resolution
        elif symbol == 't':
            S = resolution
        else:
            raise ValueError('tgthighres must be start with f, a or t')

        st_expr = [S, T] + expr.tolist() # list
        st_gene_ids = [S_token, T_token] + list(np.arange(n_genes)) # list

        with torch.no_grad():
            encoder_data = gather_encoder_data(st_expr, st_gene_ids, device)
            decoder_data = gather_decoder_data(expr, st_gene_ids, device)
            encoder_decoder_data = {
                "encoder": encoder_data,
                "decoder": decoder_data
            }

            # cell embedding
            if config["output"]["output_type"]=='cell':
                out = model.cell_embedding(encoder_data, pool_type=config["output"]["pool_type"])

            # gene embedding
            elif config["output"]["output_type"]=='gene':
                out = model.gene_embedding(encoder_decoder_data)

            # gene expression
            elif config["output"]["output_type"]=='gene_expression':
                out = model.gene_expression(encoder_decoder_data, n_genes)
                # log2変換されているであろうデータを元に戻す
                out = torch.exp2(out)

            else:
                raise ValueError('output_type must be cell or gene or gene_batch or gene_expression')

            embeddings.append(out.detach().cpu().numpy())

    embeddings = np.squeeze(np.array(embeddings))
    np.save(config["output"]["save_npy_path"], embeddings)
    pd.DataFrame(embeddings).to_csv(config["output"]["save_csv_path"], index=False)


if __name__ == "__main__":
    main()
