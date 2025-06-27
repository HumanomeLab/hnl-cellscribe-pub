# CellScribe -シングルセル遺伝子発現量基盤モデル-

CellScribeは、細胞の機能や状態をあらわす遺伝子発現量を大規模に学習することで、医薬品などが細胞に及ぼす影響を効率的に予測する基盤モデルです。約9億細胞のデータを利用し、3億パラメータの基盤モデルを学習しました。CellScribeは世界中の細胞状態を示す「地図」となり、効率的な薬効予測や評価を実現します。モデルの概要、学習に利用したデータなどは[弊社のリリース](https://humanome.jp/press/detail/geniac-results/) を御覧ください。本レポジトリは、CellScribeをテストし、CellScribeの入出力を知っていただくために作成したものとなっています。

```
本成果は、経済産業省とNEDOが実施する、国内の生成AIの開発力強化を目的としたプロジェクト「GENIAC（Generative AI Accelerator Challenge）」の支援を受けて得られたものです。
```

## 公開用サンプルモデル

公開用サンプルモデルは以下のURLからダウンロードできます。<br>
https://bit.ly/CellScribe

公開しているモデルは学習過程の10kステップ時点のものです。継続的な学習によってさらに精度が向上したモデルのご利用については、お手数ですが個別にお問い合わせください。

## 📊 入出力仕様

### 入力形式
- **ファイル形式**: CSVまたはparquet形式（行：細胞、列：遺伝子シンボル）
- **遺伝子名**: HGNCの遺伝子名（ヒト）
- **発現量**: シングルセルRNA-seqの生カウントデータ

### 出力形式
以下のいずれかを返します。
- **細胞の埋め込みベクトル**
- **遺伝子の埋め込みベクトル**
- **遺伝子発現量ベクトル**

## 🚀 テストコードの実行方法

### 前提条件

- `torch >= 2.4.1`
- `pytorch-fast-transformers==0.4.0`
- 実行環境
  - AWSの場合、g4dn.xlarge相当以上を推奨


### モデル、データ、スクリプトの準備

1. リポジトリをクローン

```bash
git clone https://github.com/HumanomeLab/hnl-cellscribe-pub
cd hnl-cellscribe-pub
```

2. 事前学習済みモデルをダウンロード

上述の公開用サンプルモデルをダウンロードし、`hnl-cellscribe-pub`直下に保存する。モデルファイルのサイズは約1.2GBである。
モデルのパスは`config.yaml`で変更できる。


3. デモデータ

遺伝子発現量デモデータは`hnl-cellscribe-pub/demo_data.csv`を使用する。
10細胞分の遺伝子発現量データが用意されている。遺伝子発現量デモデータのパスは`config.yaml`で変更できる。



### 設定ファイルの詳細
実行時の設定はconfig.yaml に記載されている。詳細は以下のとおりである。

- demo: Trueにすると入力データのうち、最初の10細胞の結果を出力する
- input
  - data_path: 遺伝子発現量データのファイルを指定する。csvまたはparquetを指定できる
  - pre_normalized: Fは標準化未実施、Tは標準化実施済み、Aは標準化+log2(x+1)変換実施済み
- output
  - output_type: cellは細胞の埋め込み、geneは遺伝子の埋め込み、gene_expressionは遺伝子発現量の予測結果
  - pool_type: allはmaxとmeanをとる、maxはmaxのみ
  - tgthighres: 解像度を指定する。t10000, f2, a100などで指定できる。
    t始まりの場合は、予測する遺伝子発現量の総カウント数を指定する。f始まりの場合は、fold changeの倍率を指定する。
    a始まりの場合は、追加するリード数を指定する。
  - is_mixed_precision: 混合精度演算にするかどうかを指定する(推論時はFalseで良い)
  - save_path: 出力結果を保存するファイル名を指定する
  - pkl_path: モデルのファイル名を指定する

config.yaml の例：

```yaml
demo: True # True or False

input:
  data_path: demo_data.csv
  pre_normalized: F # F, T or A

output:
  output_type: gene_expression # cell, gene or gene_expression
  pool_type: all # all or max
  tgthighres: f1 # t10000, f5 or a5
  is_mixed_precision: False
  save_path: embedding.npy
  pkl_path: CellScribe-300M-step10k.pkl
```

### 推論スクリプトを実行する

```bash
python inference.py config.yaml
```

## ライセンス

- 本開発には、[scFoundation](https://github.com/biomap-research/scFoundation)  のソースコードを参考・改変いたしました。
- 本レポジトリのソースコードのライセンスは、各ファイルに記載の内容に従います。

