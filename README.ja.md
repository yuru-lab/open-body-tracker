# open-body-tracker

[English](README.md)

**RGBカメラ1台で全身の3D姿勢を推定する**軽量モデルです。1人の23点（身体＋足）を、画像上の2Dとメートル単位の3Dで出力します。一般的なGPUならリアルタイムで動き、WebGPUでブラウザでも動きます。

**[ブラウザデモを試す](https://yuru-lab.github.io/open-body-tracker/)**：処理はすべて手元のPCで行い、映像は外部に送りません。

![ASPset-510の動画でのobt-body-v1：映像に2D骨格、横に3Dの予測と正解](docs/sample-aspset.gif)

<sub>[ASPset-510](https://github.com/anibali/aspset-510)（CC0）の、学習に一度も使っていない人の映像です。
グレーはASPsetの多視点カメラから作った正解の3D、色はobt-body-v1（最初の版。カメラ1台、1フレームずつ独立、平滑化なし）。
決まった規則（正面カメラ、誤差が評価平均の±8mm以内、動きがいちばん多い）で選んだ1本で、
誤差はこのクリップで64mm（PA-MPJPE）、評価全体の平均は67.9mmです。しゃがむ・背中向きなどの難しい姿勢では約86mmになります。</sub>

- 重み：[GitHub Releases](https://github.com/yuru-lab/open-body-tracker/releases) · [Hugging Face](https://huggingface.co/Yurup/open-body-tracker)
- モデルの版と評価：[docs/MODELS.md](docs/MODELS.md) · 精度・学習データ・制限：[MODEL_CARD.md](MODEL_CARD.md) · [DATA.md](DATA.md)（英語）

| | **obt-body-v2**（標準） | obt-body-v1 |
|---|---:|---:|
| COCO val2017 キーポイントAP（2D、正解の人物枠、clean） | 71.6 | 71.7 |
| ASPset-510 学習に使っていない3人 PA-MPJPE（3D、身体12点）¹ | 53.7 mm | 67.9 mm |
| Harmony4D test（別の場所、2人が密着）PA-MPJPE | 71.3 mm | 76.0 mm |
| モデルの大きさ | 9,560万パラメータ · 191 MB（fp16 ONNX） | 同じ |
| 速度（1人） | RTX 3060 Tiで約6〜10 ms | 同じ |

<sub>¹ v2はASPsetと同じスタジオで撮った別の7人でも学習しているため、ASPsetでの改善は普段の環境より大きく出ます。
別の環境での目安はHarmony4Dの値です。区間や反転テストの詳細は [docs/MODELS.md](docs/MODELS.md)（英語）。</sub>

## しくみ

```
カメラ映像 ──► YOLOX-nano / tiny ──► 人物の枠 ──► 256×192の切り抜き ──► obt-body ──► 23点の (x, y) と (X, Y, Z)
              （人物検出）            ▲                                       │
                                     └──────── 前のフレームの姿勢から枠を作る ┘  （検出は1秒ごと）
```

## はじめかた

[mise](https://mise.jdx.dev/)（Python・uv・Nodeを入れてくれます）か、Python 3.12と[uv](https://docs.astral.sh/uv/)が必要です。

```sh
git clone https://github.com/yuru-lab/open-body-tracker
cd open-body-tracker
mise install          # Python 3.12、uv、Node
mise run setup        # uv sync

mise run demo         # Webカメラのデモ（初回にモデルを ./models へダウンロード）
uv run obt-demo --image photo.jpg            # photo.pose.jpg を書き出す
uv run obt-demo --detector tiny              # 誤検出が少ない（少し遅い）
uv run obt-demo --flip                       # 反転テスト：少し正確で揺れが少ない（約2倍遅い）
uv run obt-demo --model-version v1           # 前の版のモデル
uv sync --extra gpu && uv run obt-demo --gpu # CUDAを使う
```

ブラウザデモを手元で動かす：`mise run web` を実行して http://localhost:8000 を開きます。`web/models/` にONNXを置くとダウンロードを省けます。

人物検出は既定がYOLOX-nano（速い）です。椅子などを人と間違えるときはYOLOX-tinyに切り替えてください。

## コードから使う

```python
import cv2
from open_body_tracker.detector import PersonDetector
from open_body_tracker.hub import fetch, model_file
from open_body_tracker.pipeline import PosePipeline

pipeline = PosePipeline(fetch(model_file("v2")), PersonDetector(fetch("yolox_nano.onnx")))  # flip=True で反転テスト
pose = pipeline(cv2.imread("photo.jpg")[:, :, ::-1].copy())  # RGB
# pose["xy"]    (23, 2) 画像のピクセル座標
# pose["score"] (23,)   自信度
# pose["xyz"]   (23, 3) メートル、腰の中心が原点、カメラの軸（x右・y下・z奥）
```

点の順番は `open_body_tracker/keypoints.py` にあります（COCOの17点＋左右の親指・小指・かかと）。

## 構成

```
src/open_body_tracker/   モデル（PyTorch）、前処理、人物検出、追跡、デモ
web/                     ブラウザデモ（ONNX Runtime Web、WebGPU）
scripts/                 checkpointの変換とONNXの書き出し
docs/MODELS.md           モデルの版と評価
docs/REPRODUCE.md        学習の手順と再現
```

## 再現性

学習コードは [`training/`](training/) にあり、SAM 3D Bodyの教師ラベルを含むラベル一式はHugging Faceの [`Yurup/open-body-tracker-labels`](https://huggingface.co/datasets/Yurup/open-body-tracker-labels) にあります（SAM 3D Bodyを自分で動かす必要はありません）。このコードとラベルだけで最初から学習し直し、公開モデルと同じ精度になることを確かめました（COCO AP 71.6 対 71.7、ASPset PA-MPJPE 67.6mm 対 67.9mm）。obt-body-v2はASPset-510（CC0）の切り抜きを作る手順が1つ増え、学習のバッチと最初の損失が公開モデルの学習と完全に一致することを確かめています。手順と詳細は [docs/REPRODUCE.md](docs/REPRODUCE.md)（英語）にあります。

## ライセンス

- コード：[Apache-2.0](LICENSE)
- モデルの重みと教師ラベル：[SAM License](licenses/SAM-LICENSE.txt)（SAM 3D Bodyの出力で学習したため）。再配布するときは契約書を添付してください。用途の制限（軍事など）も引き継がれます。
- obt-body-v2はASPset-510（CC0）でも学習しています
- 人物検出：YOLOX（Apache-2.0、Megvii）
- 第三者の表示：[NOTICE](NOTICE)

研究用のソフトウェアです。医療機器や計測機器ではありません。
