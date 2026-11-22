<p align="right"><a href="README.md">English</a> · <b>日本語</b></p>

<h1 align="center">open-body-tracker</h1>

<p align="center">
  RGBカメラ1台で全身の3D姿勢を推定。23点を画像上の2Dとメートル単位の3Dで、リアルタイムに。<br>
  <a href="https://yuru-lab.github.io/open-body-tracker/"><b>ブラウザデモ</b></a> ·
  <a href="https://huggingface.co/Yurup/open-body-tracker">モデル（Hugging Face）</a> ·
  <a href="docs/MODELS.md">評価</a>
</p>

<p align="center"><img src="docs/sample-aspset.gif" alt="映像上の2D骨格と、正解と並べた3Dの予測" width="800"></p>

<p align="center"><sub>学習に一度も使っていない人の映像（<a href="https://github.com/anibali/aspset-510">ASPset-510</a>、CC0）。
グレーが正解の3D、色がobt-body-v0.3.0（カメラ1台、平滑化なし）。<br>このクリップの3D誤差は56mm、全体の平均は53.7mmです。</sub></p>

## モデル

| | **obt-body-v0.3.0** | obt-body-v0.3.0-int8 |
|---|---:|---:|
| 3D誤差・別の環境（Harmony4D、PA-MPJPE） | 71.3 mm | 71.8 mm |
| 2D精度（COCO val2017 AP） | 71.6 | 71.7 |
| 速度（1人） | GPU 6 ms · CPU 103 ms | CPU 46 ms |

INT8版は同じモデルをCPU向けに軽くしたものです。版の一覧やほかの評価、測り方は [docs/MODELS.md](docs/MODELS.md)（英語）にあります。

ほかのモデルとの比較（同じフレーム・同じ採点、[詳細](docs/MODELS.md#other-models-on-the-same-benchmarks)）：

| | 3D誤差・別の環境 | 2D AP | 速度（1人） |
|---|---:|---:|---:|
| **obt-body-v0.3.0** | **71.3 mm** | 71.6 | GPU 6 ms · CPU 46 ms（int8） |
| MediaPipe Pose Full / Heavy | 103.3 / 105.8 mm | — | CPU 25 / 52 ms |
| RTMW-m（2Dのみ） | — | 74.1 | GPU 9 ms |
| SAM 3D Body（学習ラベルの元、ずっと大きいモデル） | 34.6 mm | — | GPU 115 ms |

## しくみ

フレームごとにYOLOXで人を見つけ（その後の約1秒は前の姿勢の枠を使い回し）、256 × 192に切り抜いて姿勢モデルに通し、One Euroフィルターで揺れを抑えます。

<p align="center"><img src="docs/architecture.svg" alt="姿勢モデル：パッチ埋め込み、DINOv2の12ブロック、2段階の特徴、関節のヘッドと分類器" width="420"></p>

## はじめかた

```sh
git clone https://github.com/yuru-lab/open-body-tracker && cd open-body-tracker
mise install && mise run setup   # または Python 3.12 + uv で `uv sync`
mise run demo                    # Webカメラ。初回にモデルをダウンロード
```

```sh
uv run obt-demo --image photo.jpg               # photo.pose.jpg を書き出す
uv run obt-demo --model-version v0.3.0-int8     # CPUで速い
uv sync --extra gpu && uv run obt-demo --gpu    # CUDA
```

ブラウザデモを手元で動かす：`mise run web` を実行して http://localhost:8000 を開きます。

## コードから使う

```python
import cv2
from open_body_tracker.detector import PersonDetector
from open_body_tracker.hub import fetch, model_file
from open_body_tracker.pipeline import PosePipeline

pipeline = PosePipeline(fetch(model_file()), PersonDetector(fetch("yolox_nano.onnx")))
pose = pipeline(cv2.imread("photo.jpg")[:, :, ::-1].copy())  # RGB
pose["xy"]     # (23, 2) 画像のピクセル座標
pose["score"]  # (23,)   自信度
pose["xyz"]    # (23, 3) メートル、腰の中心が原点、カメラの軸（x右・y下・z奥）
```

点はCOCOの17点＋左右の足の親指・小指・かかとです（[keypoints.py](src/open_body_tracker/keypoints.py)）。

## もっと知る

- [モデルの版と評価](docs/MODELS.md) · [モデルカード](MODEL_CARD.md) · [学習データ](DATA.md)（英語）
- [学習の再現](docs/REPRODUCE.md)：コードは [`training/`](training/)、ラベルは [Hugging Face](https://huggingface.co/datasets/Yurup/open-body-tracker-labels)

## ライセンス

コードとモデルの重み：[Apache-2.0](LICENSE)。学習ラベルの一部にMetaの[SAM 3D Body](https://github.com/facebookresearch/sam-3d-body)の推定結果を使いました。感謝します。詳細と第三者の表示：[docs/LICENSING.md](docs/LICENSING.md)（英語）· [NOTICE](NOTICE)

研究用のソフトウェアです。医療機器や計測機器ではありません。
