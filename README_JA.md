# PPO / DPO / GRPO / DAPO：大規模言語モデル向け強化学習アルゴリズムの比較研究

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

**著者:** Aitachi
**連絡先:** 44158892@qq.com
**ライセンス:** MIT

**Language / 语言 / 言語:** [English](README.md) | [中文](README_CN.md) | [日本語](README_JA.md)

---

## 概要

本稿では、大規模言語モデル（LLM）のアライメントと訓練に使用される4つの代表的な強化学習アルゴリズム、すなわち近端方策最適化（PPO, Schulman et al., 2017）、直接嗜好最適化（DPO, Rafailov et al., 2023）、群相対方策最適化（GRPO, DeepSeek-AI, 2025）、および動的利得方策最適化（DAPO, Yu et al., 2025）の包括的な比較研究を提示する。Qwen2-1.5Bアーキテクチャに基づく完全な実装を提供し、数式的定式化、アーキテクチャ要件、計算効率、ダウンストリームタスク性能の複数の次元にわたって体系的な比較を行う。AIME 2024ベンチマークでの実験により、DAPOは50%の精度を達成し、バニラGRPO（30%）から67%の改善を示し、DeepSeek-R1-Zero（47%）を上回ることが示された。

**論文ダウンロード：**
- [英語版 IEEE 論文 (PDF, 10ページ)](docs/RL_LLM_Survey_IEEE_EN.pdf)
- [中国語版 IEEE 論文 (PDF, 8ページ)](docs/RL_LLM_Survey_IEEE_CN.pdf)

---

## 目次

- [1. はじめに](#1-はじめに)
- [2. 背景と記法](#2-背景と記法)
- [3. アルゴリズムの詳細](#3-アルゴリズムの詳細)
- [4. 比較分析](#4-比較分析)
- [5. 実験的評価](#5-実験的評価)
- [6. 議論](#6-議論)
- [7. 結論](#7-結論)
- [8. クイックスタート](#8-クイックスタート)
- [9. 参考文献](#9-参考文献)

---

## 1. はじめに

人間のフィードバックからの強化学習（RLHF）は、大規模言語モデルを人間の嗜好に合わせ、推論能力を向上させるための重要なパラダイムとして確立されている。InstructGPTとChatGPTを支えた基礎的なPPOアルゴリズム（2017）から、明示的な報酬モデリングを排除した簡素化されたDPOアプローチ（2023）、そして計算要件を劇的に削減しながら最先端の推論性能を達成する最近のGRPOおよびDAPOアルゴリズム（2025）まで、アルゴリズムの状況は急速に進化している。

### 貢献

1. 4つのアルゴリズムの**完全な実装**と詳細なインラインドキュメント
2. アーキテクチャ要件、損失関数特性、計算コスト、ベンチマーク性能にわたる**体系的比較**
3. 各DAPOイノベーションの寄与を分離する**アブレーション分析**
4. REINFORCE++、KTO、IPO、プロセス報酬モデルを含むより広範な文献との**議論**

---

## 2. 背景と記法

LLMアライメントの文脈では、訓練は**文脈付きバンディット**問題としてモデル化される。

| 要素 | 定義 |
|:---|:---|
| 状態/コンテキスト $x \sim \mathcal{D}$ | 訓練分布からサンプリングされた入力プロンプト |
| 行動/応答 $y = (y_1, \ldots, y_T)$ | モデルが生成するトークン列 |
| 方策 $\pi_\theta(y \mid x)$ | $\theta$でパラメータ化された自己回帰言語モデル |
| 報酬 $r(x, y) \in \mathbb{R}$ | 応答の品質を評価するスカラー信号 |

最適化目的：

$$\theta^{\ast} = \underset{\theta}{\mathrm{argmax}}\; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x)}\left[r(x, y)\right]$$

### 統一記法

| 記号 | 定義 |
|:---|:---|
| $\pi_\theta$ | 現在の方策ネットワーク（訓練中のLLM） |
| $\pi_{\mathrm{ref}}$ | 参照（凍結）方策、通常はSFTモデル |
| $\varepsilon$ | PPO/GRPO対称クリッピングパラメータ |
| $\varepsilon_{\mathrm{low}}, \varepsilon_{\mathrm{high}}$ | DAPO非対称クリッピング境界 |
| $G$ | GRPO/DAPOの群サンプリングサイズ |

---

## 3. アルゴリズムの詳細

### 3.1 PPO — 近端方策最適化

PPO（Schulman et al., 2017）は、LLMアライメントで最も広く適用されているRLアルゴリズムである。InstructGPTとChatGPTのコアアルゴリズムである。

$$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\hat{A}_t\right)\right]$$

![PPO Flow](docs/figures/ppo_flowchart.svg)

**長所：** よく研究された収束特性；ハイパーパラメータの選択に頑健；任意の報酬関数と互換。

**短所：** 価値ネットワークによる高いメモリオーバーヘッド。

---

### 3.2 DPO — 直接嗜好最適化

DPO（Rafailov et al., 2023）は、閉形式の損失関数を導出することで明示的な報酬モデリングを排除し、LLMアライメントにおけるパラダイムシフトを代表する。

$$L^{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta \cdot h(y_w,y_l,x)\right)\right]$$

![DPO Flow](docs/figures/dpo_flowchart.svg)

**長所：** 最もシンプルな実装；報酬モデル不要；安定した訓練。

**短所：** 事前収集された嗜好ペアが必要；オンライン探索不可。

---

### 3.3 GRPO — 群相対方策最適化

GRPO（DeepSeek-AI, 2025）は、群レベルの報酬統計を利得ベースラインとして使用することで価値ネットワークを排除し、GPUメモリを約50%削減する。

$$\hat{A}_i = \frac{R_i - \mu_G}{\sigma_G + \epsilon}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^{G}R_j$$

![GRPO Flow](docs/figures/grpo_flowchart.svg)

**長所：** 価値ネットワークの排除（メモリ50%削減）；検証可能なタスクに自然に適合。

**短所：** 対称クリッピングはエントロピー崩壊を引き起こす可能性。

---

### 3.4 DAPO — 動的利得方策最適化

DAPO（ByteDance, 2025）はGRPOを3つのイノベーションで拡張し、AIME 2024で**50%の精度**を達成する（バニラGRPOの30%から67%改善）。

#### イノベーション1：非対称 Clip-Higher

$$\mathrm{clip}(r_t,\ 1-\varepsilon_{\mathrm{low}},\ 1+\varepsilon_{\mathrm{high}}), \quad \varepsilon_{\mathrm{low}}=0.2,\ \varepsilon_{\mathrm{high}}=0.28$$

#### イノベーション2：動的サンプリング

$$0 < \left\lvert \left\{o_i : \mathrm{correct}(o_i)\right\}\right\rvert < G$$

#### イノベーション3：トークンレベル損失正規化

$$J_{\mathrm{DAPO}}(\theta) = \mathbb{E}\left[\frac{1}{\displaystyle\sum_{i=1}^{G}\lvert o_i\rvert}\sum_{i=1}^{G}\sum_{t=1}^{\lvert o_i\rvert} \ell_{i,t}\right]$$

![DAPO Flow](docs/figures/dapo_flowchart.svg)

---

## 4. 比較分析

| 次元 | PPO | DPO | GRPO | DAPO |
|:---|:---|:---|:---|:---|
| 年 | 2017 | 2023 | 2025 | 2025 |
| 価値ネットワーク | 必要 | 不要 | 不要 | 不要 |
| クリッピング戦略 | 対称 | なし | 対称 | **非対称** |
| 損失粒度 | トークンレベル | 列レベル | サンプルレベル | **トークンレベル** |
| 相対GPUメモリ | ~2.0x | ~1.0x | ~1.0x | ~1.2x |

---

## 5. 実験的評価

### AIME 2024ベンチマーク（Qwen2.5-32B, k=32）

| アルゴリズム | avg@32 | pass@32 | cons@32 |
|:---|:---|:---|:---|
| バニラGRPO | 30% | --- | --- |
| DeepSeek-R1-Zero | 47% | 60% | 62% |
| **DAPO** | **50%** | **75%** | **78%** |

---

## 6. 議論

### アルゴリズム選択ガイド

| シナリオ | 推奨アルゴリズム | 理由 |
|:---|:---|:---|
| 嗜好データによる主観的アライメント | **DPO** | 最もシンプルな実装、安定した訓練 |
| リソース制約のある効率的推論 | **GRPO** | 最高のメモリ効率 |
| 最大推論性能の追求 | **DAPO** | 最高のAIMEスコア、長CoTの利点 |
| 学習型報酬モデルが必要なタスク | **PPO** | 理論的保証、汎用RL |

---

## 7. 結論

4つのアルゴリズムはLLM訓練における明確な進化の軌跡を表す：

- **PPO** (2017) クリップドサロゲート基盤とデュアルネットワークアーキテクチャを確立
- **DPO** (2023) 閉形式Bradley-Terry損失によりRLHFパイプラインを大幅に簡素化
- **GRPO** (2025) 群利得正規化により価値ネットワークを排除、メモリ約50%削減
- **DAPO** (2025) 3つのイノベーションでAIME 2024精度を67%向上（30% -> 50%）

---

## 8. クイックスタート

```bash
pip install -r requirements.txt
python algorithms/grpo_trainer.py   # GRPO訓練
python algorithms/dapo_trainer.py   # DAPO訓練
python run_comparison.py            # 全アルゴリズム比較
```

---

## 9. 参考文献

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
2. Rafailov, R., et al. "Direct Preference Optimization." NeurIPS 2023.
3. DeepSeek-AI. "DeepSeek-R1." arXiv:2501.12948, 2025.
4. Yu, Q., et al. "DAPO: An Open-Source LLM RL System." arXiv:2503.14476, 2025.

---

**最終更新:** 2025-04-17 | **バージョン:** 3.0.0
