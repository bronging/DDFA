# DDFA

Toward Generalizable Multi-domain Graph Foundation Models via Dynamic Domain-Aware Feature Alignment
(**DASFAA 2026**)

## Overview

DDFA is a cross-domain few-shot node classification framework that transfers knowledge from multiple source graph domains to a target domain. It consists of two stages:

1. **Pre-training**: Learns domain-invariant representations across source domains using Intra-Domain Dimension Alignment, Inter-Domain Semantic Alignment, and Wasserstein Barycenter Regularization.
2. **Adaptation**: Adapts the pre-trained model to the target domain using a prompt-based few-shot learning approach.

## Requirements

```bash
pip install torch torch_geometric
pip install POT geomloss torch_scatter
pip install numpy scipy scikit-learn tqdm pandas networkx
```

## Data Preparation

Datasets are automatically downloaded via PyTorch Geometric on first run.

Few-shot episode files must be generated before running adaptation:

```bash
cd src
python utils/generate_fewshot.py
```

This generates 100 episodes × 10 shot settings for each dataset under `data/fewshot_{dataset}/`.

## Usage

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--target_id` | Target domain index (see below) | 0 |
| `--skip_pretrain` | Load existing checkpoint (1) or train from scratch (0) | 1 |
| `--pre_epochs` | Number of pre-training epochs | 1000 |
| `--adapt_epochs` | Number of adaptation epochs per episode | 400 |
| `--shot_num` | K for K-shot classification | 1 |
| `--alpha` | Weight of Wasserstein barycenter loss | 3.0 |
| `--beta` | Weight of diversity loss | 100.0 |
| `--gamma` | Weight of DE uniformity loss during adaptation | 100.0 |
| `--unify_dim` | Unified feature dimension k | 50 |
| `--seed` | Random seed | 39 |

Target domain index:

| ID | Dataset |
|---|---|
| 0 | Cora |
| 1 | Citeseer |
| 2 | Pubmed |
| 3 | Photo |
| 4 | Computers |
| 5 | FacebookPagePage |
| 6 | LastFMAsia |

### Running

To evaluate on all target domains:

```bash
python ./src/fug_wrapper.py --experiment DDFA --target_id 0 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.01    --l2_coef 0.0001
python ./src/fug_wrapper.py --experiment DDFA --target_id 1 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.01    --l2_coef 0.0001
python ./src/fug_wrapper.py --experiment DDFA --target_id 2 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.01    --l2_coef 0.0001
python ./src/fug_wrapper.py --experiment DDFA --target_id 3 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.00005 --l2_coef 0.0001
python ./src/fug_wrapper.py --experiment DDFA --target_id 4 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.0001  --l2_coef 0.0001
python ./src/fug_wrapper.py --experiment DDFA --target_id 5 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.001   --l2_coef 0.0001
python ./src/fug_wrapper.py --experiment DDFA --target_id 6 --skip_pretrain 1 --alpha 3 --beta 100 --gamma 100 --negative_samples_num 50 --downlr 0.001   --l2_coef 0.0001
```

For full evaluation (5 seeds × 100 episodes = 500 evaluations), repeat each command with `--seed 39`, `40`, `41`, `42`, `43`.

### Pre-training from Scratch

Set `--skip_pretrain 0`:

```bash
python ./src/fug_wrapper.py --experiment DDFA --target_id 0 --skip_pretrain 0 \
    --pre_epochs 10000 --lr 0.0001 --alpha 3 --beta 100 --unify_dim 50
```

Pre-trained checkpoints are saved under `checkpoints/{experiment}/`.

## Output

| Path | Description |
|---|---|
| `checkpoints/{experiment}/` | Pre-trained model checkpoints |
| `result/{experiment}/` | Evaluation results (CSV) |
| `{experiment}_log.txt` | Training and evaluation logs |

## Project Structure

```
src/
├── fug_wrapper.py          # Entry point
├── train.py                # Pre-training loop
├── adaptation.py           # Adaptation loop
├── preprompt.py            # Pre-training model (PrePromptBaryBasis)
├── downprompt.py           # Adaptation model (downpromptBaryBasis)
├── models/
│   ├── gcnlayers.py        # GCN backbone
│   ├── dimension.py        # Dimension encoder (DE)
│   └── LP.py               # Link prediction head
├── layers/
│   ├── prompt.py           # Prompt modules
│   └── fug.py              # Sampler
└── utils/
    ├── dataset.py           # Dataset loader
    ├── process.py           # Graph preprocessing
    ├── generate_fewshot.py  # Few-shot episode generation
    └── logging_.py          # Logging utilities
```

## Acknowledgements

This code is built upon [SAMGPT](https://github.com/blue-soda/samgpt). We thank the authors for open-sourcing their implementation.
