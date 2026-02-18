<p align="center">
  <img src="assets/easydub.png" width="200" alt="EasyDUB mascot">
</p>

<h1 align="center">EasyDUB</h1>
<h3 align="center">Easy <b>D</b>ata <b>U</b>nlearning <b>B</b>ench</h3>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="MIT License"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg?style=flat-square" alt="Python >= 3.11"></a>
  <a href="https://huggingface.co/datasets/easydub/EasyDUB-dataset"><img src="https://img.shields.io/badge/-Dataset-gray.svg?logo=huggingface&style=flat-square" alt="Dataset on HuggingFace"></a>
</p>

A minimal CIFAR-10 benchmark for evaluating data-unlearning methods using **KLOM** (KL-divergence of Margins). Ships with 200 pretrain models, 2,000 oracle models, and precomputed margins -- everything needed to score a new method in a few lines of code.

---

* [Getting Started](#getting-started)
* [KLOM Metric](#klom-metric)
* [Running Unlearning](#running-unlearning)
* [Strong Test](#strong-test)
* [Citation](#citation)

## Getting Started

Run the demo with a single command -- the dataset downloads automatically on first run:

```bash
git clone https://github.com/easydub/EasyDUB-code.git && cd EasyDUB-code
uv run python demo.py --method noisy_descent --forget-set 1 --n-models 100
```

This runs noisy-SGD unlearning on 100 pretrain models for forget set 1 and prints KLOM scores against the oracle. No separate install or dataset download step needed.

You can also compute KLOM from precomputed margins directly (no GPU required):

```python
from easydub.data import ensure_dataset, load_margins_array
from easydub.eval import klom_from_margins
import numpy as np, torch

root = ensure_dataset()  # downloads once, then caches
n_models = 100

pretrain = torch.from_numpy(np.stack([
    load_margins_array(root, kind="pretrain", phase="val", forget_id=None, model_id=i)
    for i in range(n_models)
]))
oracle = torch.from_numpy(np.stack([
    load_margins_array(root, kind="oracle", phase="val", forget_id=1, model_id=i)
    for i in range(n_models)
]))

klom = klom_from_margins(oracle, pretrain)
print(f"KLOM(pretrain, oracle): mean={klom.mean():.4f}")
```

This gives you the KLOM baseline: how far the pretrain models are from the oracle (retrained-from-scratch) models. A good unlearning method should produce a lower KLOM than this baseline.

## KLOM Metric

KLOM measures how close an unlearned model's per-sample margin distribution is to that of an oracle model retrained without the forget set.

The **margin** for a sample with logits $z$ and true label $c$ is:

$$m = z_c - \log \sum_{j \neq c} e^{z_j}$$

KLOM bins these margins across models and computes a per-sample KL divergence between the unlearned and oracle distributions. Lower is better.

## Running Unlearning

`demo.py` runs an unlearning method on pretrain models and evaluates with KLOM. The `--method` flag accepts any method in `easydub.unlearning.UNLEARNING_METHODS` (see `easydub/unlearning.py` for the full list). Pass `--data-dir` to use a local copy of the dataset, or omit it to auto-download.

```bash
uv run python demo.py --method noisy_descent --forget-set 1 --n-models 10 --epochs 5
```

## Strong Test

`strong_test.py` is a reproducible experiment that compares KLOM(pretrain, oracle) vs KLOM(noisy_descent, oracle) on validation margins. A successful unlearning method should close the gap.

```bash
uv run python strong_test.py --forget-set 1 --n-models 100
```

## Citation

If you use EasyDUB in your work, please cite:

```bibtex
@inproceedings{rinberg2025dataunlearnbench,
  title     = {Easy Data Unlearning Bench},
  author    = {Rinberg, Roy and Puigdemont, Pol and Pawelczyk, Martin and Cevher, Volkan},
  booktitle = {MUGEN Workshop at ICML},
  year      = {2025},
}
```

EasyDUB builds on the KLOM metric introduced in:

```bibtex
@misc{georgiev2024attributetodeletemachineunlearningdatamodel,
  title         = {Attribute-to-Delete: Machine Unlearning via Datamodel Matching},
  author        = {Kristian Georgiev and Roy Rinberg and Sung Min Park and Shivam Garg and Andrew Ilyas and Aleksander Madry and Seth Neel},
  year          = {2024},
  eprint        = {2410.23232},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.23232},
}
```

## License

[MIT](LICENSE)
