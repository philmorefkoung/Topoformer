# Topoformer

Topology-infused learning for 3D medical image classification.

[Paper](https://proceedings.mlr.press/v297/chakraborty26a.html) · [PDF](https://raw.githubusercontent.com/mlresearch/v297/main/assets/chakraborty26a/chakraborty26a.pdf) · [Data preparation](docs/data.md) · [Citation](#citation)

Research code for **Topoformer: Topology-Infused Transformers for Medical Imaging**, by Sayoni Chakraborty, Philmore Koung, and Baris Coskunuzer. Published in the *Proceedings of the Fifth Machine Learning for Health Symposium*, PMLR 297:23–40, 2026.

## Overview

Topoformer combines volumetric image representations with topological descriptors. The paper introduces sliding-band filtrations, topological supervised contrastive learning (TopoSupCon), and image-guided weighting of filtration widths (TopoGate).

This repository provides three standalone research scripts:

| File | Purpose |
| --- | --- |
| [TopoGate.py](TopoGate.py) | A 3D ResNet-18 image encoder guides the weighting of multiple topology feature sets; a transformer head classifies the combined features. |
| [Topoformer.py](Topoformer.py) | A 3D ResNet-18 image branch and an MLP topology branch are trained with classification and supervised contrastive objectives. |
| [sw_betti.py](sw_betti.py) | Extracts sliding-band Betti features from preprocessed BraTS-style NumPy volumes. |
| [requirements.txt](requirements.txt) | Python dependencies used by the scripts. |

The repository contains training and feature-extraction code. Prepared datasets, pretrained study checkpoints, split manifests, and a locked experiment environment are not included. Consult the paper for the complete experimental protocol and reported results.

## Installation

Start with a Python 3.10 environment, as used in the original setup instructions:

```bash
git clone https://github.com/philmorefkoung/Topoformer.git
cd Topoformer
conda create -n topoformer python=3.10 -y
conda activate topoformer
python -m pip install -r requirements.txt
```

The requirements specify minimum versions rather than a tested lockfile. Install a compatible PyTorch/torchvision pair for your hardware. GPU training is recommended for 3D volumes; both training scripts select CUDA when available. `Topoformer.py` requests torchvision's Kinetics-400 weights, while `TopoGate.py` does so only with `--use_pretrained`; the first use may download these weights.

Inspect the available training options:

```bash
python TopoGate.py --help
python Topoformer.py --help
```

Filenames are case-sensitive on Linux. The commands below are Bash examples; in PowerShell, put each command on one line or replace the continuation characters appropriately.

## Data preparation

Prepare image volumes, labels, and topological features for the same samples in the same order. The scripts join these inputs **by row position**, not by sample ID.

The two training scripts use different NPZ key conventions. Both require headerless, numeric, feature-only CSV files. See [the data guide](docs/data.md) for exact schemas, directory examples, extraction settings, and conversion requirements.

Dataset sources linked by this project include:

| Dataset | Source |
| --- | --- |
| BraTS 2019 | [Challenge data](https://www.med.upenn.edu/cbica/brats2019/data.html) |
| BraTS 2021 | [Kaggle dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) |
| ODELIA | [Hugging Face dataset](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025) |
| RSNA 2025 | [Intracranial Aneurysm Detection challenge](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection) |
| MedMNIST3D | [MedMNIST](https://medmnist.com/) |

Obtain data from the respective providers and follow their access and use terms. Dataset-specific resizing, label mapping, and creation of training/validation/test NPZ files must be completed before training.

## Training

### TopoGate: combine multiple filtration widths

The example below assumes a two-class dataset and two aligned, 150-feature topology representations. Set `--num_classes` and `--topo_dim` to match your prepared data; keep the width order identical across training and validation.

```bash
python TopoGate.py \
  --train_npz data/topogate/train.npz \
  --val_npz data/topogate/val.npz \
  --train_csvs data/topogate/M20_train.csv data/topogate/M40_train.csv \
  --val_csvs data/topogate/M20_val.csv data/topogate/M40_val.csv \
  --topo_dim 150 --num_classes 2 --use_pretrained \
  --num_workers 0 --run_name topogate
```

TopoGate saves `checkpoints/topogate_best.pt` when validation AUC improves, then reports average validation gate weights and the highest-weight width index. The checkpoint includes the configuration, model weights, training-set topology normalization statistics, epoch, and validation AUC. This entry point does not run a held-out test evaluation.

### Topoformer: image–topology fusion with TopoSupCon

This example uses a single 150-feature topology representation per volume. `--topo_dim` must match the CSV width and be divisible by `--topo_block`.

```bash
python Topoformer.py \
  --img_prefix data/topoformer \
  --topo_prefix data/topoformer \
  --tda_type M20_ \
  --topo_dim 150 --topo_block 150 --in_ch 1 \
  --num_workers 0 --run_name topoformer
```

The script loads all three splits, selects a checkpoint by validation AUC, saves `checkpoints/topoformer_best.pt`, and evaluates the selected weights on the test split. It prints AUC, accuracy, macro precision/recall/F1, and macro specificity. The checkpoint contains the configuration, model weights, epoch, and validation AUC; retain the training features to reconstruct their normalization statistics.

Both examples use `--num_workers 0` for portability. Increase it for your training environment. Use `--no_amp` to disable mixed precision, and a distinct `--run_name` or `--ckpt_dir` for each experiment to avoid overwriting checkpoints.

## Reproducibility

- Keep a sample manifest linking volume order, labels, feature rows, and split membership. Check that every evaluation split contains the classes required to compute AUC.
- Record preprocessing, filtration bounds/stride/widths, channel order, random seed, command, and package versions. Both training scripts accept `--seed` and fit topology normalization on the training split.
- Match the paper's dataset-specific protocol before comparing results. The example commands illustrate the interfaces; they are not complete reproductions of the paper's experiments.
- Runtime and model performance depend on the dataset and hardware. End-to-end training is not verified by the documentation examples alone.

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{pmlr-v297-chakraborty26a,
  title     = {{Topoformer}: Topology-Infused Transformers for Medical Imaging},
  author    = {Chakraborty, Sayoni and Koung, Philmore and Coskunuzer, Baris},
  booktitle = {Proceedings of the Fifth Machine Learning for Health Symposium},
  series    = {Proceedings of Machine Learning Research},
  volume    = {297},
  pages     = {23--40},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v297/chakraborty26a.html}
}
```

Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements

We thank the dataset creators, the PyTorch and torchvision contributors for the `r3d_18` backbone and Kinetics-400 weights, and the authors of Supervised Contrastive Learning for their work.

## License and questions

A software license has not yet been specified in this repository. For reuse permissions or implementation questions, please [open an issue](https://github.com/philmorefkoung/Topoformer/issues).
