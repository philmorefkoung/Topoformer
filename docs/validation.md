# Correctness and execution checks

[Back to the README](../README.md)

## Contrastive correspondence correction

`Topoformer.forward(..., contrastive=True)` concatenates all image embeddings followed by all topology embeddings. For a batch with labels `[0, 1]`, the matching contrastive labels must therefore be `[0, 1, 0, 1]`.

The earlier loss used `repeat_interleave(2)`, producing `[0, 0, 1, 1]` and assigning positive pairs to the wrong samples whenever batch labels differed. `SupConLoss` now repeats the complete label sequence and checks that the input contains exactly two views per sample. This corrects the existing objective without changing the model architecture. It does not establish whether the same implementation was used to produce the published results.

## Regression tests

```bash
python -m unittest discover -s tests -v
```

The three tests cover:

- The loss and its feature gradients against an independent, anchor-by-anchor supervised contrastive calculation, including more than one sample in a class.
- Lower loss for matching image/topology views than for swapped views from different classes.
- Invariance when the same sample permutation is applied to both modalities and labels.

All three passed with the corrected loss and failed when checked against the earlier loss implementation.

## Tiny training and held-out evaluation

```bash
python scripts/smoke_train.py
```

The script creates separate synthetic training, validation, and test arrays in a temporary directory and uses the repository's existing data loaders, trainers, and evaluation functions. Each split has four samples from two classes. Each model trains for one epoch with batch size two, giving two optimizer steps. Volumes are 16 × 16 × 16, and topology vectors have 150 features. CUDA is hidden in this smoke-check process, CPU threads are limited to two, mixed precision is disabled, and no pretrained weights are requested.

| Model | Checks performed |
| --- | --- |
| Topoformer | Real R3D-18 backbone and classifier parameters change; held-out loss and metrics are finite; saved-checkpoint evaluation matches the trainer's held-out metrics. |
| TopoGate | Two synthetic topology views exercise the width gate; gate and classifier parameters change; the selected checkpoint and its training normalization statistics load for a separate held-out evaluation with finite metrics. |

The synthetic topology arrays are generated directly to exercise training; the check does not test sliding-band extraction or medical-image preprocessing. TopoGate's existing backbone feature extraction remains under `no_grad`. All temporary files are removed when the check finishes.

The JSON summary at the end reports sample counts, update checks, and held-out metrics. Its metrics describe only tiny synthetic data and should not be compared with the paper's benchmarks.

## Recorded verification environment

Both training checks and all regression tests passed on November 12, 2026 using Windows and CPU execution:

| Component | Version |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| tqdm | 4.68.1 |

These checks used an existing environment; they do not certify every combination allowed by the minimum dependency versions. Current PyTorch emits deprecation notices for the original CUDA AMP API and a pinned-memory notice on CPU. These notices did not prevent either check from passing.
