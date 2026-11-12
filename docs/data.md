# Preparing data for Topoformer

[Back to the README](../README.md)

The loaders expect preprocessed arrays and aligned topology features. They do not download datasets, generate patient-level splits, or match samples by ID. Keep a separate manifest so image, label, and feature order can be checked before training.

## Shared requirements

- Labels must be integer class indices from `0` to `num_classes - 1`, stored as a one-dimensional array of shape `(N,)`.
- Every image array and feature matrix for a split must have the same number of rows.
- Feature CSVs must contain only finite numeric feature values: no header, index, sample ID, or label column. Both loaders call `pandas.read_csv(..., header=None)`.
- Keep feature ordering and preprocessing identical across splits. Supply a separate feature file for each filtration width used by TopoGate.
- Use fixed spatial dimensions within each image array and dimensions appropriate for the 3D backbone. The original scripts describe 64 × 64 × 64 volumes.

To save an already ordered feature matrix:

```python
import numpy as np

# features: (N, D), already aligned with the corresponding volume array
np.savetxt("M20_train.csv", features, delimiter=",")
```

## TopoGate input format

`TopoGate.py` reads split-prefixed keys:

| File passed to the CLI | Required keys |
| --- | --- |
| `--train_npz` | `train_images`, `train_labels` |
| `--val_npz` | `val_images`, `val_labels` |

Images must be grayscale arrays of shape `(N, T, H, W)` or `(N, T, H, W, 1)`. The model adds the channel dimension and applies its Kinetics-derived normalization; the loader casts the data to `float32` but does not scale the intensity range or standardize each volume. Prepare image intensities consistently for your experiment.

```python
import numpy as np

np.savez_compressed("train.npz", train_images=train_images, train_labels=train_labels)
np.savez_compressed("val.npz", val_images=val_images, val_labels=val_labels)
```

Each `--train_csvs` / `--val_csvs` pair must represent the same filtration width. Every CSV must have exactly `--topo_dim` columns. For example, two widths with 150 features each require two `(N, 150)` CSVs per split, not one `(N, 300)` CSV.

```text
data/topogate/
├── train.npz
├── val.npz
├── M20_train.csv
├── M40_train.csv
├── M20_val.csv
└── M40_val.csv
```

The loader computes topology means and standard deviations over the training samples and supplied widths, then reuses those statistics for validation.

## Topoformer input format

`Topoformer.py` reads `train.npz`, `val.npz`, and `test.npz` from the directory passed as `--img_prefix`. Use `images` and `labels` as the NPZ keys in each file:

```python
import numpy as np

np.savez_compressed("train.npz", images=train_images, labels=train_labels)
np.savez_compressed("val.npz", images=val_images, labels=val_labels)
np.savez_compressed("test.npz", images=test_images, labels=test_labels)
```

The loader also recognizes aliases such as `volumes` for images and `y` for labels; it does **not** recognize TopoGate's `train_images` / `train_labels` keys. Images may have shape `(N, T, H, W)`, `(N, C, T, H, W)`, or `(N, T, H, W, C)`, with channel counts `1`, `3`, or `4` for five-dimensional arrays. Prefer channel-first arrays when using explicit channels, and set `--in_ch` to the actual channel count. Each volume is z-score standardized by the dataset class.

Topology files are read from `--topo_prefix`, with filenames formed by `--tda_type` followed by the split and `.csv`:

```text
data/topoformer/
├── train.npz
├── val.npz
├── test.npz
├── M20_train.csv
├── M20_val.csv
└── M20_test.csv
```

Each feature CSV must have exactly `--topo_dim` columns. This dimension must be divisible by `--topo_block` (default: `150`). The script fits block-organized feature normalization using only training data and applies it to validation and test data.

## Sliding-band feature extraction

`sw_betti.py` is a dataset-specific extraction script with configuration constants at the top of the file; it has no command-line argument parser. Before running it, update:

| Constant | Meaning |
| --- | --- |
| `INPUT_DIR` | Root directory containing class and patient folders. |
| `OUT_W04`, `OUT_W02` | Output CSV paths; their parent directories must exist. |
| `LABELS` | Class directory names and integer labels. |
| `MODALITIES` | Ordered list of NumPy volume filenames per patient. |
| `A`, `B`, `STRIDE`, `N_SLICES` | Intensity bounds and sampling configuration. |
| `W0`, `W1` | The two sliding-band widths. |
| `N_JOBS` | Parallel worker count; choose a value appropriate for available memory and CPUs. |

The provided configuration expects:

```text
INPUT_DIR/
├── LGG/
│   └── patient_id/
│       ├── flair.npy
│       ├── t1.npy
│       ├── t1ce.npy
│       └── t2.npy
└── HGG/
    └── patient_id/
        └── ...
```

Each file must contain one 3D array. The script clips values to `[-5, 5]` under its default settings. It produces 50 values for each of Betti-0, Betti-1, and Betti-2 per modality, concatenated in that order. Four modalities therefore produce **600 feature columns per width**. Patients missing a required modality are skipped.

After editing the configuration, run:

```bash
python sw_betti.py
```

The outputs contain a header, an `ID` column, 600 feature columns under the default configuration, and a `Label` column. They are **not directly compatible** with the training loaders. Use the sample IDs to align records with your split manifest and image arrays, then export only the ordered feature columns with `header=False, index=False` for each split. Store labels separately in the appropriate NPZ schema. Do not rely on automatic ID/label removal in the training loaders.

The extraction script does not build NPZ image arrays or split manifests. Its default widths are `0.4` and `0.2`; names such as `M20_` in the training examples are filename prefixes, not extraction settings. Record the actual numerical filtration configuration with each generated file.
