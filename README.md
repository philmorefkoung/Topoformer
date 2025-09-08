# Topoformer: Topology-Infused Transformers for Medical Imaging

Code for the paper: "Topoformer: Topology-Infused Transformers for Medical Imaging" <br>
Topoformer consists of two models: <br>
- TopoGate: Image guided topological embeddings, allowing us to use the best filtration width per dataset <br>
- TopoSupCon:  An augmentation free variant of Supervised Contrastive Learning via a sliding band filtration sequence <br>

We also provide the code to our sliding band filtration method.

---

## Installation

### 1) Create an environment

**Conda** (recommended):
```bash
conda create -n topoformer python=3.10 -y
conda activate topoformer
```

### 2) Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Datasets

Datasets we used for our experiments can be found below:
- [BraTS 2019](https://www.med.upenn.edu/cbica/brats2019/data.html)
- [BraTS 2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
- [ODELIA](https://huggingface.co/datasets/ODELIA-AI/ODELIA-Challenge-2025/tree/main/example-algorithm)
- [RSNA 2025](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- [MedMNIST3D](https://medmnist.com/) (Fracture and Nodule)

Provide two synchronized modalities per split:

- **Volumes**: `train.npz`, `val.npz`, `test.npz` under `--img_prefix` (e.g. `ODELIA_`)
- **Topological CSVs**: `${TDA_PREFIX}{split}.csv` under `--topo_prefix` (e.g. `ODELIA_`)

```
ODELIA_/train.npz
ODELIA_/val.npz
ODELIA_/test.npz
ODELIA_/M20_train.csv
ODELIA_/M20_val.csv
ODELIA_/M20_test.csv
```

### Volume format (`.npz`)
- Accepts keys: `volumes`, `images`, etc. (forgiving lookup)
- Values may be shaped **(N, T, H, W)** *or* **(N, C, T, H, W)** *or* **(N, T, H, W, C)**
- Values are cast to `float32`. If channels are missing, they’re added/expanded.

### Topological features (`.csv`)
- Numeric matrix of shape **(N, D)**, default `D=450`
- Obvious ID/label columns are auto-dropped if present (e.g., `id`, `label`, `Modality`…)
- Non-numeric values raise a clear error

---

## How to Run
TopoGate (Example)
```bash
python topogate.py \
  --train_npz nodulemnist3d_64_train.npz \
  --val_npz   nodulemnist3d_64_val.npz \
  --train_csvs nodulemnist3d_M20_train.csv nodulemnist3d_M40_train.csv \
  --val_csvs   nodulemnist3d_M20_val.csv   nodulemnist3d_M40_val.csv \
  --topo_dim 150 --num_classes 3 --use_pretrained
```

Basic training for Topoformer Example (grayscale volumes, default paths):
```bash
python topoformer.py \
  --img_prefix ODELIA_ \
  --topo_prefix ODELIA_ \
  --tda_type M20_ \
  --in_ch 1
```
---

## Acknowledgements

We would like to thank the dataset creators for their hard work in advancing open-source medical image analysis; the PyTorch and torchvision contributors for r3d_18 and the Kinetics-400 weights; and the authors of Supervised Contrastive Learning (SupCon) for their inspiration.

