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
- BraTS 2019
- BraTS 2021
- ODELIA
- MedMNIST3D (Fracture and Nodule)

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
- Obvious ID/label columns are auto-dropped if present (e.g., `id`, `label`, `BraTS19ID`…)
- Non-numeric values raise a clear error

---

## How to Run

Basic training (grayscale volumes, default paths):
```bash
python topoformer.py \
  --img_prefix ODELIA_ \
  --topo_prefix ODELIA_ \
  --tda_type M20_ \
  --in_ch 1
```

### Common options
```bash
# Training hyperparams
--batch_size 32 --epochs 100 --lr 1e-4 

# SupCon
--supcon_lambda 0.1 --supcon_temp 0.07

# Topology shape & block norm
--topo_dim 450 --topo_block 150

# Checkpointing
--ckpt_dir checkpoints --run_name topoformer
```

---

## Acknowledgements

We would like to thank the creators of the datasets for their hardwork towards advancing open source medical image analysis, pytorch and torvision for r3d_18 and the kinetics-400 weights, and the authors of SupCon for their inspiration. 

