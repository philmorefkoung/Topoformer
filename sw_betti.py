import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import label as cc_label, convolve
from joblib import Parallel, delayed


INPUT_DIR  = '/groups/bcoskunuzer/sxc220042/BRATS2019/brats2019_npy'
OUT_W04    = '/groups/bcoskunuzer/sxc220042/BRATS2019/brats2019_sw_04_betti.csv'
OUT_W02    = '/groups/bcoskunuzer/sxc220042/BRATS2019/brats2019_sw_02_betti.csv'
LABELS     = {'LGG': 0, 'HGG': 1}
MODALITIES = ['flair.npy', 't1.npy', 't1ce.npy', 't2.npy']

A, B     = -5.0, 5.0
N_SLICES = 50
STRIDE   = 0.2
W0, W1   = 0.4, 0.2

CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)
N_JOBS       = 64

# Euler characteristic
def compute_euler_characteristic(binary: np.ndarray) -> int:
    binary = (binary.astype(np.uint8) == 1)
    k_v  = np.ones((2, 2, 2), dtype=int)
    k_x  = np.ones((1, 2, 2), dtype=int)
    k_y  = np.ones((2, 1, 2), dtype=int)
    k_z  = np.ones((2, 2, 1), dtype=int)
    k_xy = np.ones((1, 1, 2), dtype=int)
    k_yz = np.ones((2, 1, 1), dtype=int)
    k_zx = np.ones((1, 2, 1), dtype=int)

    v = np.sum(convolve(binary, k_v,  mode='constant', cval=0) == 1)
    e = (np.sum(convolve(binary, k_x, mode='constant', cval=0) == 1) +
         np.sum(convolve(binary, k_y, mode='constant', cval=0) == 1) +
         np.sum(convolve(binary, k_z, mode='constant', cval=0) == 1))
    f = (np.sum(convolve(binary, k_xy, mode='constant', cval=0) == 1) +
         np.sum(convolve(binary, k_yz, mode='constant', cval=0) == 1) +
         np.sum(convolve(binary, k_zx, mode='constant', cval=0) == 1))
    c = np.sum(binary)
    return int(v - e + f - c)

# Betti-0/1/2 via sliding
def betti_features(volume: np.ndarray, width: float) -> list:
    vol = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vol = np.clip(vol, A, B)
    b0_list, b1_list, b2_list = [], [], []
    Z, Y, X = vol.shape

    for i in range(N_SLICES):
        lower = A + i * STRIDE
        upper = lower + width
        if i == N_SLICES - 1 or upper > B:
            binary = ((vol >= lower) & (vol <= B)).astype(np.uint8)
        else:
            binary = ((vol >= lower) & (vol < upper)).astype(np.uint8)

        # Betti-0
        _, b0 = cc_label(binary, structure=CONNECTIVITY)

        # Betti-2
        inv = (1 - binary).astype(np.uint8)
        lab_bg, n_comp = cc_label(inv, structure=CONNECTIVITY)
        b2 = 0
        for lbl in range(1, n_comp + 1):
            mask = (lab_bg == lbl)
            if not np.any(mask):
                continue
            z, y, x = np.where(mask)
            if (z.min() == 0 or z.max() == Z-1 or
                y.min() == 0 or y.max() == Y-1 or
                x.min() == 0 or x.max() == X-1):
                continue
            b2 += 1
        #Betti-1
        chi = compute_euler_characteristic(binary)
        b1 = max(b0 + b2 - chi, 0)

        b0_list.append(int(b0))
        b1_list.append(int(b1))
        b2_list.append(int(b2))

    return b0_list + b1_list + b2_list

def process_patient(class_name: str, label_value: int, pid: str):
    pdir = os.path.join(INPUT_DIR, class_name, pid)
    f_w04, f_w02 = [], []
    for m in MODALITIES:
        f = os.path.join(pdir, m)
        if not os.path.isfile(f):
            return None
        vol = np.load(f)
        f_w04.extend(betti_features(vol, W0))
        f_w02.extend(betti_features(vol, W1))
    return pid, f_w04, f_w02, label_value

tasks = []
for cname, lab in LABELS.items():
    cpath = os.path.join(INPUT_DIR, cname)
    if not os.path.isdir(cpath):
        continue
    for pid in sorted(p for p in os.listdir(cpath) if os.path.isdir(os.path.join(cpath, p))):
        tasks.append((cname, lab, pid))

# parallel
results = Parallel(n_jobs=N_JOBS)(
    delayed(process_patient)(cname, lab, pid) for (cname, lab, pid) in tqdm(tasks)
)

#collect & save 
ids, rows_w04, rows_w02, labels = [], [], [], []
for r in results:
    if r is None:
        continue
    pid, f04, f02, lab = r
    ids.append(pid)
    rows_w04.append(f04)
    rows_w02.append(f02)
    labels.append(lab)

df = pd.DataFrame(rows_w04)
df.insert(0, 'ID', ids)
df['Label'] = labels
df.to_csv(OUT_W04, index=False)

df2 = pd.DataFrame(rows_w02)
df2.insert(0, 'ID', ids)
df2['Label'] = labels
df2.to_csv(OUT_W02, index=False)

print(f"Saved SW-Betti features")

