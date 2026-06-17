"""
Generates stratified train/val/test CSV splits for the full-slide images
(root of each class folder, NOT the CROPPED subfolder).

Output files (saved at dataset root):
  train_slides.csv, val_slides.csv, test_slides.csv, train_val_slides.csv

Usage: python prepare_slides.py
"""

import os
import random
import csv
from pathlib import Path
from collections import defaultdict

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15 (remainder)

CLASSES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

BASE_DIR = Path(__file__).parent

random.seed(SEED)

all_samples = []
for class_id, cls in enumerate(CLASSES):
    folder = BASE_DIR / f"im_{cls}" / f"im_{cls}"
    bmps = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".bmp")
    for p in bmps:
        rel = p.relative_to(BASE_DIR).as_posix()
        all_samples.append((class_id, rel))
    print(f"  {cls}: {len(bmps)} slides")

print(f"\nTotal: {len(all_samples)} slides\n")

# stratified split
by_class = defaultdict(list)
for cid, rel in all_samples:
    by_class[cid].append(rel)

train, val, test, train_val = [], [], [], []
for cid, paths in by_class.items():
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    tr  = paths[:n_train]
    va  = paths[n_train:n_train + n_val]
    te  = paths[n_train + n_val:]
    for p in tr:  train.append((cid, p))
    for p in va:  val.append((cid, p))
    for p in te:  test.append((cid, p))
    for p in tr + va: train_val.append((cid, p))
    print(f"  {CLASSES[cid]}: train={len(tr)}, val={len(va)}, test={len(te)}")

def write_csv(name, rows):
    path = BASE_DIR / name
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "dir"])
        w.writerows(rows)
    print(f"Saved: {name} ({len(rows)} rows)")

print()
write_csv("train_slides.csv",     train)
write_csv("val_slides.csv",       val)
write_csv("test_slides.csv",      test)
write_csv("train_val_slides.csv", train_val)

print(f"\nDone. Train={len(train)}, Val={len(val)}, Test={len(test)}, TrainVal={len(train_val)}")
