import os
import random
import shutil
from glob import glob

import yaml


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_counts(n_total, train_ratio, val_ratio):
    train_n = int(round(n_total * train_ratio))
    val_n = int(round(n_total * val_ratio))
    test_n = n_total - train_n - val_n
    if test_n < 0:
        test_n = 0
    return train_n, val_n, test_n


def main():
    config_path = "split_dataset_config.yaml"
    if not os.path.exists(config_path):
        raise SystemExit("Error: split_dataset_config.yaml not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    input_dir = paths.get("input")
    output_dir = paths.get("output")
    train_ratio = float(params.get("train", 0.8))
    val_ratio = float(params.get("val", 0.1))
    test_ratio = float(params.get("test", 0.1))
    seed = int(params.get("seed", 42))
    pattern = params.get("pattern", "*.npy")
    move_files = bool(params.get("move", False))

    if not input_dir or not output_dir:
        raise SystemExit("Error: paths.input and paths.output are required in split_dataset_config.yaml.")

    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("Ratios must sum to a positive number")

    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    files = sorted(glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError("No files found in input")

    random.seed(seed)
    random.shuffle(files)

    train_n, val_n, test_n = split_counts(len(files), train_ratio, val_ratio)
    train_files = files[:train_n]
    val_files = files[train_n: train_n + val_n]
    test_files = files[train_n + val_n: train_n + val_n + test_n]

    out_train = os.path.join(output_dir, "train")
    out_val = os.path.join(output_dir, "val")
    out_test = os.path.join(output_dir, "test")

    ensure_dir(out_train)
    ensure_dir(out_val)
    ensure_dir(out_test)

    op = shutil.move if move_files else shutil.copy2

    for src in train_files:
        op(src, os.path.join(out_train, os.path.basename(src)))
    for src in val_files:
        op(src, os.path.join(out_val, os.path.basename(src)))
    for src in test_files:
        op(src, os.path.join(out_test, os.path.basename(src)))

    print(f"done: train={len(train_files)} val={len(val_files)} test={len(test_files)}")


if __name__ == "__main__":
    main()
