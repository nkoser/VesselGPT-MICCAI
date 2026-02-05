import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Stage1.funciones import Args
from Stage1.modelsMultitalk.stage1_vocaset import VQAutoEncoder


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_device(device_value):
    if isinstance(device_value, int):
        device_value = f"cuda:{device_value}"
    if isinstance(device_value, str) and device_value.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"Using device: {torch.cuda.get_device_name(0)}")
            return torch.device(device_value)
    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def load_state_dict(model, checkpoint_path, device, strict=True):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=strict)


def iter_files(input_dir, pattern):
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def build_model(cfg, device):
    model_cfg = cfg.get("model", {})
    params = cfg.get("params", {})
    k = int(params.get("k", 39))

    args_cfg = Args()
    args_cfg.in_dim = int(model_cfg.get("in_dim", k))
    for key, value in model_cfg.items():
        if hasattr(args_cfg, key):
            setattr(args_cfg, key, value)

    model = VQAutoEncoder(args_cfg).to(device)
    model.eval()
    return model, args_cfg


def main():
    parser = argparse.ArgumentParser(description="Tokenize tree datasets with a trained Stage1 VQ-VAE.")
    parser.add_argument("--config", default="tokenize_dataset_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_dir = paths.get("input_dir")
    output_dir = paths.get("output_dir")
    checkpoint_path = paths.get("model_checkpoint")

    if not input_dir or not output_dir or not checkpoint_path:
        raise ValueError("paths.input_dir, paths.output_dir, and paths.model_checkpoint are required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(params.get("device", 0))
    pattern = params.get("pattern", "*.npy")
    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    node_dim = k + 1 if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"} else k
    zero_threshold = float(params.get("zero_threshold", 1e-3))
    overwrite = bool(params.get("overwrite", False))
    add_bos_eos = bool(params.get("add_bos_eos", False))
    bos_id = int(params.get("bos_id", 256))
    eos_id = int(params.get("eos_id", 256))
    strict_load = bool(params.get("strict_load", True))

    model, args_cfg = build_model(cfg, device)
    load_state_dict(model, checkpoint_path, device, strict=strict_load)

    null_id = params.get("null_id", None)
    if null_id is None:
        null_id = int(args_cfg.n_embed) + 1
    else:
        null_id = int(null_id)

    if null_id in (bos_id, eos_id):
        raise ValueError("null_id cannot match bos_id or eos_id.")

    files = iter_files(input_dir, pattern)
    written = 0
    skipped = 0

    for file_path in files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = output_dir / f"{name}.tok"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        data = np.load(file_path)
        if data.ndim == 1:
            data = data.reshape((-1, node_dim))
        if data.shape[1] != node_dim:
            raise ValueError(f"{file_path} has {data.shape[1]} features, expected {node_dim}.")

        if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}:
            data_attrs = data[:, 1:]
        else:
            data_attrs = data
        zero_mask = np.all(np.abs(data_attrs) <= zero_threshold, axis=1)
        tokens_per_row = int(args_cfg.face_quan_num)
        if zero_mask.all():
            tokens = torch.full((data.shape[0] * tokens_per_row,), null_id, dtype=torch.long)
        else:
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _quant, indices = model.get_quant(tensor)
            tokens = indices.view(-1).detach().cpu().long()
            if tokens.numel() % data.shape[0] != 0:
                raise ValueError(
                    f"{file_path} produced {tokens.numel()} tokens for {data.shape[0]} rows. "
                    "Check quant_factor or preprocessing."
                )
            tokens_per_row = tokens.numel() // data.shape[0]
            if tokens_per_row != int(args_cfg.face_quan_num):
                print(
                    f"Warning: tokens per row ({tokens_per_row}) != face_quan_num ({args_cfg.face_quan_num})."
                )
            if zero_mask.any():
                mask = torch.from_numpy(zero_mask).repeat_interleave(tokens_per_row)
                tokens[mask] = null_id

        if add_bos_eos:
            tokens = torch.cat([torch.tensor([bos_id]), tokens, torch.tensor([eos_id])])

        torch.save(tokens, out_path)
        written += 1

    print(f"Done. Written: {written} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
