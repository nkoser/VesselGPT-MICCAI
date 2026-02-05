import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from funciones import IntraDataset
from Stage1.modelsMultitalk.stage1_vocaset import VQAutoEncoder

try:
    from Stage1.modelsMultitalk.stage1Emma import VQAutoEncoder as VQbatch
except Exception:
    VQbatch = None


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_value):
    if isinstance(device_value, int):
        device_value = f"cuda:{device_value}"
    if isinstance(device_value, str) and device_value.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"Using device: {torch.cuda.get_device_name(0)}")
            return torch.device(device_value)
    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def load_file_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def list_npy_files(input_dir):
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(input_dir, "*.npy"))])


class Args:
    def __init__(self):
        self.quant_loss_weight = 1.0
        self.in_dim = 39
        self.hidden_size = 1024
        self.num_hidden_layers = 6
        self.num_attention_heads = 8
        self.intermediate_size = 1536
        self.window_size = 1
        self.quant_factor = 0
        self.face_quan_num = 16
        self.neg = 0.2
        self.INaffine = False
        self.n_embed = 256
        self.zquant_dim = 64
        self.quantization_mode = "legacy"
        self.factor_count = 4
        self.factor_dim = 128


def build_args(model_cfg):
    args = Args()
    for key, value in model_cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return ckpt


def get_filename(file_name):
    if isinstance(file_name, (list, tuple)):
        return os.path.basename(file_name[0])
    return os.path.basename(file_name)


def run_reconstruction(cfg, model, device):
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    split_file_list = paths.get("split_file_list")
    input_dir = paths.get("input_dir")
    root_dir = paths.get("root_dir")
    p_value = paths.get("p")
    output_dir = paths.get("output_dir")

    if split_file_list:
        file_list = load_file_list(split_file_list)
    elif input_dir:
        file_list = list_npy_files(input_dir)
        if not root_dir:
            root_dir = input_dir
    else:
        raise ValueError("Provide paths.split_file_list or paths.input_dir.")

    if not root_dir:
        raise ValueError("paths.root_dir is required.")

    os.makedirs(output_dir, exist_ok=True)

    mode = params.get("mode", "post_order")
    batch_size = int(params.get("batch_size", 1))
    shuffle = bool(params.get("shuffle", True))
    threshold = float(params.get("threshold", 1e-2))
    sample_limit = params.get("sample_limit", None)
    if sample_limit is not None:
        sample_limit = int(sample_limit)

    dataset = IntraDataset(file_list, mode=mode, p=p_value, root_dir=root_dir, val=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    saved = 0
    model.eval()
    with torch.no_grad():
        for inputs, file_name in loader:
            if inputs.shape[1] <= 1:
                continue
            inputs = inputs.to(device)
            generated_tree, _quant = model.sample_step(inputs)
            generated = generated_tree.detach().cpu().numpy()[0]
            generated[np.abs(generated) < threshold] = 0
            name = get_filename(file_name)
            np.save(os.path.join(output_dir, name), generated)
            saved += 1
            if sample_limit is not None and saved >= sample_limit:
                break

    print(f"Saved {saved} reconstructions to: {output_dir}")


def compare_example(cfg):
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    k = int(params.get("k", 39))

    original_path = paths.get("compare_original")
    generated_path = paths.get("compare_generated")

    if not original_path or not generated_path:
        print("compare_original or compare_generated not set. Skipping compare.")
        return

    original = np.load(original_path).reshape(-1, k)
    generated = np.load(generated_path).reshape(-1, k)

    print("original shape:", original.shape)
    print("generated shape:", generated.shape)
    print("original last row:", original[-1])
    print("generated last row:", generated[-1])
    print("max original:", float(np.max(original)))
    print("max generated:", float(np.max(generated)))


def visualize_example(cfg):
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    k = int(params.get("k", 39))
    mode = params.get("visualize_mode", "pre_order")

    original_path = paths.get("compare_original")
    generated_path = paths.get("compare_generated")

    if not original_path or not generated_path:
        print("compare_original or compare_generated not set. Skipping visualize.")
        return

    from tree_functions import deserialize
    from Preprocessing.tree_viewer import draw_tree_splines

    original = np.load(original_path).reshape(-1, k)
    generated = np.load(generated_path).reshape(-1, k)

    serial = list(original.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    print("original")
    draw_tree_splines(tree)

    serial = list(generated.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    print("generated")
    draw_tree_splines(tree)


def main():
    parser = argparse.ArgumentParser(description="Clean Stage1 evaluation script (from testStage1.ipynb).")
    parser.add_argument("--config", default="test_stage1_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})

    seed_all(int(params.get("seed", 12)))
    device = resolve_device(params.get("device", 0))

    model_type = model_cfg.get("type", "vocaset")
    checkpoint_path = model_cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("model.checkpoint is required.")

    args_cfg = build_args(model_cfg)
    if model_type == "emma":
        if VQbatch is None:
            raise RuntimeError("stage1Emma model is not available in this environment.")
        hidden_dim = int(model_cfg.get("emma", {}).get("hidden_dim", 512))
        codebook_size = int(model_cfg.get("emma", {}).get("codebook_size", 2048))
        model = VQbatch(args_cfg, hidden_dim=hidden_dim, codebook_size=codebook_size).to(device)
    else:
        model = VQAutoEncoder(args_cfg).to(device)

    ckpt = load_checkpoint(model, checkpoint_path, device)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print("checkpoint epoch:", ckpt["epoch"])

    if bool(params.get("run_reconstruction", True)):
        run_reconstruction(cfg, model, device)

    if bool(params.get("run_compare", False)):
        compare_example(cfg)

    if bool(params.get("run_visualize", False)):
        visualize_example(cfg)


if __name__ == "__main__":
    main()
