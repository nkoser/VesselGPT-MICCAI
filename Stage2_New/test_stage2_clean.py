import argparse
import os
import random
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

from funciones import Args, erase_all_files
from Stage1.modelsMultitalk.stage1_vocaset import VQAutoEncoder
from tree_functions import tokens_to_data

try:
    from transformers import GPT2LMHeadModel
except Exception as exc:
    raise RuntimeError("transformers is required for GPT-2 inference") from exc


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


def load_token_dir(token_dir, limit=None):
    files = sorted([f for f in os.listdir(token_dir) if not f.startswith(".")])
    if limit is not None:
        files = files[:limit]
    tokens = []
    for file_name in files:
        path = os.path.join(token_dir, file_name)
        tokens.append(torch.load(path))
    return tokens, files


def load_original_npy(original_dir, token_file):
    if not original_dir:
        return None
    base = os.path.splitext(os.path.basename(token_file))[0]
    npy_path = os.path.join(original_dir, base + ".npy")
    if not os.path.isfile(npy_path):
        return None
    return np.load(npy_path)


def clean_knots(data, tol=0.01):
    if data.shape[1] < 39:
        return data
    knots = data[:, 27:39]
    mask = np.abs(knots - 1.0) < tol
    knots[mask] = 1.0
    data[:, 27:39] = knots
    return data


def build_output_dir(output_root, output_subdir, gpt2_name, autoencoder_name, p, num_beams):
    if output_subdir:
        return os.path.join(output_root, output_subdir)
    if p is None:
        folder_name = f"beam{num_beams}_a{autoencoder_name}_gpt2{gpt2_name}"
    else:
        folder_name = f"p{p}_beam{num_beams}_a{autoencoder_name}_gpt2{gpt2_name}"
    return os.path.join(output_root, folder_name)


def generate_samples(model, val_tokens, device, cfg_params):
    eos_token = int(cfg_params.get("eos_token", 256))
    pad_token = int(cfg_params.get("pad_token", 257))
    max_length = int(cfg_params.get("max_length", 2514))
    temperature = float(cfg_params.get("temperature", 1.0))
    do_sample = bool(cfg_params.get("do_sample", True))
    num_beams = int(cfg_params.get("num_beams", 1))
    prefix_length = int(cfg_params.get("prefix_length", 16))
    add_eos_prefix = bool(cfg_params.get("add_eos_prefix", True))
    n_samples = int(cfg_params.get("n_samples", 1))

    samples = []
    count = min(n_samples, len(val_tokens))
    for i in range(count):
        if add_eos_prefix:
            start_seq = torch.tensor([[eos_token]], device=device, dtype=torch.long)
        else:
            start_seq = torch.empty((1, 0), device=device, dtype=torch.long)

        # Takes the first n_nodes as precondition
        if prefix_length > 0:
            prefix = val_tokens[i][:prefix_length].unsqueeze(0).to(device)
            start_seq = torch.cat((start_seq, prefix), dim=1)

        tokens = model.generate(
            start_seq,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            eos_token_id=eos_token,
            pad_token_id=pad_token,
            num_beams=num_beams,
        ).flatten()

        samples.append(tokens)

    return samples


def decode_tokens(tokens, autoencoder, device, null_id, zero_threshold, clean_knots_flag, knot_tolerance):
    data = tokens_to_data(tokens, device, autoencoder, null_id=null_id).detach().cpu().numpy()
    data = data.squeeze(0)
    data[np.abs(data) < zero_threshold] = 0
    if clean_knots_flag:
        data = clean_knots(data, tol=knot_tolerance)
    return data


def plot_tree_splines(data, out_path, title, mode, k):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    from tree_functions import deserialize
    from Preprocessing.tree_viewer import draw_tree_splines

    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    draw_tree_splines(tree)
    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def collect_nodes_edges(node, nodes, edges):
    if node is None:
        return None
    idx = len(nodes)
    nodes.append(node)
    left_idx = collect_nodes_edges(node.left, nodes, edges)
    if left_idx is not None:
        edges.append((idx, left_idx))
    right_idx = collect_nodes_edges(node.right, nodes, edges)
    if right_idx is not None:
        edges.append((idx, right_idx))
    return idx


def build_arrays(tree):
    nodes = []
    edges = []
    collect_nodes_edges(tree, nodes, edges)
    xyz = np.array([[n.data["x"], n.data["y"], n.data["z"]] for n in nodes], dtype=np.float32)
    root = np.array([tree.data["x"], tree.data["y"], tree.data["z"]], dtype=np.float32)
    return xyz, edges, root, nodes


def draw_edges(ax, xyz, edges, color, alpha, width):
    for i, j in edges:
        p1 = xyz[i]
        p2 = xyz[j]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=color,
            alpha=alpha,
            linewidth=width,
        )


def sample_spline_coeffs(coeffs, n_samples):
    from scipy.interpolate import splev

    coeffs = list(coeffs)
    t = np.array(coeffs[24:])
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8 : (i * 8) + 8]) for i in range(3)]
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)
    return np.column_stack((x, y, z))


def draw_splines(ax, nodes, root_pos, n_samples, color, alpha, size, center_root):
    for node in nodes:
        coeffs = node.data.get("r", [])
        if not isinstance(coeffs, (list, tuple, np.ndarray)) or len(coeffs) < 36:
            continue
        try:
            points = sample_spline_coeffs(coeffs, n_samples)
        except Exception:
            continue
        if center_root:
            points = points - root_pos
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=".", s=size, alpha=alpha)


def plot_overlay_tree(original, generated, out_path, params):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    from tree_functions import deserialize

    mode = params.get("overlay_mode", "pre_order")
    k = int(params.get("overlay_k", 39))
    align_root = bool(params.get("overlay_align_root", True))
    draw_splines_flag = bool(params.get("overlay_draw_splines", False))
    edges_source = params.get("overlay_edges_source", "each")
    edge_alpha = float(params.get("overlay_edge_alpha", 0.6))
    edge_width = float(params.get("overlay_edge_width", 1.0))
    node_size = float(params.get("overlay_node_size", 8.0))
    color_orig = params.get("overlay_color_original", "#1f77b4")
    color_gen = params.get("overlay_color_generated", "#ff7f0e")

    tree_orig = deserialize(list(original.flatten()), mode=mode, k=k)
    tree_gen = deserialize(list(generated.flatten()), mode=mode, k=k)
    if tree_orig is None or tree_gen is None:
        return

    xyz_o, edges_o, root_o, nodes_o = build_arrays(tree_orig)
    xyz_g, edges_g, root_g, nodes_g = build_arrays(tree_gen)

    if align_root:
        xyz_o = xyz_o - root_o
        xyz_g = xyz_g - root_g
        root_o = np.zeros(3, dtype=np.float32)
        root_g = np.zeros(3, dtype=np.float32)

    fig = plt.figure(figsize=tuple(params.get("overlay_figsize", [10, 10])))
    ax = fig.add_subplot(111, projection="3d")

    draw_edges(ax, xyz_o, edges_o, color=color_orig, alpha=edge_alpha, width=edge_width)
    if edges_source == "a":
        draw_edges(ax, xyz_g, edges_o, color=color_gen, alpha=edge_alpha, width=edge_width)
    else:
        draw_edges(ax, xyz_g, edges_g, color=color_gen, alpha=edge_alpha, width=edge_width)

    ax.scatter(xyz_o[:, 0], xyz_o[:, 1], xyz_o[:, 2], s=node_size, c=color_orig, alpha=0.8, label="original")
    ax.scatter(xyz_g[:, 0], xyz_g[:, 1], xyz_g[:, 2], s=node_size, c=color_gen, alpha=0.8, label="generated")

    if draw_splines_flag:
        n_samples = int(params.get("overlay_spline_samples", 50))
        alpha = float(params.get("overlay_spline_alpha", 0.6))
        size = float(params.get("overlay_spline_size", 0.5))
        center_root = bool(params.get("overlay_spline_center_root", True))
        draw_splines(ax, nodes_o, root_o, n_samples, params.get("overlay_spline_color_original", color_orig), alpha, size, center_root)
        draw_splines(ax, nodes_g, root_g, n_samples, params.get("overlay_spline_color_generated", color_gen), alpha, size, center_root)

    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Clean Stage2 test/generation script (from test-stage2.ipynb).")
    parser.add_argument("--config", default="test_stage2_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})

    seed_all(int(params.get("seed", 12)))
    device = resolve_device(params.get("device", 0))

    gpt2_dir = paths.get("gpt2_model_dir")
    stage1_ckpt = paths.get("stage1_checkpoint")
    token_val_dir = paths.get("token_val_dir")
    original_val_dir = paths.get("original_val_dir")
    output_root = paths.get("output_root", "generated")
    output_subdir = paths.get("output_subdir")

    if not gpt2_dir or not stage1_ckpt or not token_val_dir:
        raise ValueError("paths.gpt2_model_dir, paths.stage1_checkpoint, and paths.token_val_dir are required.")

    gpt2_name = os.path.basename(os.path.normpath(gpt2_dir))
    autoencoder_name = os.path.splitext(os.path.basename(stage1_ckpt))[0]

    args_cfg = build_args(model_cfg)
    autoencoder = VQAutoEncoder(args_cfg).to(device)
    ckpt = load_checkpoint(autoencoder, stage1_ckpt, device)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print("autoencoder epoch:", ckpt["epoch"])
    autoencoder.eval()

    gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_dir).to(device)
    gpt2.eval()

    val_limit = params.get("val_limit", None)
    if val_limit is not None:
        val_limit = int(val_limit)
    val_tokens, val_files = load_token_dir(token_val_dir, limit=val_limit)
    print(f"Loaded {len(val_tokens)} validation token sequences")

    output_dir = build_output_dir(
        output_root,
        output_subdir,
        gpt2_name,
        autoencoder_name,
        params.get("p"),
        int(params.get("num_beams", 1)),
    )

    if bool(params.get("clean_output", True)):
        os.makedirs(output_dir, exist_ok=True)
        erase_all_files(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    samples = generate_samples(gpt2, val_tokens, device, params)

    save_tokens = bool(params.get("save_tokens", False))
    save_numpy = bool(params.get("save_numpy", True))
    null_id = params.get("null_id", None)
    if null_id is not None:
        null_id = int(null_id)
    zero_threshold = float(params.get("zero_threshold", 1e-2))
    clean_knots_flag = bool(params.get("clean_knots", True))
    knot_tolerance = float(params.get("knot_tolerance", 0.01))
    visualize = bool(params.get("visualize", False))
    visualize_limit = int(params.get("visualize_limit", 3))
    visualize_mode = params.get("visualize_mode", "pre_order")
    visualize_dir = params.get("visualize_dir", output_dir)
    visualize_compare = bool(params.get("visualize_compare", False))
    visualize_k = int(params.get("visualize_k", 39))
    overlay = bool(params.get("overlay", False))
    overlay_limit = int(params.get("overlay_limit", 3))
    overlay_dir = params.get("overlay_dir", output_dir)

    decoded_samples = []

    for idx, tokens in enumerate(samples):
        if save_tokens:
            torch.save(tokens.detach().cpu(), os.path.join(output_dir, f"{idx}.tok"))
        if save_numpy or visualize:
            data = decode_tokens(tokens, autoencoder, device, null_id, zero_threshold, clean_knots_flag, knot_tolerance)
            decoded_samples.append(data)
            if save_numpy:
                np.save(os.path.join(output_dir, f"{idx}.npy"), data)

    print(f"Saved {len(samples)} samples to: {output_dir}")

    if visualize:
        os.makedirs(visualize_dir, exist_ok=True)
        count = min(visualize_limit, len(samples))
        for i in range(count):
            gen_data = decoded_samples[i]
            gen_path = os.path.join(visualize_dir, f"sample_{i}_generated.png")
            plot_tree_splines(gen_data, gen_path, f"generated {i}", visualize_mode, visualize_k)

            if visualize_compare and val_files:
                original = load_original_npy(original_val_dir, val_files[i])
                if original is not None:
                    orig_path = os.path.join(visualize_dir, f"sample_{i}_original.png")
                    plot_tree_splines(original, orig_path, f"original {i}", visualize_mode, visualize_k)

    if overlay and original_val_dir and val_files:
        os.makedirs(overlay_dir, exist_ok=True)
        count = min(overlay_limit, len(samples))
        for i in range(count):
            original = load_original_npy(original_val_dir, val_files[i])
            if original is None:
                continue
            gen_data = decoded_samples[i]
            out_path = os.path.join(overlay_dir, f"sample_{i}_overlay.png")
            plot_overlay_tree(original, gen_data, out_path, params)


if __name__ == "__main__":
    main()
