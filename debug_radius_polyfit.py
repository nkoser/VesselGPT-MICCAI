import argparse
import os
from glob import glob

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("matplotlib is required. Install with: pip install matplotlib") from exc

try:
    from scipy.interpolate import splev
except Exception as exc:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_inputs(path, pattern):
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, pattern)))
    return [path]


def sample_spline_coeffs(coeffs, n_samples):
    coeffs = list(coeffs)
    t = np.array(coeffs[24:36], dtype=np.float64)
    c = [np.array(coeffs[i * 8 : (i + 1) * 8], dtype=np.float64) for i in range(3)]
    if np.abs(np.mean(t) - 1) < 1e-1:
        return None
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)
    return np.column_stack((x, y, z))


def find_worst_node(nodes, coeffs, n_samples=50):
    worst_idx = None
    worst_ratio = -np.inf
    for i in range(len(nodes)):
        sp = sample_spline_coeffs(coeffs[i], n_samples)
        if sp is None:
            continue
        center = nodes[i]
        distances = np.linalg.norm(sp - center, axis=1)
        xs = np.linspace(0, 1, n_samples)
        coeff = np.polyfit(xs, distances, 5)
        xs_fine = np.linspace(0, 1, 200)
        poly_vals = np.polyval(coeff, xs_fine)
        max_poly = float(np.max(poly_vals))
        max_data = float(np.max(distances))
        if max_data <= 0:
            continue
        ratio = max_poly / max_data
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_idx = i
    return worst_idx, worst_ratio


def plot_node(file_path, output_dir, params):
    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    n_samples = int(params.get("n_samples", 50))
    node_index = params.get("node_index")
    find_worst = bool(params.get("find_worst", True))
    save = bool(params.get("save", True))
    show = bool(params.get("show", False))
    image_ext = params.get("image_ext", ".png")

    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape((-1, k))
    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    if tree is None:
        return None

    # collect nodes in pre-order for consistent indexing
    nodes = []
    coeffs = []

    def dfs(node):
        if node is None:
            return
        nodes.append([node.data["x"], node.data["y"], node.data["z"]])
        coeffs.append(node.data.get("r", [0.0] * (k - 3)))
        dfs(node.left)
        dfs(node.right)

    dfs(tree)
    nodes = np.array(nodes, dtype=np.float32)
    coeffs = np.array(coeffs, dtype=object)

    if node_index is None and find_worst:
        node_index, ratio = find_worst_node(nodes, coeffs, n_samples=n_samples)
        if node_index is None:
            return None
        print(f"worst node: {node_index}, overshoot ratio ~ {ratio:.2f}")
    elif node_index is None:
        node_index = 0

    node_index = int(node_index)
    if node_index < 0 or node_index >= len(nodes):
        raise ValueError(f"node_index out of range: {node_index}")

    center = nodes[node_index]
    sp = sample_spline_coeffs(coeffs[node_index], n_samples)
    if sp is None:
        raise RuntimeError("Selected node has no valid spline coeffs.")

    distances = np.linalg.norm(sp - center, axis=1)
    xs = np.linspace(0, 1, n_samples)
    coeff = np.polyfit(xs, distances, 5)
    xs_fine = np.linspace(0, 1, 200)
    poly_vals = np.polyval(coeff, xs_fine)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, distances, "o", label="radius samples")
    ax.plot(xs_fine, poly_vals, "-", label="polyfit (deg=5)")
    ax.set_title(f"{os.path.basename(file_path)} | node {node_index}")
    ax.set_xlabel("sample parameter (0..1)")
    ax.set_ylabel("radius")
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    out_path = None
    if save:
        if not image_ext.startswith("."):
            image_ext = "." + image_ext
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, f"{base}_node{node_index}{image_ext}")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Debug polyfit overshoot for spline radii.")
    parser.add_argument("--config", default="debug_radius_polyfit_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_path = paths.get("input")
    output_dir = paths.get("output_dir", "debug_radius")
    pattern = params.get("pattern", "*.npy")
    max_files = params.get("max_files")

    if not input_path:
        raise SystemExit("Error: paths.input is required")

    files = iter_inputs(input_path, pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    for idx, file_path in enumerate(files, start=1):
        out_path = plot_node(file_path, output_dir, params)
        if out_path:
            print(f"[{idx}/{len(files)}] saved {out_path}")
        else:
            print(f"[{idx}/{len(files)}] skipped {os.path.basename(file_path)}")


if __name__ == "__main__":
    main()
