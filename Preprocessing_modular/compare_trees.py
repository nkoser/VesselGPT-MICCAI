import argparse
import os

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.interpolate import splev
except Exception:
    splev = None

import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_tree(file_path, k, mode, threshold, zero_mask=None):
    data = np.load(file_path)
    node_dim = k + 1 if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"} else k
    if data.ndim == 1:
        data = data.reshape((-1, node_dim))
    if zero_mask is not None:
        data = data.copy()
        data[zero_mask] = 0
    if threshold is not None:
        data = data.copy()
        data[np.abs(data) < threshold] = 0
    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    if tree is None:
        raise ValueError(f"Tree is empty: {file_path}")
    return tree


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


def sample_spline_coeffs(coeffs, n_samples):
    coeffs = list(coeffs)
    t = np.array(coeffs[24:])
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8 : (i * 8) + 8]) for i in range(3)]
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)
    return np.column_stack((x, y, z))


def draw_splines(ax, nodes, root_pos, n_samples, color, alpha, size, center_root):
    if splev is None:
        raise RuntimeError("scipy is required for spline visualization.")
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


def set_equal_aspect(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


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


def main():
    parser = argparse.ArgumentParser(description="Overlay two trees for visual comparison.")
    parser.add_argument("--config", default="compare_trees_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    tree_a_path = paths.get("tree_a")
    tree_b_path = paths.get("tree_b")
    if not tree_a_path or not tree_b_path:
        raise ValueError("paths.tree_a and paths.tree_b are required")

    k = int(params.get("k", 39))
    mode_a = params.get("mode_a", params.get("mode", "pre_order"))
    mode_b = params.get("mode_b", params.get("mode", "pre_order"))
    threshold_a = params.get("threshold_a")
    threshold_b = params.get("threshold_b")
    threshold_a = float(threshold_a) if threshold_a is not None else None
    threshold_b = float(threshold_b) if threshold_b is not None else None
    align_root = bool(params.get("align_root", True))
    mask_from = paths.get("mask_from")
    mask_threshold = params.get("mask_threshold")
    mask_threshold = float(mask_threshold) if mask_threshold is not None else None
    edges_source = params.get("edges_source", "each")

    zero_mask = None
    if mask_from:
        mask_data = np.load(mask_from)
        if mask_data.ndim == 1:
            mask_data = mask_data.reshape((-1, k))
        if mask_threshold is None:
            mask_threshold = 0.0
        zero_mask = np.all(np.abs(mask_data) <= mask_threshold, axis=1)

    tree_a = load_tree(tree_a_path, k, mode_a, threshold_a)
    tree_b = load_tree(tree_b_path, k, mode_b, threshold_b, zero_mask=zero_mask)

    xyz_a, edges_a, root_a, tree_a_nodes = build_arrays(tree_a)
    xyz_b, edges_b, root_b, tree_b_nodes = build_arrays(tree_b)

    if align_root:
        xyz_a = xyz_a - root_a
        xyz_b = xyz_b - root_b
        root_a = np.zeros(3, dtype=np.float32)
        root_b = np.zeros(3, dtype=np.float32)

    fig = plt.figure(figsize=tuple(params.get("figsize", [10, 10])))
    ax = fig.add_subplot(111, projection="3d")

    edge_alpha = float(params.get("edge_alpha", 0.4))
    edge_width = float(params.get("edge_width", 1.0))
    node_size = float(params.get("node_size", 8.0)) 

    color_a = params.get("color_a", "#1f77b4")
    color_b = params.get("color_b", "#ff7f0e")

    draw_edges_flag = bool(params.get("draw_edges", True))
    if draw_edges_flag:
        draw_edges(ax, xyz_a, edges_a, color=color_a, alpha=edge_alpha, width=edge_width)
        if edges_source == "a":
            draw_edges(ax, xyz_b, edges_a, color=color_b, alpha=edge_alpha, width=edge_width)
        else:
            draw_edges(ax, xyz_b, edges_b, color=color_b, alpha=edge_alpha, width=edge_width)

    ax.scatter(xyz_a[:, 0], xyz_a[:, 1], xyz_a[:, 2], s=node_size, c=color_a, alpha=0.8, label="A")
    ax.scatter(xyz_b[:, 0], xyz_b[:, 1], xyz_b[:, 2], s=node_size, c=color_b, alpha=0.8, label="B")

    if bool(params.get("draw_splines", False)):
        n_samples = int(params.get("spline_samples", 50))
        alpha = float(params.get("spline_alpha", 0.4))
        size = float(params.get("spline_size", 0.4))
        center_root = bool(params.get("spline_center_root", True))
        draw_splines(ax, tree_a_nodes, root_a, n_samples, params.get("spline_color_a", color_a), alpha, size, center_root)
        draw_splines(ax, tree_b_nodes, root_b, n_samples, params.get("spline_color_b", color_b), alpha, size, center_root)

    if bool(params.get("draw_root", True)):
        root_marker = params.get("root_marker", "x")
        root_size = float(params.get("root_size", 60.0))
        ax.scatter([root_a[0]], [root_a[1]], [root_a[2]], c=color_a, marker=root_marker, s=root_size)
        ax.scatter([root_b[0]], [root_b[1]], [root_b[2]], c=color_b, marker=root_marker, s=root_size)

    all_xyz = np.vstack((xyz_a, xyz_b))
    set_equal_aspect(ax, all_xyz)

    ax.set_title(params.get("title", "Tree overlay"))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    output_image = paths.get("output_image")
    if output_image:
        fig.savefig(output_image, dpi=200, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()

# %%
