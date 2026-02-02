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
    from scipy.interpolate import splprep, splev
    from scipy.spatial import KDTree
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


def get_branches(tree, k):
    if tree is None:
        return []

    branches = []

    def dfs(node, path):
        if node is None:
            return
        row = [node.data["x"], node.data["y"], node.data["z"]] + list(node.data.get("r", []))
        path.append(row[:k])
        if node.left is None and node.right is None:
            branches.append(path[:])
        else:
            if node.left:
                dfs(node.left, path)
            if node.right:
                dfs(node.right, path)
        path.pop()

    dfs(tree, [])
    return [np.array(branch, dtype=np.float32) for branch in branches]


def create_3d_spline(points, smooth):
    points = np.asarray(points, dtype=np.float64)
    mask = np.all(np.isfinite(points), axis=1)
    points = points[mask]
    if len(points) > 1:
        _, idx = np.unique(np.round(points, 6), axis=0, return_index=True)
        points = points[np.sort(idx)]
    if len(points) < 2:
        return None
    k = 3
    if len(points) <= 3:
        k = max(1, len(points) - 1)
    try:
        tck, _ = splprep(points.T, s=smooth, k=k)
        return tck
    except Exception:
        return None


def sample_spline(tck, num_samples):
    t = np.linspace(0, 1, num_samples)
    x, y, z = splev(t, tck)
    return np.vstack((x, y, z)).T


def set_equal_aspect(ax, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    centers = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_file(file_path, output_dir, params):
    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    centerline_samples = int(params.get("centerline_samples", 200))
    centerline_smooth = float(params.get("centerline_smooth", 0.0))
    node_stride = int(params.get("node_stride", 1))
    max_nodes = params.get("max_nodes")
    show_connections = bool(params.get("show_connections", True))
    show_nearest = bool(params.get("show_nearest", True))

    node_color = params.get("node_color", "#1f77b4")
    nearest_color = params.get("nearest_color", "#d62728")
    centerline_color = params.get("centerline_color", "#888888")
    connection_color = params.get("connection_color", "#2ca02c")

    node_size = float(params.get("node_size", 8))
    nearest_size = float(params.get("nearest_size", 8))
    line_width = float(params.get("line_width", 1.0))
    connection_alpha = float(params.get("connection_alpha", 0.4))

    elev = params.get("elev")
    azim = params.get("azim")
    save = bool(params.get("save", True))
    show = bool(params.get("show", False))
    image_ext = params.get("image_ext", ".png")

    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape((-1, k))
    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    branches = get_branches(tree, k)
    if not branches:
        return None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(os.path.basename(file_path))

    all_points = []

    for branch in branches:
        nodes = branch[:, :3]
        all_points.append(nodes)
        tck = create_3d_spline(nodes, centerline_smooth)
        if tck is None:
            continue
        sampled = sample_spline(tck, centerline_samples)
        all_points.append(sampled)
        ax.plot(sampled[:, 0], sampled[:, 1], sampled[:, 2], color=centerline_color, linewidth=1.0)

        if node_stride > 1:
            nodes_plot = nodes[::node_stride]
        else:
            nodes_plot = nodes
        if max_nodes is not None:
            nodes_plot = nodes_plot[: int(max_nodes)]

        ax.scatter(nodes_plot[:, 0], nodes_plot[:, 1], nodes_plot[:, 2], c=node_color, s=node_size)

        if show_connections or show_nearest:
            kdtree = KDTree(sampled)
            _, idxs = kdtree.query(nodes_plot)
            nearest = sampled[idxs]
            if show_nearest:
                ax.scatter(nearest[:, 0], nearest[:, 1], nearest[:, 2], c=nearest_color, s=nearest_size)
            if show_connections:
                for p, q in zip(nodes_plot, nearest):
                    ax.plot(
                        [p[0], q[0]],
                        [p[1], q[1]],
                        [p[2], q[2]],
                        color=connection_color,
                        alpha=connection_alpha,
                        linewidth=line_width,
                    )

    if all_points:
        xyz = np.vstack(all_points)
        set_equal_aspect(ax, xyz)

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    os.makedirs(output_dir, exist_ok=True)
    out_path = None
    if save:
        if not image_ext.startswith("."):
            image_ext = "." + image_ext
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_path = os.path.join(output_dir, base + image_ext)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize mapping from tree nodes to centerline spline.")
    parser.add_argument("--config", default="debug_centerline_mapping_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_path = paths.get("input")
    output_dir = paths.get("output_dir", "debug_centerlines")
    pattern = params.get("pattern", "*.npy")
    max_files = params.get("max_files")

    if not input_path:
        raise SystemExit("Error: paths.input is required")

    files = iter_inputs(input_path, pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    for idx, file_path in enumerate(files, start=1):
        out_path = plot_file(file_path, output_dir, params)
        if out_path:
            print(f"[{idx}/{len(files)}] saved {out_path}")
        else:
            print(f"[{idx}/{len(files)}] skipped {os.path.basename(file_path)}")


if __name__ == "__main__":
    main()
