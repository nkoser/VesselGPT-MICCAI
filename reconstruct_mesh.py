import argparse
import os
import sys
from glob import glob

import numpy as np

try:
    import pyvista as pv
except Exception:
    pv = None

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.interpolate import splprep, splev
except Exception as exc:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize
from sdf import union
from sdf.d3 import vessel3


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_inputs(path, pattern):
    if os.path.isdir(path):
        return sorted(glob(os.path.join(path, pattern)))
    return [path]


def _coerce_row(row, k):
    row = list(row)
    head = row[:3]
    tail = row[3:]
    needed = k - 3
    if len(tail) < needed:
        tail = tail + [0.0] * (needed - len(tail))
    return head + tail[:needed]


def get_branches(tree, k):
    if tree is None:
        return []

    branches = []

    def dfs(node, path):
        if node is None:
            return
        row = [node.data["x"], node.data["y"], node.data["z"]] + list(node.data.get("r", []))
        path.append(_coerce_row(row, k))
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
    if len(points) < 2:
        return None
    k = 3
    if len(points) <= 3:
        k = max(1, len(points) - 1)
    tck, _ = splprep(points.T, s=smooth, k=k)
    return tck


def sample_spline(tck, num_samples):
    t = np.linspace(0, 1, num_samples)
    x, y, z = splev(t, tck)
    return np.vstack((x, y, z)).T


def build_sdf(tree, k, centerline_samples, centerline_smooth):
    branches = get_branches(tree, k)
    if not branches:
        return None

    vessels = []
    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]
        tck = create_3d_spline(nodes, centerline_smooth)
        if tck is None:
            continue
        sampled = sample_spline(tck, centerline_samples)
        vessels.append(vessel3(sampled, nodes, splines))

    if not vessels:
        return None
    if len(vessels) == 1:
        return vessels[0]
    return union(*vessels)


def parse_int(value, default):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return default


def process_file(path, output_dir, params):
    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    output_ext = params.get("output_ext", ".stl")
    overwrite = bool(params.get("overwrite", False))
    centerline_samples = int(params.get("centerline_samples", 100))
    centerline_smooth = float(params.get("centerline_smooth", 0.0))
    samples_coarse = parse_int(params.get("samples_coarse"), 32)
    samples_fine = parse_int(params.get("samples_fine"), 262144)
    sparse = bool(params.get("sparse", True))
    use_bounds = bool(params.get("use_bounds", True))

    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape((-1, k))
    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    sdf_obj = build_sdf(tree, k, centerline_samples, centerline_smooth)
    if sdf_obj is None:
        return "skip", None

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, base + output_ext)
    if os.path.exists(out_path) and not overwrite:
        return "skip", out_path

    os.makedirs(output_dir, exist_ok=True)
    bounds = sdf_obj.save(out_path, samples=samples_coarse, sparse=sparse)
    if samples_fine:
        if use_bounds:
            sdf_obj.save(out_path, samples=samples_fine, bounds=bounds, sparse=sparse)
        else:
            sdf_obj.save(out_path, samples=samples_fine, sparse=sparse)
    return "ok", out_path


def preview_mesh(mesh_path, params):
    if pv is None:
        print("preview skipped: pyvista not installed")
        return
    color = params.get("preview_color", "red")
    off_screen = bool(params.get("preview_offscreen", True))
    image_dir = params.get("preview_image_dir")
    plotter = pv.Plotter(off_screen=off_screen)
    mesh = pv.read(mesh_path)
    plotter.add_mesh(mesh, color=color, show_edges=False)
    plotter.add_axes()
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(
            image_dir, os.path.splitext(os.path.basename(mesh_path))[0] + ".png"
        )
        plotter.screenshot(image_path)
        plotter.close()
        print(f"preview saved {image_path}")
    else:
        plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Reconstruct mesh from tree splines via SDF.")
    parser.add_argument("--config", default="reconstruct_mesh_config.yaml", help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Input .npy file or directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for meshes")
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    input_path = args.input or paths.get("input")
    output_dir = args.output_dir or paths.get("output_dir")
    pattern = params.get("pattern", "*.npy")
    max_files = params.get("max_files")
    preview = bool(params.get("preview", False))
    preview_max = params.get("preview_max")

    if not input_path or not output_dir:
        raise SystemExit("Error: --input and --output-dir are required (or provide them in config).")

    files = iter_inputs(input_path, pattern)
    if max_files is not None:
        files = files[: int(max_files)]

    total = len(files)
    written = 0
    for idx, path in enumerate(files, start=1):
        status, out_path = process_file(path, output_dir, params)
        if status == "ok":
            written += 1
            print(f"[{idx}/{total}] saved {out_path}")
            if preview:
                if preview_max is None or written <= int(preview_max):
                    preview_mesh(out_path, params)
        else:
            print(f"[{idx}/{total}] skipped {os.path.basename(path)}")

    print(f"done: {written}/{total} written")


if __name__ == "__main__":
    main()
