import argparse
import os
import random
from glob import glob

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    from scipy.interpolate import splev, splprep
except Exception as exc:
    raise RuntimeError("scipy is required. Install with: pip install scipy") from exc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize, serialize_pre_order_k


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rotation_matrix(angle_degrees, axis):
    angle = np.radians(angle_degrees)
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )


def zero_root(data, mode):
    if mode not in {"pre_order", "post_order"}:
        raise ValueError("mode must be pre_order or post_order")
    root = data[0, :3] if mode == "pre_order" else data[-1, :3]
    not_zero_mask = np.mean(data, axis=1) != 0
    data[not_zero_mask, :3] = data[not_zero_mask, :3] - root
    return data, root, not_zero_mask


def all_elements_equal(values):
    return all(np.allclose(x, values[0], atol=1e-4) for x in values)


def limpiarRadiosSplines(tck):
    cleaned = []
    array_list = tck[1]
    for arr in array_list:
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) < 8:
            arr = np.pad(arr, (0, 8 - len(arr)), mode="edge")
        cleaned.extend(arr[:8])

    arr = np.asarray(tck[0], dtype=np.float32)
    if len(arr) < 12:
        arr = np.pad(arr, (0, 12 - len(arr)), mode="edge")
    cleaned.extend(arr[:12])
    return cleaned


def sample_spline_coeffs(coeffs, n_samples):
    coeffs = list(coeffs)
    t = np.array(coeffs[24:])
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8: (i * 8) + 8]) for i in range(3)]
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples)
    x, y, z = splev(u, tck)
    return np.column_stack((x, y, z))


def _coerce_len(values, target_len):
    values = np.asarray(values, dtype=np.float32)
    if len(values) >= target_len:
        return values[:target_len]
    return np.pad(values, (0, target_len - len(values)), mode="edge")


def pack_spline_coeffs(tck, target_ctrl=8, target_knot=12):
    t, c, _k = tck
    cx = _coerce_len(c[0], target_ctrl)
    cy = _coerce_len(c[1], target_ctrl)
    cz = _coerce_len(c[2], target_ctrl)
    tt = _coerce_len(t, target_knot)
    return np.concatenate((cx, cy, cz, tt))


def _transform_spline_coeffs(coeffs, root, rot, scale):
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if coeffs.size < 36:
        return coeffs
    ctrl = coeffs[:24].reshape(3, 8).T
    if root is not None:
        ctrl = ctrl - root
    if rot is not None:
        ctrl = ctrl @ rot.T
    if scale is not None:
        ctrl = ctrl / scale
    return np.concatenate((ctrl.T.reshape(24), coeffs[24:36]))


def build_spline_dataset(
    data,
    mode,
    n_rotations,
    n_samples,
    smooth,
    enable_rotation,
    enable_scaling,
    refit_splines,
):
    root = data[-1, :3] if mode == "post_order" else data[0, :3]
    not_zero_mask = np.mean(data, axis=1) != 0

    outputs = []
    for r in range(n_rotations):
        # rotation disabled by default to match datasets.ipynb
        if enable_rotation and r != 0:
            angle = random.randint(10, 350)
            axis = np.random.rand(3)
            rot = rotation_matrix(angle, axis)
        else:
            rot = None

        spline_points = np.zeros((len(data) * n_samples, 3), dtype=np.float32)
        j = 0
        for _, datum in enumerate(data):
            if np.any(datum):
                if all_elements_equal(datum[3:11]):
                    points = np.hstack(
                        (
                            np.full(n_samples, datum[0]).reshape(-1, 1),
                            np.full(n_samples, datum[1]).reshape(-1, 1),
                            np.full(n_samples, datum[2]).reshape(-1, 1),
                        )
                    ) - root
                else:
                    points = sample_spline_coeffs(datum[3:], n_samples=n_samples) - root
                spline_points[j * n_samples: (j + 1) * n_samples] = points
            else:
                spline_points[j * n_samples: (j + 1) * n_samples] = np.zeros((n_samples, 3))
            j += 1

        # zero root and rotations
        data_xyz = data[:, :3].copy()
        data_xyz[not_zero_mask, :] = data_xyz[not_zero_mask, :] - root

        if rot is not None:
            data_xyz = data_xyz @ rot.T
            spline_points = spline_points @ rot.T

        scale = None
        if enable_scaling:
            all_data = np.vstack((data_xyz, spline_points))
            abs_max = abs(all_data).max()
            if abs_max > 0:
                all_data = all_data / abs_max
                scale = abs_max
            data_xyz = all_data[: len(data_xyz), :]
            spline_points = all_data[len(data_xyz):, :]

        if refit_splines:
            data_splines = []
            for i in range(0, len(data) * n_samples, n_samples):
                segment = spline_points[i: i + n_samples]
                if np.any(segment):
                    xs = segment[:, 0].flatten()
                    ys = segment[:, 1].flatten()
                    zs = segment[:, 2].flatten()

                    if all_elements_equal(xs):
                        t = np.ones(12, dtype=np.float32)
                        c = [xs[:8], ys[:8], zs[:8]]
                        tck = (t, c, 3)
                    else:
                        try:
                            tck, _ = splprep([xs, ys, zs], s=smooth, per=True, nest=12, k=3)
                        except Exception:
                            tck = None

                    if tck is None:
                        datum = data[i // n_samples]
                        if np.any(datum):
                            new_row = _transform_spline_coeffs(datum[3:], root, rot, scale)
                        else:
                            new_row = np.zeros(36, dtype=np.float32)
                    else:
                        new_row = limpiarRadiosSplines(tck)
                    data_splines.append(new_row)
                else:
                    data_splines.append(np.zeros(36))

            data_splines = np.array(data_splines, dtype=np.float32)
        else:
            data_splines = []
            for datum in data:
                if np.any(datum):
                    data_splines.append(
                        _transform_spline_coeffs(datum[3:], root, rot, scale)
                    )
                else:
                    data_splines.append(np.zeros(36))
            data_splines = np.array(data_splines, dtype=np.float32)
        new_data = np.hstack((data_xyz, data_splines))
        outputs.append(new_data)

    return outputs


def iter_files(input_dir, pattern):
    return sorted(glob(os.path.join(input_dir, pattern)))


def erase_all_files(folder_path):
    if not os.path.isdir(folder_path):
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def process_file(
    file_path,
    output_dir,
    k,
    mode,
    n_rotations,
    overwrite,
    n_samples,
    smooth,
    enable_rotation,
    enable_scaling,
    refit_splines,
):
    data = np.load(file_path)
    if data.ndim == 1:
        data = data.reshape((-1, k))
    base = np.array(data, dtype=np.float32).reshape((-1, k))

    if k == 39:
        outputs = build_spline_dataset(
            base,
            mode,
            n_rotations,
            n_samples,
            smooth,
            enable_rotation,
            enable_scaling,
            refit_splines,
        )
    else:
        base, _root, not_zero_mask = zero_root(base, mode)
        outputs = [base]
        for i in range(n_rotations):
            angle = random.randint(10, 350)
            axis = np.random.rand(3)
            rot = rotation_matrix(angle, axis)
            rotated = base.copy()
            rotated[not_zero_mask, :3] = rotated[not_zero_mask, :3] @ rot.T
            outputs.append(rotated)

    written = 0
    skipped = 0
    failures_total = 0
    for idx, arr in enumerate(outputs):
        name = os.path.basename(file_path)
        if idx > 0:
            name = f"rot{idx}-" + name
        out_path = os.path.join(output_dir, name)
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue
        np.save(out_path, arr)
        written += 1

    return written, skipped, failures_total


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset: depth cut, zero-root, rotations, spline refit (k=39)."
    )
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--input", default=None, help="Input folder with .npy trees")
    parser.add_argument("--output", default=None, help="Output folder")
    parser.add_argument("--k", type=int, default=39, help="Feature dimension (default: 39)")
    parser.add_argument("--mode", default="pre_order", choices=["pre_order", "post_order"])
    parser.add_argument("--n-rotations", type=int, default=0)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--spline-samples", type=int, default=50)
    parser.add_argument("--spline-smooth", type=float, default=0.0000001)
    parser.add_argument("--enable-rotation", action="store_true")
    parser.add_argument("--enable-scaling", action="store_true")
    parser.add_argument("--refit-splines", dest="refit_splines", action="store_true", default=True)
    parser.add_argument("--no-refit-splines", dest="refit_splines", action="store_false")
    parser.add_argument("--erase-output", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        default_path = os.path.join(os.path.dirname(__file__), "prepare_dataset_config.yaml")
        if os.path.exists(default_path):
            config_path = default_path

    if config_path:
        cfg = load_config(config_path)
        paths = cfg.get("paths", {})
        params = cfg.get("params", {})
        args.input = paths.get("input", args.input)
        args.output = paths.get("output", args.output)
        args.k = int(params.get("k", args.k))
        args.mode = params.get("mode", args.mode)
        args.n_rotations = int(params.get("n_rotations", args.n_rotations))
        args.pattern = params.get("pattern", args.pattern)
        args.overwrite = bool(params.get("overwrite", args.overwrite))
        if "seed" in params:
            args.seed = params.get("seed")
        if "spline_samples" in params:
            args.spline_samples = int(params.get("spline_samples"))
        if "spline_smooth" in params:
            args.spline_smooth = float(params.get("spline_smooth"))
        if "enable_rotation" in params:
            args.enable_rotation = bool(params.get("enable_rotation"))
        if "enable_scaling" in params:
            args.enable_scaling = bool(params.get("enable_scaling"))
        if "refit_splines" in params:
            args.refit_splines = bool(params.get("refit_splines"))
        if "erase_output" in params:
            args.erase_output = bool(params.get("erase_output"))

    if not args.input or not args.output:
        raise SystemExit("Error: --input and --output are required (or provide them in a config).")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    if args.erase_output:
        erase_all_files(args.output)

    files = iter_files(args.input, args.pattern)
    total_written = 0
    total_skipped = 0
    total_failures = 0

    if args.n_rotations <= 0:
        raise SystemExit("Error: n_rotations must be >= 1 to match datasets.ipynb behavior.")

    for idx, file_path in enumerate(files, start=1):
        written, skipped, failures = process_file(
            file_path,
            args.output,
            args.k,
            args.mode,
            args.n_rotations,
            args.overwrite,
            args.spline_samples,
            args.spline_smooth,
            args.enable_rotation,
            args.enable_scaling,
            args.refit_splines,
        )
        total_written += written
        total_skipped += skipped
        total_failures += failures
        print(f"[{idx}/{len(files)}] {os.path.basename(file_path)} -> +{written} (-{skipped})")

    print(f"done: {total_written} written, {total_skipped} skipped, spline failures {total_failures}")


if __name__ == "__main__":
    main()
