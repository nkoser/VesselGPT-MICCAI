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
from sdf.d3 import vessel3, vessel3_robust, vessel3_stable 


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


def get_edges(tree):
    edges = []

    def dfs(node):
        if node is None:
            return
        if node.left is not None:
            edges.append((node, node.left))
            dfs(node.left)
        if node.right is not None:
            edges.append((node, node.right))
            dfs(node.right)

    dfs(tree)
    return edges


def create_3d_spline(points, smooth):
    points = np.asarray(points, dtype=np.float64)
    # drop NaN/Inf
    mask = np.all(np.isfinite(points), axis=1)
    points = points[mask]
    # drop duplicates (approx)
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


def sample_spline_coeffs(coeffs, n_samples):
    coeffs = list(coeffs)
    if len(coeffs) < 36:
        return None
    t = np.array(coeffs[24:36], dtype=np.float64)
    if t.shape[0] != 12:
        return None
    if not np.all(np.isfinite(t)):
        return None
    t = np.where(np.abs(t - 1) < 0.01, 1.0, t)
    c = [np.array(coeffs[i * 8 : (i + 1) * 8], dtype=np.float64) for i in range(3)]
    if not all(np.all(np.isfinite(ci)) for ci in c):
        return None
    tck = (t, c, 3)
    u = np.linspace(0, 1, n_samples, endpoint=False)
    x, y, z = splev(u, tck)
    return np.column_stack((x, y, z))


def align_ring(prev, curr, allow_flip=True):
    if prev is None or curr is None:
        return curr
    if prev.shape != curr.shape:
        return curr

    def best_shift_score(reference, ring):
        n = reference.shape[0]
        best_shift = 0
        best_score = None
        for shift in range(n):
            rolled = np.roll(ring, -shift, axis=0)
            score = np.sum((reference - rolled) ** 2)
            if best_score is None or score < best_score:
                best_score = score
                best_shift = shift
        return np.roll(ring, -best_shift, axis=0), best_score

    aligned, score = best_shift_score(prev, curr)
    if allow_flip:
        flipped = curr[::-1].copy()
        aligned_flip, score_flip = best_shift_score(prev, flipped)
        if score_flip is not None and (score is None or score_flip < score):
            return aligned_flip
    return aligned


def ring_radius(points):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return float(np.median(distances))


def build_loft_polydata(rings, add_caps):
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError("VTK is required for loft reconstruction.") from exc

    if len(rings) < 2:
        return None

    n_points = rings[0].shape[0]
    points = vtk.vtkPoints()
    polys = vtk.vtkCellArray()

    for ring in rings:
        if ring.shape[0] != n_points:
            return None
        for point in ring:
            points.InsertNextPoint(float(point[0]), float(point[1]), float(point[2]))

    for r in range(len(rings) - 1):
        for i in range(n_points):
            i0 = r * n_points + i
            i1 = r * n_points + (i + 1) % n_points
            i2 = (r + 1) * n_points + (i + 1) % n_points
            i3 = (r + 1) * n_points + i
            quad = vtk.vtkQuad()
            quad.GetPointIds().SetId(0, i0)
            quad.GetPointIds().SetId(1, i1)
            quad.GetPointIds().SetId(2, i2)
            quad.GetPointIds().SetId(3, i3)
            polys.InsertNextCell(quad)

    if add_caps:
        start_center = np.mean(rings[0], axis=0)
        end_center = np.mean(rings[-1], axis=0)
        start_id = points.InsertNextPoint(float(start_center[0]), float(start_center[1]), float(start_center[2]))
        end_id = points.InsertNextPoint(float(end_center[0]), float(end_center[1]), float(end_center[2]))

        for i in range(n_points):
            i0 = i
            i1 = (i + 1) % n_points
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, start_id)
            tri.GetPointIds().SetId(1, i1)
            tri.GetPointIds().SetId(2, i0)
            polys.InsertNextCell(tri)

        base = (len(rings) - 1) * n_points
        for i in range(n_points):
            i0 = base + i
            i1 = base + (i + 1) % n_points
            tri = vtk.vtkTriangle()
            tri.GetPointIds().SetId(0, end_id)
            tri.GetPointIds().SetId(1, i0)
            tri.GetPointIds().SetId(2, i1)
            polys.InsertNextCell(tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)
    return polydata


def build_loft_mesh(tree, k, params):
    try:
        import vtk
    except Exception as exc:
        raise RuntimeError("VTK is required for loft reconstruction.") from exc

    branches = get_branches(tree, k)
    if not branches:
        return None

    spline_samples = int(params.get("spline_samples", 50))
    loft_align = bool(params.get("loft_align", True))
    loft_allow_flip = bool(params.get("loft_allow_flip", True))
    loft_caps = bool(params.get("loft_caps", False))
    loft_clean = bool(params.get("loft_clean", True))
    loft_min_radius = float(params.get("loft_min_radius", 0.0))
    append = vtk.vtkAppendPolyData()

    for branch in branches:
        splines = branch[:, 3:]
        rings = []
        prev = None
        for coeffs in splines:
            points = sample_spline_coeffs(coeffs, spline_samples)
            if points is None:
                if prev is not None:
                    points = prev.copy()
                else:
                    continue
            if not np.all(np.isfinite(points)):
                if prev is not None:
                    points = prev.copy()
                else:
                    continue
            if loft_min_radius > 0 and ring_radius(points) < loft_min_radius:
                if prev is not None:
                    points = prev.copy()
                else:
                    continue
            if loft_align:
                points = align_ring(prev, points, allow_flip=loft_allow_flip)
            rings.append(points)
            prev = points

        poly = build_loft_polydata(rings, loft_caps)
        if poly is None:
            continue
        append.AddInputData(poly)

    append.Update()
    polydata = append.GetOutput()
    if loft_clean:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        polydata = cleaner.GetOutput()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(polydata)
    tri.Update()
    return tri.GetOutput()


def build_sdf(tree, k, centerline_samples, centerline_smooth, params):
    branches = get_branches(tree, k)
    if not branches:
        return None

    vessels = []
    recon_mode = params.get("recon_mode", "legacy")
    if recon_mode not in {"legacy", "stable", "sdf_offset"}:
        recon_mode = "legacy"
    legacy_variant = params.get("legacy_variant", "original")  # original | robust

    radius_mode = params.get("radius_mode", "median")
    radius_percentile = params.get("radius_percentile", 90)
    radius_cap = params.get("radius_cap")
    center_mode = params.get("center_mode", "node")
    fallback_radius = params.get("fallback_radius", 0.0)
    spline_samples = int(params.get("spline_samples", 50))
    for branch in branches:
        nodes = branch[:, :3]
        splines = branch[:, 3:]
        centerline_nodes = nodes
        if recon_mode == "sdf_offset":
            centers = []
            last_center = None
            for coeffs, node in zip(splines, nodes):
                points = sample_spline_coeffs(coeffs, spline_samples)
                if points is None:
                    center = last_center if last_center is not None else node
                else:
                    center = np.mean(points, axis=0)
                centers.append(center)
                last_center = center
            centerline_nodes = np.array(centers, dtype=np.float32)
        tck = create_3d_spline(centerline_nodes, centerline_smooth)
        if tck is None:
            continue
        sampled = sample_spline(tck, centerline_samples)
        if recon_mode in {"stable", "sdf_offset"}:
            centers_for_radius = centerline_nodes if recon_mode == "sdf_offset" else nodes
            vessels.append(
                vessel3_stable(
                    sampled,
                    centers_for_radius,
                    splines,
                    radius_mode=radius_mode,
                    radius_percentile=radius_percentile,
                    radius_cap=radius_cap,
                    center_mode=center_mode,
                    fallback_radius=fallback_radius,
                )
            )
        else:
            if legacy_variant == "robust":
                vessels.append(vessel3_robust(sampled, nodes, splines))
            else:
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
    save_centerline = bool(params.get("save_centerline", False))
    centerline_out_dir = params.get("centerline_output_dir") or output_dir
    centerline_suffix = params.get("centerline_suffix", "_centerline.npy")
    centerline_format = params.get("centerline_format", "npy").lower()
    centerline_tube_radius = float(params.get("centerline_tube_radius", 0.0))
    centerline_tube_sides = int(params.get("centerline_tube_sides", 8))
    centerline_mode = params.get("centerline_mode", "branches")
    centerline_edge_samples = int(params.get("centerline_edge_samples", 2))

    data = np.load(path)
    if data.ndim == 1:
        data = data.reshape((-1, k))
    serial = list(data.flatten())
    tree = deserialize(serial, mode=mode, k=k)
    recon_mode = params.get("recon_mode", "legacy")
    if recon_mode not in {"legacy", "stable", "sdf_offset", "loft"}:
        recon_mode = "legacy"

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, base + output_ext)
    if os.path.exists(out_path) and not overwrite:
        return "skip", out_path

    os.makedirs(output_dir, exist_ok=True)

    # Optional: save resampled centerlines
    if save_centerline:
        centerline_points = []
        centerline_branches = []
        if centerline_mode == "edges":
            edges = get_edges(tree)
            samples = max(2, centerline_edge_samples)
            for e_idx, (parent, child) in enumerate(edges):
                p0 = np.array([parent.data["x"], parent.data["y"], parent.data["z"]], dtype=np.float32)
                p1 = np.array([child.data["x"], child.data["y"], child.data["z"]], dtype=np.float32)
                t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
                sampled = p0[None, :] * (1.0 - t[:, None]) + p1[None, :] * t[:, None]
                branch_col = np.full((sampled.shape[0], 1), e_idx, dtype=np.int32)
                arr = np.hstack((sampled.astype(np.float32), branch_col))
                centerline_points.append(arr)
                centerline_branches.append(sampled.astype(np.float32))
        else:
            branches = get_branches(tree, k)
            for b_idx, branch in enumerate(branches):
                nodes = branch[:, :3]
                tck = create_3d_spline(nodes, centerline_smooth)
                if tck is None:
                    continue
                sampled = sample_spline(tck, centerline_samples)
                if sampled is None or len(sampled) == 0:
                    continue
                branch_col = np.full((sampled.shape[0], 1), b_idx, dtype=np.int32)
                arr = np.hstack((sampled.astype(np.float32), branch_col))
                centerline_points.append(arr)
                centerline_branches.append(sampled.astype(np.float32))
        if centerline_points:
            os.makedirs(centerline_out_dir, exist_ok=True)
            if centerline_format in {"stl", "obj", "vtp"}:
                try:
                    import vtk
                except Exception as exc:
                    raise RuntimeError("VTK is required for centerline_format=stl/obj/vtp.") from exc
                append = vtk.vtkAppendPolyData()
                for b_idx, pts in enumerate(centerline_branches):
                    if pts.shape[0] < 2:
                        continue
                    vtk_pts = vtk.vtkPoints()
                    for p in pts:
                        vtk_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
                    polyline = vtk.vtkPolyLine()
                    polyline.GetPointIds().SetNumberOfIds(pts.shape[0])
                    for i in range(pts.shape[0]):
                        polyline.GetPointIds().SetId(i, i)
                    cells = vtk.vtkCellArray()
                    cells.InsertNextCell(polyline)
                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(vtk_pts)
                    polydata.SetLines(cells)
                    append.AddInputData(polydata)
                append.Update()
                cl_poly = append.GetOutput()

                # STL/OBJ require surface triangles; convert lines -> tube if requested
                if centerline_format in {"stl", "obj"}:
                    tube = vtk.vtkTubeFilter()
                    tube.SetInputData(cl_poly)
                    tube.SetNumberOfSides(max(3, centerline_tube_sides))
                    if centerline_tube_radius > 0:
                        tube.SetRadius(centerline_tube_radius)
                    else:
                        tube.SetRadius(0.001)
                    tube.CappingOn()
                    tube.Update()
                    cl_poly = tube.GetOutput()

                if centerline_format == "stl":
                    cl_ext = centerline_suffix if centerline_suffix.lower().endswith(".stl") else ".stl"
                    cl_path = os.path.join(centerline_out_dir, base + cl_ext)
                    writer = vtk.vtkSTLWriter()
                    writer.SetFileName(cl_path)
                    writer.SetInputData(cl_poly)
                    writer.Write()
                elif centerline_format == "obj":
                    cl_ext = centerline_suffix if centerline_suffix.lower().endswith(".obj") else ".obj"
                    cl_path = os.path.join(centerline_out_dir, base + cl_ext)
                    writer = vtk.vtkOBJWriter()
                    writer.SetFileName(cl_path)
                    writer.SetInputData(cl_poly)
                    writer.Write()
                else:
                    cl_ext = centerline_suffix if centerline_suffix.lower().endswith(".vtp") else ".vtp"
                    cl_path = os.path.join(centerline_out_dir, base + cl_ext)
                    writer = vtk.vtkXMLPolyDataWriter()
                    writer.SetFileName(cl_path)
                    writer.SetInputData(cl_poly)
                    writer.Write()
            else:
                cl_path = os.path.join(centerline_out_dir, base + centerline_suffix)
                np.save(cl_path, np.vstack(centerline_points))

    if recon_mode == "loft":
        try:
            import vtk
        except Exception as exc:
            raise RuntimeError("VTK is required for loft reconstruction.") from exc
        mesh = build_loft_mesh(tree, k, params)
        if mesh is None:
            return "skip", None
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".stl":
            writer = vtk.vtkSTLWriter()
        elif ext == ".vtp":
            writer = vtk.vtkXMLPolyDataWriter()
        elif ext == ".ply":
            writer = vtk.vtkPLYWriter()
        else:
            raise RuntimeError(f"Unsupported loft output extension: {ext}")
        writer.SetFileName(out_path)
        writer.SetInputData(mesh)
        writer.Write()
    else:
        sdf_obj = build_sdf(tree, k, centerline_samples, centerline_smooth, params)
        if sdf_obj is None:
            return "skip", None
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
