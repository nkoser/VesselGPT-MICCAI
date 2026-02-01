import numpy as np
import matplotlib.pyplot as plt
import vtk
import os

def rdp(points, epsilon):
    """
    Simplify a curve using the Ramer-Douglas-Peucker algorithm.

    Parameters:
    - points: List of (x, y) coordinates representing the curve.
    - epsilon: The maximum distance allowed between the original curve and the simplified curve.

    Returns:
    - List of simplified (x, y) coordinates.
    """
    if len(points) <= 2:
        return points

    # Find the point with the maximum distance
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        d = np.linalg.norm(np.cross(
            np.array(points[end]) - np.array(points[0]),
            np.array(points[i]) - np.array(points[0])
        )) / np.linalg.norm(np.array(points[end]) - np.array(points[0]))

        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        rec_results1 = rdp(points[:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)

        # Combine the results
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[end]]

    return result


def interpolarRDP (centerline, epsilon = 0.05):

    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    lista_puntos = []
    lista_rama = []
    for j in range(centerline.GetNumberOfCells()):#iterar por ramas  
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints();# number of points of the branch
        for i in range(numberOfCellPoints):
            originalPointId = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(originalPointId)
            lista_puntos.append(list(point))
            
        #Ya tengo la lista de puntos de la rama entonces la resampleo y la agrego a la linea punto y polydata
        rama_resampleada = rdp(lista_puntos, epsilon)
        polyline = vtk.vtkPolyLine()
        for point in rama_resampleada:
            newPointId = points.InsertNextPoint(point)
            polyline.GetPointIds().InsertNextId(newPointId)
        cellarray.InsertNextCell(polyline)
        lista_rama.append(lista_puntos)
        lista_puntos = []
    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    return resampleada

def interpolarRDP_conRadio(centerline, epsilon=0.05):
    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()
    lista_rama = []

    # Prepare radius array output
    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")

    # Get original radius array
    original_radius = centerline.GetPointData().GetArray("Radius")

    for j in range(centerline.GetNumberOfCells()):  # iterate over branches  
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()
        puntos_rama = []
        radios_rama = []

        for i in range(numberOfCellPoints):
            pid = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(pid)
            radius = original_radius.GetValue(pid)
            puntos_rama.append(list(point))
            radios_rama.append(radius)

        # Apply RDP resampling to the branch
        rama_resampleada = rdp(puntos_rama, epsilon)

        # Reconstruct polyline and resample radius
        polyline = vtk.vtkPolyLine()
        for point in rama_resampleada:
            newPointId = points.InsertNextPoint(point)

            # Find index of the point in original list to get corresponding radius
            index = puntos_rama.index(point)
            radius_array.InsertNextValue(radios_rama[index])

            polyline.GetPointIds().InsertNextId(newPointId)

        cellarray.InsertNextCell(polyline)
        lista_rama.append(puntos_rama)

    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    resampleada.GetPointData().AddArray(radius_array)

    return resampleada


def _clean_polyline(points, scalars=None):
    if len(points) <= 1:
        return points, scalars
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-12:
            keep.append(i)
    points = points[keep]
    if scalars is not None:
        scalars = scalars[keep]
    return points, scalars


def _resample_polyline_uniform(points, scalars, step):
    if len(points) <= 1:
        return points, scalars
    pts = np.asarray(points, dtype=np.float64)
    scal = None if scalars is None else np.asarray(scalars, dtype=np.float64)
    pts, scal = _clean_polyline(pts, scal)
    if len(pts) <= 1:
        return pts, scal

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if not np.isfinite(total) or total <= 1e-12:
        return pts, scal

    s = np.concatenate([[0.0], np.cumsum(seg)])
    step = float(step)
    if not np.isfinite(step) or step <= 0:
        step = total

    s_new = np.arange(0.0, total, step)
    if s_new.size == 0 or s_new[-1] < total:
        s_new = np.concatenate([s_new, [total]])

    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    z = np.interp(s_new, s, pts[:, 2])
    new_pts = np.column_stack([x, y, z])

    new_scal = None
    if scal is not None:
        new_scal = np.interp(s_new, s, scal)
    return new_pts, new_scal


def _cap_samples(points, scalars, max_points):
    if max_points is None:
        return points, scalars
    max_points = int(max_points)
    if max_points <= 0 or len(points) <= max_points:
        return points, scalars
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    points = points[idx]
    if scalars is not None:
        scalars = scalars[idx]
    return points, scalars


def _compute_point_degrees(centerline):
    n = centerline.GetNumberOfPoints()
    deg = np.zeros(n, dtype=np.int32)
    lines = centerline.GetLines()
    lines.InitTraversal()
    idlist = vtk.vtkIdList()
    while lines.GetNextCell(idlist):
        m = idlist.GetNumberOfIds()
        for i in range(m - 1):
            a = idlist.GetId(i)
            b = idlist.GetId(i + 1)
            deg[a] += 1
            deg[b] += 1
    return deg


def _resample_polyline_radius_adaptive(
    points,
    scalars,
    step_min,
    step_max,
    radius_scale,
    radius_mode="inverse",
    drds_threshold=0.0,
    drds_boost=1.0,
    junction_s=None,
    junction_window=0.0,
    junction_factor=1.0,
):
    if len(points) <= 1:
        return points, scalars
    pts = np.asarray(points, dtype=np.float64)
    scal = None if scalars is None else np.asarray(scalars, dtype=np.float64)
    pts, scal = _clean_polyline(pts, scal)
    if len(pts) <= 1:
        return pts, scal

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if not np.isfinite(total) or total <= 1e-12:
        return pts, scal

    s = np.concatenate([[0.0], np.cumsum(seg)])
    step_min = float(step_min)
    step_max = float(step_max)
    radius_scale = float(radius_scale)
    radius_mode = (radius_mode or "inverse").lower()
    drds_threshold = float(drds_threshold)
    drds_boost = float(drds_boost)
    junction_window = float(junction_window)
    junction_factor = float(junction_factor)
    if junction_s is None:
        junction_s = []
    junction_s = np.asarray(junction_s, dtype=np.float64)
    if junction_s.size > 0:
        junction_s = np.unique(np.clip(junction_s, 0.0, total))
        junction_s.sort()

    if scal is None:
        return _resample_polyline_uniform(pts, scal, step_max)

    if radius_mode == "drds":
        drds = np.gradient(scal, s)
        drds = np.abs(drds)
        if not np.all(np.isfinite(drds)):
            drds = np.where(np.isfinite(drds), drds, 0.0)
    else:
        drds = None

    def interp_with_ptr(x_val, xs, ys, ptr):
        n = len(xs)
        while ptr < n - 2 and x_val > xs[ptr + 1]:
            ptr += 1
        x0, x1 = xs[ptr], xs[ptr + 1]
        y0, y1 = ys[ptr], ys[ptr + 1]
        if x1 <= x0:
            return float(y0), ptr
        t = (x_val - x0) / (x1 - x0)
        return float(y0 + t * (y1 - y0)), ptr

    s_new = [0.0]
    cur = 0.0
    ptr_r = 0
    ptr_g = 0
    max_iter = int(np.ceil(total / max(step_min, 1e-12))) + 10
    junction_ptr = 0

    for _ in range(max_iter):
        if radius_mode == "drds":
            g, ptr_g = interp_with_ptr(cur, s, drds, ptr_g)
            if not np.isfinite(g):
                step = step_max
            else:
                if g <= drds_threshold:
                    step = step_max
                else:
                    step = step_max / (1.0 + drds_boost * (g - drds_threshold))
        else:
            r, ptr_r = interp_with_ptr(cur, s, scal, ptr_r)
            if not np.isfinite(r):
                step = step_max
            else:
                r = max(r, 1e-12)
                if radius_mode == "direct":
                    step = radius_scale * r
                else:
                    step = radius_scale / r

        if junction_window > 0 and junction_s.size > 0:
            while junction_ptr < len(junction_s) - 1 and junction_s[junction_ptr] < cur:
                junction_ptr += 1
            d0 = abs(cur - junction_s[junction_ptr])
            d1 = abs(cur - junction_s[junction_ptr - 1]) if junction_ptr > 0 else d0
            if min(d0, d1) <= junction_window:
                step *= junction_factor

        if not np.isfinite(step) or step <= 0:
            step = step_max
        step = float(np.clip(step, step_min, step_max))
        cur += step
        if cur >= total:
            break
        s_new.append(cur)

    if s_new[-1] < total:
        s_new.append(total)
    s_new = np.asarray(s_new, dtype=np.float64)

    x = np.interp(s_new, s, pts[:, 0])
    y = np.interp(s_new, s, pts[:, 1])
    z = np.interp(s_new, s, pts[:, 2])
    new_pts = np.column_stack([x, y, z])
    new_scal = np.interp(s_new, s, scal)
    return new_pts, new_scal


def resample_centerline_step(centerline, step=0.003, use_radius=False, step_min=0.001, step_max=0.006,
                             radius_scale=0.3, radius_mode="inverse",
                             drds_threshold=0.0, drds_boost=1.0,
                             junction_window=0.0, junction_factor=1.0, junction_degree=3,
                             max_points_per_branch=None):
    resampleada = vtk.vtkPolyData()
    cellarray = vtk.vtkCellArray()
    points = vtk.vtkPoints()

    radius_array = vtk.vtkFloatArray()
    radius_array.SetName("Radius")
    original_radius = centerline.GetPointData().GetArray("Radius")
    deg = _compute_point_degrees(centerline)

    for j in range(centerline.GetNumberOfCells()):
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints()
        pts = []
        rads = []
        pids = []

        for i in range(numberOfCellPoints):
            pid = centerline.GetCell(j).GetPointId(i)
            point = centerline.GetPoints().GetPoint(pid)
            pts.append(point)
            pids.append(pid)
            if original_radius is not None:
                rads.append(float(original_radius.GetValue(pid)))

        pts = np.asarray(pts, dtype=np.float64)
        rads_arr = None if original_radius is None else np.asarray(rads, dtype=np.float64)
        pids = np.asarray(pids, dtype=np.int64)

        if use_radius:
            junction_s = []
            if junction_window > 0 and pids.size > 0:
                seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
                s = np.concatenate([[0.0], np.cumsum(seg)])
                for idx, pid in enumerate(pids):
                    if deg[pid] >= junction_degree:
                        junction_s.append(float(s[idx]))
            new_pts, new_rads = _resample_polyline_radius_adaptive(
                pts, rads_arr, step_min, step_max, radius_scale, radius_mode,
                drds_threshold=drds_threshold, drds_boost=drds_boost,
                junction_s=junction_s, junction_window=junction_window,
                junction_factor=junction_factor
            )
        else:
            new_pts, new_rads = _resample_polyline_uniform(pts, rads_arr, step)

        new_pts, new_rads = _cap_samples(new_pts, new_rads, max_points_per_branch)

        polyline = vtk.vtkPolyLine()
        for idx in range(new_pts.shape[0]):
            newPointId = points.InsertNextPoint(new_pts[idx].tolist())
            polyline.GetPointIds().InsertNextId(newPointId)
            if new_rads is not None:
                radius_array.InsertNextValue(float(new_rads[idx]))
        cellarray.InsertNextCell(polyline)

    resampleada.SetPoints(points)
    resampleada.SetLines(cellarray)
    if radius_array.GetNumberOfTuples() > 0:
        resampleada.GetPointData().AddArray(radius_array)
    return resampleada


def resample_centerline_vmtk(centerline, step=0.003):
    try:
        from vmtk import vtkvmtk
    except Exception as exc:
        raise RuntimeError("vtkvmtk not available for centerline resampling") from exc

    f = vtkvmtk.vtkvmtkCenterlineResampling()
    f.SetInputData(centerline)
    f.SetResamplingStepLength(float(step))
    f.Update()
    return f.GetOutput()

def vtpToObj (file, folderin, folderout):
    polydata_reader = vtk.vtkXMLPolyDataReader()
    #polydata_reader.SetFileName("centerlines/"+file)
    polydata_reader.SetFileName(folderin + "/"+file)
   

    polydata_reader.Update()
        
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(polydata_reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.AddActor(actor)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Assign actor to the renderer
    ren.AddActor(actor)
    print("writing file", folderout + "/" + file.split(".")[0])
    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix(folderout + "/" + file.split(".")[0]);
    writer.SetInput(renWin);
    writer.Write()
