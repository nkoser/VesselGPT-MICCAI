import glob
import os
import pickle
import shutil
import traceback
from os.path import join

import numpy as np
import networkx as nx
import vtk
from vtk.util.numpy_support import vtk_to_numpy

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

from Preprocessing.ResamplingRDP import interpolarRDP_conRadio, vtpToObj
from Preprocessing.splines import (
    calculate_splines,
    limpiarRadiosSplines,
    traversefeaturesSerializado,
    binarizar,
    grafo2arbol,
)
from Preprocessing.parseObj import calcularMatriz, calcularMatrizSplines
import Preprocessing.Arbol as modelo


class PreprocessingPipeline:
    STEP_ORDER = [
        "copy_raw_data",
        "normalize_meshes",
        "extract_centerlines",
        "resample_centerlines",
        "centerlines_to_obj",
        "vessels_to_obj",
        "radius_arrays",
        "splines",
        "graphs",
        "trees",
        "graphs_splines",
        "trees_splines",
    ]

    REQUIRED_PATHS = [
        "raw_meshes",
        "vessels_normalized",
        "centerlines",
        "centerlines_resampled",
        "centerlines_resampled_obj",
        "vessels_obj",
        "radius_arrays",
        "splines_coef",
        "splines_knots",
        "grafos",
        "trees_numpy",
        "trees_serialized",
        "grafos_splines",
        "trees_splines",
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = cfg.get("_paths") or {}
        self.params = cfg.get("params", {})
        self.flags = cfg.get("flags", {})

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _resolve_paths(cfg, config_dir):
        paths = cfg.get("paths", {})
        resolved = {}
        for key, value in paths.items():
            if value is None:
                resolved[key] = None
                continue
            if os.path.isabs(value):
                resolved[key] = value
            else:
                resolved[key] = os.path.normpath(os.path.join(config_dir, value))
        return resolved

    @staticmethod
    def _apply_path_defaults(paths):
        raw_root = paths.get("raw_root")
        output_root = paths.get("output_root")

        if raw_root and not paths.get("raw_meshes"):
            paths["raw_meshes"] = raw_root

        if output_root:
            defaults = {
                "vessels_normalized": os.path.join(output_root, "vesselsNormalized"),
                "centerlines": os.path.join(output_root, "centerlines"),
                "centerlines_resampled": os.path.join(output_root, "centerlinesResampled"),
                "centerlines_resampled_obj": os.path.join(output_root, "centerlinesResampledOBJ"),
                "vessels_obj": os.path.join(output_root, "vesselsOBJ"),
                "radius_arrays": os.path.join(output_root, "radius_arrays"),
                "splines_coef": os.path.join(output_root, "splines", "coeficientes"),
                "splines_knots": os.path.join(output_root, "splines", "knots"),
                "grafos": os.path.join(output_root, "grafos"),
                "trees_numpy": os.path.join(output_root, "TreesNumpy"),
                "trees_serialized": os.path.join(output_root, "Trees"),
                "grafos_splines": os.path.join(output_root, "grafosSplines"),
                "trees_splines": os.path.join(output_root, "TreesSplines"),
            }
            for key, value in defaults.items():
                paths.setdefault(key, value)

        return paths

    @staticmethod
    def _ensure_trailing_sep(path):
        if path.endswith(os.sep):
            return path
        return path + os.sep

    @classmethod
    def load_config(cls, path):
        config_path = os.path.abspath(path)
        config_dir = os.path.dirname(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = cfg or {}
        cfg["_config_path"] = config_path
        cfg["_config_dir"] = config_dir
        cfg["paths"] = cls._apply_path_defaults(cfg.get("paths", {}) or {})
        cfg["_paths"] = cls._resolve_paths(cfg, config_dir)
        return cfg

    def _validate_paths(self):
        for key in self.REQUIRED_PATHS:
            if key not in self.paths:
                raise ValueError("Missing path in config: " + key)

    def normalize_meshes(self):
        self._ensure_dir(self.paths["vessels_normalized"])
        input_dir = self.paths["raw_meshes"]
        overwrite = bool(self.params.get("normalize_overwrite", False))

        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()

        meshes = glob.glob(join(input_dir, "*_models", "*", "surface", "*.vtp"))

        for mesh in meshes:
            src_path = mesh
            id = mesh.split(os.sep)[-3]
            dst_path = join(self.paths["vessels_normalized"], id + '.vtp')
            if os.path.exists(dst_path) and not overwrite:
                continue

            reader.SetFileName(src_path)
            reader.Update()
            polydata = reader.GetOutput()

            points = polydata.GetPoints()
            if points is None:
                print("No points found in:", mesh)
                continue

            n_points = points.GetNumberOfPoints()
            coords = np.array([points.GetPoint(i) for i in range(n_points)])

            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            center = (maxs + mins) / 2
            half_ranges = (maxs - mins) / 2
            uniform_scale = np.max(half_ranges)
            if uniform_scale == 0:
                uniform_scale = 1.0

            normalized_coords = (coords - center) / uniform_scale

            new_points = vtk.vtkPoints()
            for p in normalized_coords:
                new_points.InsertNextPoint(p.tolist())

            polydata.SetPoints(new_points)
            polydata.Modified()

            writer.SetFileName(dst_path)
            writer.SetInputData(polydata)
            writer.Write()
            print("Normalized:", mesh)

    def extract_centerlines(self):
        from vmtk import pypes

        self._ensure_dir(self.paths["centerlines"])
        vessels = [f for f in os.listdir(self.paths["vessels_normalized"]) if f.endswith(".vtp")]
        existing = set(os.listdir(self.paths["centerlines"]))
        vmtk_script = self.params.get("vmtk_script", "vmtknetworkextraction")

        for file in vessels:
            base = os.path.splitext(file)[0]
            out_name = base + "-network.vtp"
            if out_name in existing:
                continue
            print("Processing file:", file)
            input_file = f'-ifile "{os.path.join(self.paths["vessels_normalized"], file)}"'
            output_file = f'-ofile "{os.path.join(self.paths["centerlines"], out_name)}"'
            pypes.PypeRun(vmtk_script + " " + input_file + " " + output_file)

    def resample_centerlines(self):
        self._ensure_dir(self.paths["centerlines_resampled"])
        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()
        epsilon = self.params.get("resample_epsilon", 0.02)

        centerlines = os.listdir(self.paths["centerlines"])
        for file in centerlines:
            if not file.endswith(".vtp"):
                continue
            if file in os.listdir(self.paths["centerlines_resampled"]):
                continue
            print("Resampling:", file)
            reader.SetFileName(os.path.join(self.paths["centerlines"], file))
            reader.Update()
            centerline = reader.GetOutput()
            resampled = interpolarRDP_conRadio(centerline, epsilon)
            writer.SetFileName(os.path.join(self.paths["centerlines_resampled"], file))
            writer.SetInputData(resampled)
            writer.Write()

    def centerlines_to_obj(self):
        self._ensure_dir(self.paths["centerlines_resampled_obj"])
        centerlines = os.listdir(self.paths["centerlines_resampled"])
        existing = set(os.listdir(self.paths["centerlines_resampled_obj"]))
        for file in centerlines:
            if not file.endswith(".vtp"):
                continue
            out_name = os.path.splitext(file)[0] + ".obj"
            if out_name in existing:
                continue
            print("Converting centerline to obj:", file)
            vtpToObj(file, self.paths["centerlines_resampled"], self.paths["centerlines_resampled_obj"])

    def vessels_to_obj(self):
        self._ensure_dir(self.paths["vessels_obj"])
        vessels = os.listdir(self.paths["vessels_normalized"])
        existing = set(os.listdir(self.paths["vessels_obj"]))
        for file in vessels:
            if not file.endswith(".vtp"):
                continue
            out_name = os.path.splitext(file)[0] + ".obj"
            if out_name in existing:
                continue
            print("Converting vessel to obj:", file)
            vtpToObj(file, self.paths["vessels_normalized"], self.paths["vessels_obj"])

    def build_radius_arrays(self):
        self._ensure_dir(self.paths["radius_arrays"])
        reader = vtk.vtkXMLPolyDataReader()

        for filename in os.listdir(self.paths["centerlines_resampled"]):
            if not filename.endswith(".vtp"):
                continue
            filepath = os.path.join(self.paths["centerlines_resampled"], filename)
            reader.SetFileName(filepath)
            reader.Update()
            polydata = reader.GetOutput()

            point_data = polydata.GetPointData()
            radius_array = point_data.GetArray("Radius")
            if radius_array is None:
                print("No 'Radius' array found in", filename)
                continue

            radius_np = vtk_to_numpy(radius_array)
            output_filename = os.path.splitext(filename)[0] + "_radius.npy"
            output_path = os.path.join(self.paths["radius_arrays"], output_filename)
            if os.path.exists(output_path):
                continue
            np.save(output_path, radius_np)
            print("Saved:", output_path)

    def build_splines(self):
        self._ensure_dir(self.paths["splines_coef"])
        self._ensure_dir(self.paths["splines_knots"])

        centerfolder = self._ensure_trailing_sep(self.paths["centerlines_resampled"])
        meshfolder = self._ensure_trailing_sep(self.paths["vessels_normalized"])
        coef_folder = self._ensure_trailing_sep(self.paths["splines_coef"])

        meshes = sorted(os.listdir(meshfolder))
        for mesh in meshes:
            calculate_splines(mesh, coef_folder, centerfolder, meshfolder)

    def build_graphs(self):
        self._ensure_dir(self.paths["grafos"])
        gfolder = set(os.listdir(self.paths["grafos"]))
        mesh_obj = os.listdir(self.paths["vessels_obj"])

        for file in mesh_obj:
            if not file.endswith(".obj"):
                continue
            graph_name = os.path.splitext(file)[0] + "-grafo.gpickle"
            if graph_name in gfolder:
                continue
            try:
                center_obj = os.path.join(
                    self.paths["centerlines_resampled_obj"], os.path.splitext(file)[0] + "-network.obj"
                )
                radius_path = os.path.join(
                    self.paths["radius_arrays"], os.path.splitext(file)[0] + "-network_radius.npy"
                )
                file_obj = open(center_obj, "r")
                grafo = calcularMatriz(file_obj, radius_path)
                print("Calculating graph:", file)
                with open(os.path.join(self.paths["grafos"], graph_name), "wb") as f:
                    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
            except Exception:
                print("Problem with:", file)
                traceback.print_exc()

    def build_trees(self):
        self._ensure_dir(self.paths["trees_numpy"])
        self._ensure_dir(self.paths["trees_serialized"])

        t_list = set(os.listdir(self.paths["trees_numpy"]))
        gfolder = set(os.listdir(self.paths["grafos"]))
        files = os.listdir(self.paths["vessels_obj"])

        for file in files:
            if not file.endswith(".obj"):
                continue
            base = os.path.splitext(file)[0]
            graph_name = base + "-grafo.gpickle"
            if graph_name not in gfolder:
                continue
            if base + ".npy" in t_list:
                continue

            try:
                grafo = pickle.load(open(os.path.join(self.paths["grafos"], graph_name), "rb"))
                grafo = grafo.to_undirected()

                if len(nx.cycle_basis(grafo)) > 0:
                    print("Graph has cycles:", file)
                    continue

                for nodo in grafo.nodes:
                    if len(grafo.edges(nodo)) > 3:
                        binarizar(grafo)
                        break

                a_recorrer = []
                numero_nodo_inicial = 1
                distancias = nx.floyd_warshall(grafo)

                par_maximo = (-1, -1)
                maxima = -1
                for nodo_inicial in distancias.keys():
                    for nodo_final in distancias[nodo_inicial]:
                        if distancias[nodo_inicial][nodo_final] > maxima:
                            maxima = distancias[nodo_inicial][nodo_final]
                            par_maximo = (nodo_inicial, nodo_final)

                for nodo in grafo.nodes:
                    if distancias[par_maximo[0]][nodo] == int(maxima / 2):
                        numero_nodo_inicial = nodo
                        if len(grafo.edges(numero_nodo_inicial)) > 2:
                            numero_nodo_inicial = list(grafo.edges(numero_nodo_inicial))[0][1]
                        break

                rad = list(grafo.nodes[numero_nodo_inicial]["radio"])
                nodo_raiz = modelo.Node(numero_nodo_inicial, radius=rad)

                for vecino in grafo.neighbors(numero_nodo_inicial):
                    if vecino != numero_nodo_inicial:
                        a_recorrer.append((vecino, numero_nodo_inicial, nodo_raiz))

                while a_recorrer:
                    nodo_agregar, nodo_padre_id, nodo_padre = a_recorrer.pop(0)
                    radius = list(grafo.nodes[nodo_agregar]["radio"])
                    nodo_actual = modelo.Node(nodo_agregar, radius=radius)
                    nodo_padre.agregarHijo(nodo_actual)
                    for vecino in grafo.neighbors(nodo_agregar):
                        if vecino != nodo_padre_id:
                            a_recorrer.append((vecino, nodo_agregar, nodo_actual))

                serial = grafo2arbol(grafo)
                f = []
                traversefeaturesSerializado(nodo_raiz, f, k=4)
                array = np.array(f)
                np.save(os.path.join(self.paths["trees_numpy"], base), array)
                print("Calculated tree:", file)

                tree_path = os.path.join(self.paths["trees_serialized"], base + "_tree.dat")
                with open(tree_path, "w", encoding="utf-8") as out_f:
                    out_f.write(serial)
            except Exception:
                print("Error with:", file)
                traceback.print_exc()

    def build_graphs_splines(self):
        self._ensure_dir(self.paths["grafos_splines"])
        gfolder = set(os.listdir(self.paths["grafos_splines"]))
        files = os.listdir(self.paths["vessels_obj"])

        for file in files:
            if not file.endswith(".obj"):
                continue
            graph_name = os.path.splitext(file)[0] + "-grafo.gpickle"
            if graph_name in gfolder:
                continue
            try:
                obj_path = os.path.join(
                    self.paths["centerlines_resampled_obj"], os.path.splitext(file)[0] + "-network.obj"
                )
                file_obj = open(obj_path, "r")
                coef_path = os.path.join(self.paths["splines_coef"], os.path.splitext(file)[0] + ".pkl")
                grafo = calcularMatrizSplines(file_obj, coef_path)
                print("Calculating spline graph:", file)
                with open(os.path.join(self.paths["grafos_splines"], graph_name), "wb") as f:
                    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
            except Exception:
                print("Problem with:", file)
                traceback.print_exc()

    def build_trees_splines(self):
        self._ensure_dir(self.paths["trees_splines"])
        t_list = set(os.listdir(self.paths["trees_splines"]))
        gfolder = set(os.listdir(self.paths["grafos_splines"]))
        files = os.listdir(self.paths["vessels_obj"])

        for file in files:
            if not file.endswith(".obj"):
                continue
            base = os.path.splitext(file)[0]
            graph_name = base + "-grafo.gpickle"
            if graph_name not in gfolder:
                continue
            if base + ".npy" in t_list:
                continue

            try:
                grafo = pickle.load(open(os.path.join(self.paths["grafos_splines"], graph_name), "rb"))
                grafo = grafo.to_undirected()

                if len(nx.cycle_basis(grafo)) > 0:
                    print("Graph has cycles:", file)
                    continue

                for nodo in grafo.nodes:
                    if len(grafo.edges(nodo)) > 3:
                        binarizar(grafo)
                        break

                a_recorrer = []
                numero_nodo_inicial = 1
                distancias = nx.floyd_warshall(grafo)

                par_maximo = (-1, -1)
                maxima = -1
                for nodo_inicial in distancias.keys():
                    for nodo_final in distancias[nodo_inicial]:
                        if distancias[nodo_inicial][nodo_final] > maxima:
                            maxima = distancias[nodo_inicial][nodo_final]
                            par_maximo = (nodo_inicial, nodo_final)

                for nodo in grafo.nodes:
                    if distancias[par_maximo[0]][nodo] == int(maxima / 2):
                        numero_nodo_inicial = nodo
                        if len(grafo.edges(numero_nodo_inicial)) > 2:
                            numero_nodo_inicial = list(grafo.edges(numero_nodo_inicial))[0][1]
                        break

                rad = list(grafo.nodes[numero_nodo_inicial]["radio"])
                rad = limpiarRadiosSplines(rad)
                nodo_raiz = modelo.Node(numero_nodo_inicial, radius=rad)

                for vecino in grafo.neighbors(numero_nodo_inicial):
                    if vecino != numero_nodo_inicial:
                        a_recorrer.append((vecino, numero_nodo_inicial, nodo_raiz))

                while a_recorrer:
                    nodo_agregar, nodo_padre_id, nodo_padre = a_recorrer.pop(0)
                    radius = list(grafo.nodes[nodo_agregar]["radio"])
                    radius = limpiarRadiosSplines(radius)
                    nodo_actual = modelo.Node(nodo_agregar, radius=radius)
                    nodo_padre.agregarHijo(nodo_actual)
                    for vecino in grafo.neighbors(nodo_agregar):
                        if vecino != nodo_padre_id:
                            a_recorrer.append((vecino, nodo_agregar, nodo_actual))

                f = []
                traversefeaturesSerializado(nodo_raiz, f, k=39)
                array = np.array(f)
                np.save(os.path.join(self.paths["trees_splines"], base), array)
                print("Calculated spline tree:", file)
            except Exception:
                print("Error with:", file)
                traceback.print_exc()

    def copy_raw_data(self):
        self._ensure_dir(self.paths["vessels_normalized"])
        input_dir = self.paths["raw_meshes"]
        overwrite = bool(self.params.get("normalize_overwrite", False))

        meshes = glob.glob(join(input_dir, "*_models", "*", "surface", "*.vtp"))
        for mesh in meshes:
            id = mesh.split(os.sep)[-3]
            dst_path = join(self.paths["vessels_normalized"], id + ".vtp")
            if os.path.exists(dst_path) and not overwrite:
                continue
            shutil.copy2(mesh, dst_path)
        print("normalize_meshes copy-only: copied raw meshes to vessels_normalized")
        return

    def run(self, only_steps=None):
        self._validate_paths()

        step_funcs = {
            "copy_raw_data": self.copy_raw_data,  # Placeholder if needed
            "normalize_meshes": self.normalize_meshes,
            "extract_centerlines": self.extract_centerlines,
            "resample_centerlines": self.resample_centerlines,
            "centerlines_to_obj": self.centerlines_to_obj,
            "vessels_to_obj": self.vessels_to_obj,
            "radius_arrays": self.build_radius_arrays,
            "splines": self.build_splines,
            "graphs": self.build_graphs,
            "trees": self.build_trees,
            "graphs_splines": self.build_graphs_splines,
            "trees_splines": self.build_trees_splines,
        }

        if only_steps:
            steps_to_run = [s for s in self.STEP_ORDER if s in set(only_steps)]
        else:
            steps_to_run = [s for s in self.STEP_ORDER if self.flags.get(s, False)]

        for step in steps_to_run:
            print("=== Running step:", step)
            step_funcs[step]()


def load_config(path):
    return PreprocessingPipeline.load_config(path)


def run_pipeline(cfg, only_steps=None):
    pipeline = PreprocessingPipeline(cfg)
    pipeline.run(only_steps=only_steps)
