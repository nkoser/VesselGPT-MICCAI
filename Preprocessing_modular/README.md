# Modular preprocessing

This folder adds a configurable, modular version of the preprocessing pipeline
from `Preprocessing/sections.ipynb`. The original notebook is unchanged.

## Usage

1) Edit `config.yaml` paths to match your data layout.
2) Run:

```
python run_preprocessing.py --config config.yaml
```

Optional: run only selected steps:

```
python run_preprocessing.py --config config.yaml --only extract_centerlines,resample_centerlines
```

## Path model

You can point the pipeline to your raw meshes once and let it build the rest
under a single output root:

- `paths.raw_root` points to the folder containing raw `.vtp` meshes.
- `paths.output_root` is where all outputs (centerlines, splines, graphs, trees, etc.) are created.

If you need a different layout, uncomment any override under `paths` in `config.yaml`.

## Requirements

- PyYAML (`pip install pyyaml`)
- vtk, vmtk, numpy, scipy, networkx

## Steps

- normalize_meshes
- extract_centerlines
- resample_centerlines
- centerlines_to_obj
- vessels_to_obj
- radius_arrays
- splines
- graphs
- trees
- graphs_splines
- trees_splines
