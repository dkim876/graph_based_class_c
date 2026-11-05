# 05_color_knn_edges.py

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_binding_values(labels_path, smiles_path):
    molecules = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    required_columns = {"molecule_chembl_id", "smiles", "pKi_D2", "pKi_D3"}
    if molecules.empty or not required_columns.issubset(molecules.columns):
        raise ValueError("Labels require molecules with IDs, SMILES, pKi_D2, and pKi_D3")
    identifiers = molecules["molecule_chembl_id"]
    if identifiers.str.strip().eq("").any() or not identifiers.is_unique:
        raise ValueError("Molecule IDs must be nonempty and unique")
    smiles = molecules["smiles"].tolist()
    if any(not value or value != value.strip() or "\n" in value or "\r" in value for value in smiles):
        raise ValueError("SMILES must be nonempty single-line strings without surrounding whitespace")
    if ("\n".join(smiles) + "\n").encode("utf-8") != smiles_path.read_bytes():
        raise ValueError("Labels and smiles.txt are not aligned")
    binding_values = molecules[["pKi_D2", "pKi_D3"]].to_numpy(dtype=float)
    return binding_values, identifiers.to_numpy(dtype=str)


def color_edges(binding_values, neighbors, min_increase_percent):
    if binding_values.ndim != 2 or binding_values.shape[1] != 2:
        raise ValueError("Binding values must have D2 and D3 columns")
    if np.iscomplexobj(binding_values) or not np.isfinite(binding_values).all():
        raise ValueError("Binding values must be finite and real")
    molecule_count = len(binding_values)
    if neighbors.ndim != 2 or neighbors.shape[0] != molecule_count:
        raise ValueError("Graph rows must match the binding dataset")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError("Neighbor indices must be integers")
    if not 1 <= neighbors.shape[1] < molecule_count:
        raise ValueError("Neighbor count must be positive and smaller than the molecule count")
    if np.any(neighbors < 0) or np.any(neighbors >= molecule_count):
        raise ValueError("Neighbor indices are out of range")
    if np.any(neighbors == np.arange(molecule_count)[:, None]):
        raise ValueError("Graph contains self-loops")
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise ValueError("Graph contains duplicate neighbors")
    if not np.isfinite(min_increase_percent) or min_increase_percent < 0:
        raise ValueError("Minimum increase percent must be finite and nonnegative")

    source_values = binding_values[:, None, :]
    changes = binding_values[neighbors] - source_values
    thresholds = min_increase_percent / 100 * np.abs(source_values)
    if not np.isfinite(changes).all() or not np.isfinite(thresholds).all():
        raise ValueError("Binding changes or thresholds are nonfinite")
    return changes, changes > thresholds


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn", type=Path,
                        default=repository_root / "data/processed/layer18/q50_pca_whiten_k50_knn.npz")
    parser.add_argument("--labels", type=Path, default=repository_root / "data/processed/d2_d3_m50.csv")
    parser.add_argument("--smiles", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-increase-percent", type=float, default=5.0)
    args = parser.parse_args()

    smiles_path = args.smiles or args.labels.with_name("smiles.txt")
    percentage_label = np.format_float_positional(args.min_increase_percent, trim="-")
    output_path = args.out or args.knn.with_name(
        f"{args.knn.stem.removesuffix('_knn')}_{args.labels.stem}_increase{percentage_label}pct_colored.npz"
    )
    if output_path.suffix != ".npz":
        parser.error("--out must end in .npz")
    if output_path.resolve() in {path.resolve() for path in (args.knn, args.labels, smiles_path)}:
        parser.error("Output path must differ from every input file")

    binding_values, molecule_ids = load_binding_values(args.labels, smiles_path)
    with np.load(args.knn, allow_pickle=False) as archive:
        if "molecule_ids" not in archive.files:
            raise ValueError("kNN archive lacks molecule IDs; regenerate it with Step 04")
        if not np.array_equal(archive["molecule_ids"], molecule_ids):
            raise ValueError("kNN graph and binding labels have different molecule orders")
        neighbors, distances = archive["neighbors"], archive["distances"]
    changes, retained = color_edges(binding_values, neighbors, args.min_increase_percent)
    if distances.shape != neighbors.shape or np.iscomplexobj(distances):
        raise ValueError("Distances must be real and have the same shape as neighbors")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Distances must be finite and nonnegative")
    if np.any(np.diff(distances, axis=1) < 0):
        raise ValueError("Distances must be sorted within each row")
    red_keep, blue_keep = retained[:, :, 0], retained[:, :, 1]

    metadata = {
        "min_increase_percent": args.min_increase_percent,
        "color_targets": {"red": "pKi_D2", "blue": "pKi_D3"},
    }
    for name, path in (("knn", args.knn), ("labels", args.labels), ("smiles", smiles_path)):
        metadata[f"{name}_file"] = str(path.resolve())
        metadata[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path, neighbors=neighbors, distances=distances,
        d2_changes=changes[:, :, 0], d3_changes=changes[:, :, 1],
        red_keep=red_keep, blue_keep=blue_keep, molecule_ids=molecule_ids,
        metadata_json=np.array(json.dumps(metadata, allow_nan=False)),
    )
    print(f"Nodes: {len(binding_values)}; k: {neighbors.shape[1]}; minimum increase: {args.min_increase_percent:g}%")
    print(f"Red edges: {red_keep.sum()}; blue edges: {blue_keep.sum()}; both: {(red_keep & blue_keep).sum()}")
    print(f"Retained directed edges: {(red_keep | blue_keep).sum()}/{neighbors.size}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
