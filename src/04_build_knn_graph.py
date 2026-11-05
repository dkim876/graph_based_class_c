# 04_build_knn_graph.py

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def knn_graph(coordinates, k):
    if coordinates.ndim != 2 or np.iscomplexobj(coordinates) or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be a finite, real two-dimensional array")
    molecule_count, dimension_count = coordinates.shape
    if molecule_count < 2 or dimension_count < 1:
        raise ValueError("Coordinates require at least two molecules and one dimension")
    if not 1 <= k < molecule_count:
        raise ValueError(f"k must be between 1 and {molecule_count - 1}")

    tree = cKDTree(coordinates)
    queried_distances, queried_neighbors = tree.query(coordinates, k=k + 1)
    neighbors = np.empty((molecule_count, k), dtype=np.int64)
    distances = np.empty((molecule_count, k), dtype=np.float64)

    for molecule_index in range(molecule_count):
        other_molecules = queried_neighbors[molecule_index] != molecule_index
        neighbors[molecule_index] = queried_neighbors[molecule_index][other_molecules][:k]
        distances[molecule_index] = queried_distances[molecule_index][other_molecules][:k]

    if not np.isfinite(distances).all():
        raise ValueError("Neighbor search produced nonfinite distances")
    return neighbors, distances


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", "--pca-whiten", dest="pca", type=Path,
                        default=repository_root / "data/processed/layer18/q50_pca_whiten.npz")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")

    output_path = args.out or args.pca.with_name(f"{args.pca.stem}_k{args.k}_knn.npz")
    if output_path.suffix != ".npz":
        parser.error("--out must end in .npz")
    if output_path.resolve() == args.pca.resolve():
        parser.error("Output path must differ from the PCA archive")

    with np.load(args.pca, allow_pickle=False) as archive:
        if "Xw" not in archive.files:
            raise ValueError("PCA archive does not contain Xw coordinates")
        if "molecule_ids" not in archive.files:
            raise ValueError("PCA archive lacks molecule IDs; regenerate it with Step 03")
        coordinates = archive["Xw"]
        molecule_ids = archive["molecule_ids"].astype(str)
    if molecule_ids.shape != (len(coordinates),) or len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("PCA rows require aligned, unique molecule IDs")

    neighbors, distances = knn_graph(coordinates, args.k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, neighbors=neighbors, distances=distances, molecule_ids=molecule_ids)

    print(f"Nodes: {len(coordinates)}; dimensions: {coordinates.shape[1]}; k: {args.k}")
    print(f"Directed edges: {neighbors.size}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
