# 03_pca_whiten.py

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


WHITENING_REGULARIZATION = 1e-3


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path,
                        default=repository_root / "data/processed/activations.npz")
    parser.add_argument("--out", type=Path, default=repository_root / "data/processed")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--q", type=int, default=50)
    parser.add_argument("--whiten", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.layer < 0 or args.q < 1:
        parser.error("--layer must be nonnegative and --q must be positive")

    layer_name = f"layer_{args.layer}"
    with np.load(args.activations, allow_pickle=False) as archive:
        if layer_name not in archive.files:
            raise ValueError(f"Activation archive does not contain {layer_name}")
        if "molecule_ids" not in archive.files:
            raise ValueError("Activation archive lacks molecule IDs; regenerate it with Step 02")
        molecule_ids = archive["molecule_ids"].astype(str)
        hidden_states = archive[layer_name].astype(np.float32)
    if hidden_states.ndim != 2 or not np.isfinite(hidden_states).all():
        raise ValueError("Activations must be a finite two-dimensional array")
    if molecule_ids.shape != (len(hidden_states),) or len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("Activation rows require aligned, unique molecule IDs")
    if len(hidden_states) < 2 or args.q > min(hidden_states.shape):
        raise ValueError("Invalid PCA dimension or insufficient molecules")
    if not np.any(np.ptp(hidden_states, axis=0) > 0):
        raise ValueError("PCA requires nonzero variation between molecules")

    pca = PCA(n_components=args.q, svd_solver="randomized", random_state=42)
    pca_coordinates = pca.fit_transform(hidden_states)
    whitening_transform = np.eye(args.q)
    coordinates = pca_coordinates
    if args.whiten:
        covariance = np.atleast_2d(np.cov(pca_coordinates, rowvar=False))
        covariance += WHITENING_REGULARIZATION * np.trace(covariance) / args.q * np.eye(args.q)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        whitening_transform = (eigenvectors / np.sqrt(np.maximum(eigenvalues, 1e-12))) @ eigenvectors.T
        coordinates = pca_coordinates @ whitening_transform
    if not np.isfinite(coordinates).all():
        raise ValueError("PCA or whitening produced nonfinite coordinates")

    output_directory = args.out / f"layer{args.layer}"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_name = f"q{args.q}_pca_whiten.npz" if args.whiten else f"q{args.q}_pca.npz"
    output_path = output_directory / output_name
    if output_path.resolve() == args.activations.resolve():
        raise ValueError("Output path must differ from the activation archive")
    np.savez(
        output_path,
        Xp=pca_coordinates,
        Xw=coordinates,
        molecule_ids=molecule_ids,
        whitened=args.whiten,
        layer=args.layer,
        q=args.q,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        whitening_transform=whitening_transform,
        whitening_regularization=WHITENING_REGULARIZATION if args.whiten else 0.0,
        random_state=42,
    )
    print(f"{layer_name}: {hidden_states.shape} -> {coordinates.shape}; whiten={args.whiten}")
    print(f"Explained variance retained: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
