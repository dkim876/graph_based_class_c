# 01_prepare_dataset.py

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_target(raw_directory, target_name, target_id):
    measurements = pd.read_csv(
        raw_directory / f"{target_name}_Ki.csv",
        sep=";",
        dtype=str,
        keep_default_na=False,
    )
    ki_nm = pd.to_numeric(measurements["Standard Value"], errors="coerce")
    valid_measurements = (
        (measurements["Target ChEMBL ID"] == target_id)
        & (measurements["Target Organism"] == "Homo sapiens")
        & (measurements["Standard Type"] == "Ki")
        & (measurements["Standard Units"].str.lower() == "nm")
        & (measurements["Standard Relation"].str.strip().str.strip("'\"") == "=")
        & np.isfinite(ki_nm)
        & (ki_nm > 0)
        & (measurements["Molecule ChEMBL ID"] != "")
        & (~measurements["Smiles"].str.strip().str.lower().isin(["", "none", "nan"]))
        & (measurements["Data Validity Comment"].str.strip() == "")
        & (measurements["Potential Duplicate"].str.strip() == "0")
    )
    valid_data = measurements.loc[
        valid_measurements, ["Molecule ChEMBL ID", "Smiles"]
    ].copy()
    valid_data["pKi"] = 9.0 - np.log10(ki_nm.loc[valid_measurements])
    molecules = valid_data.groupby("Molecule ChEMBL ID", as_index=False, sort=True).agg(
        smiles=("Smiles", "first"),
        pKi=("pKi", "median"),
        n=("pKi", "size"),
    )
    print(f"{target_name}: {len(measurements)} raw rows -> {len(molecules)} valid molecules")
    return molecules


def normalize_pki(pki_values, min_value, max_value):
    if not np.isfinite([min_value, max_value]).all() or not min_value < 0 < max_value:
        raise ValueError("Normalization bounds must be finite and satisfy min_value < 0 < max_value")
    pki_values = np.asarray(pki_values, dtype=float)
    if pki_values.ndim != 1 or pki_values.size == 0 or not np.isfinite(pki_values).all():
        raise ValueError("Expected a nonempty one-dimensional array of finite pKi values")
    mean_pki = float(pki_values.mean())
    centered_pki = pki_values - mean_pki
    centered_min = float(centered_pki.min())
    centered_max = float(centered_pki.max())
    if not centered_min < 0 < centered_max:
        raise ValueError("Cannot scale constant pKi values to a nonzero boundary")
    scale = min(min_value / centered_min, max_value / centered_max)
    normalized_pki = scale * centered_pki
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(normalized_pki).all():
        raise ValueError("Normalization bounds produce an unrepresentable scale")
    return normalized_pki, {"mean_pKi": mean_pki, "scale": scale}


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=repository_root / "data/raw")
    parser.add_argument("--out-dir", type=Path, default=repository_root / "data/processed")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-value", type=float, default=-50.0)
    parser.add_argument("--max-value", type=float, default=50.0)
    args = parser.parse_args()
    if args.normalize and (
        not np.isfinite([args.min_value, args.max_value]).all()
        or not args.min_value < 0 < args.max_value
    ):
        parser.error("--min-value and --max-value must be finite and satisfy min_value < 0 < max_value")

    dataset_stem = "d2_d3"
    if args.normalize:
        minimum_label = str(args.min_value).removesuffix(".0")
        maximum_label = str(args.max_value).removesuffix(".0")
        if args.min_value == -args.max_value:
            dataset_stem = f"d2_d3_m{maximum_label}"
        else:
            dataset_stem = f"d2_d3_min{minimum_label}_max{maximum_label}"

    d2_molecules = load_target(args.raw_dir, "D2", "CHEMBL217")
    d3_molecules = load_target(args.raw_dir, "D3", "CHEMBL234")
    matched = d2_molecules.merge(
        d3_molecules,
        on="Molecule ChEMBL ID",
        how="inner",
        suffixes=("_D2", "_D3"),
        validate="one_to_one",
    )
    if matched.empty:
        raise ValueError("No molecules have valid Ki measurements for both D2 and D3")
    smiles_disagreements = int((matched["smiles_D2"] != matched["smiles_D3"]).sum())
    if smiles_disagreements:
        raise ValueError(f"Found {smiles_disagreements} cross-target SMILES disagreements")
    matched = matched.rename(columns={
        "Molecule ChEMBL ID": "molecule_chembl_id",
        "smiles_D2": "smiles",
    })
    matched = matched[
        ["molecule_chembl_id", "smiles", "pKi_D2", "pKi_D3", "n_D2", "n_D3"]
    ].sort_values("molecule_chembl_id").reset_index(drop=True)

    normalization = {
        "normalize": args.normalize,
        "min_value": args.min_value if args.normalize else None,
        "max_value": args.max_value if args.normalize else None,
        "matched_molecules": len(matched),
        "targets": {},
    }
    for target_name in ("D2", "D3"):
        column = f"pKi_{target_name}"
        parameters = {}
        if args.normalize:
            matched[column], parameters = normalize_pki(
                matched[column], args.min_value, args.max_value
            )
        normalization["targets"][target_name] = {"column": column, **parameters}
        values = matched[column].to_numpy()
        print(
            f"{target_name}: normalize={args.normalize}, "
            f"mean={values.mean():.6g}, min={values.min():.6f}, max={values.max():.6f}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.out_dir / f"{dataset_stem}.csv"
    normalization_path = args.out_dir / f"{dataset_stem}_normalization.json"
    matched.to_csv(dataset_path, index=False)
    (args.out_dir / "smiles.txt").write_text("\n".join(matched["smiles"]) + "\n")
    normalization_path.write_text(
        json.dumps(normalization, indent=2, allow_nan=False) + "\n"
    )
    print(f"Matched molecules: {len(matched)}")
    print(f"Saved dataset: {dataset_path}")
    print(f"Saved normalization metadata: {normalization_path}")


if __name__ == "__main__":
    main()
