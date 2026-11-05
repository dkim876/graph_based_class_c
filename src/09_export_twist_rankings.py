# 09_export_twist_rankings.py

import argparse
from collections import defaultdict
from importlib import import_module
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from holonomy import cycle_rotation_angles, fit_binding_frames


def rotation_key(pair):
    angles = pair["lobe_twist_degrees"]
    return -max(angles), -min(angles), pair["figure_eight_id"]


def rank_hubs(pairs, count):
    hubs = {}
    for pair in pairs:
        angles = pair["lobe_twist_degrees"]
        if len(angles) != 2 or any(not math.isfinite(angle) or not 0 <= angle <= 180 for angle in angles):
            raise ValueError("Expected two finite holonomy angles between zero and 180 degrees")
        hub = pair["hub"]
        if set(pair["lobe1"]["nodes"]) & set(pair["lobe2"]["nodes"]) != {hub}:
            raise ValueError("Figure-eight lobes must share exactly their hub")
        if hub not in hubs or rotation_key(pair) < rotation_key(hubs[hub]):
            hubs[hub] = pair
    return sorted(hubs.values(), key=rotation_key)[:count]


def rank_cycles(cycles, angles, count):
    eligible = np.flatnonzero(np.isfinite(angles))
    at_hub = defaultdict(list)
    for index in eligible:
        index = int(index)
        for node in cycles[index]["nodes"][:-1]:
            at_hub[node].append(index)
    identifiers = np.array([cycles[index]["cycle_id"] for index in eligible])
    ordered = eligible[np.lexsort((identifiers, -angles[eligible]))]
    selected, seen = [], set()
    for index in ordered:
        cycle = cycles[index]
        vertices = frozenset(cycle["nodes"][:-1])
        if vertices in seen:
            continue
        seen.add(vertices)
        partner = next(((hub, other) for hub in cycle["nodes"][:-1] for other in at_hub[hub]
                        if not any(node != hub and node in vertices for node in cycles[other]["nodes"][:-1])), None)
        if partner is not None:
            hub, other = partner
            selected.append((cycle, float(angles[index]), hub, cycles[other]["cycle_id"]))
            if len(selected) == count:
                break
    return selected


def export_rows(hubs, cycles, labels):
    hub_rows, cycle_rows = [], []
    for rank, pair in enumerate(hubs, 1):
        index = pair["hub"]
        angles = pair["lobe_twist_degrees"]
        first = max(range(2), key=lambda position: (angles[position], -pair[f"lobe{position + 1}"]["cycle_id"]))
        hub_rows.append({
            "rank": rank, "hub": index, **labels.iloc[index].to_dict(),
            "twist_degrees": angles[first], "other_lobe_twist_degrees": angles[1 - first],
            "cycle_id": pair[f"lobe{first + 1}"]["cycle_id"],
            "other_cycle_id": pair[f"lobe{2 - first}"]["cycle_id"],
            "figure_eight_id": pair["figure_eight_id"],
        })
    for rank, (cycle, angle, hub, partner_id) in enumerate(cycles, 1):
        nodes = cycle["nodes"]
        row = {"rank": rank, "cycle_id": cycle["cycle_id"], "twist_degrees": angle,
               "length": cycle["length"], "color_pattern": cycle["color_pattern"],
               "hub": hub, "hub_molecule_chembl_id": labels.iloc[hub]["molecule_chembl_id"],
               "partner_cycle_id": partner_id, "nodes": nodes, "pki_changes": cycle["pki_changes"]}
        row.update({column: labels.iloc[nodes][column].tolist() for column in labels.columns})
        cycle_rows.append({key: json.dumps(value, separators=(",", ":"), allow_nan=False)
                           if isinstance(value, list) else value for key, value in row.items()})
    return hub_rows, cycle_rows


def main():
    root = Path(__file__).resolve().parents[1]
    pairing = import_module("07_find_figure_eights")
    cycles_path = root / pairing.CYCLES_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-eights", type=Path,
                        default=cycles_path.with_name(cycles_path.stem + "_figure_eights_top100_twist.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    output = args.out_dir or args.figure_eights.parent
    paths = [output / f"top{args.count}_twist_{name}.csv" for name in ("hubs", "cycles")]
    if not args.overwrite and any(path.exists() for path in paths):
        parser.error("Output files already exist; use --overwrite")
    with args.figure_eights.open() as stream:
        payload = json.load(stream)
    for name in ("cycles", "colored_graph", "pca"):
        if pairing.file_hash(Path(payload[f"{name}_file"])) != payload[f"{name}_sha256"]:
            raise ValueError(f"Figure-eights do not match the {name} input")
    hubs = rank_hubs(payload.pop("figure_eights"), args.count)
    if not hubs:
        parser.error("No figure-eights are available")
    print("Loading cycles", flush=True)
    with Path(payload["cycles_file"]).open() as stream:
        source = json.load(stream)
    cycles, identifiers = pairing.validate_cycles(source)
    if identifiers != payload["molecule_ids"]:
        raise ValueError("Figure-eights and colored graph have different molecule orders")
    with np.load(payload["pca_file"], allow_pickle=False) as archive:
        if not np.array_equal(archive["molecule_ids"], identifiers):
            raise ValueError("PCA and graph have different molecule orders")
        coordinates = archive["Xw"]
    with np.load(payload["colored_graph_file"], allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        neighbors = archive["neighbors"]
    columns = ["molecule_chembl_id", "smiles", "pKi_D2", "pKi_D3"]
    labels = pd.read_csv(metadata["labels_file"], dtype=str, keep_default_na=False)[columns]
    if labels["molecule_chembl_id"].tolist() != identifiers:
        raise ValueError("Binding labels and graph have different molecule orders")
    labels[["pKi_D2", "pKi_D3"]] = labels[["pKi_D2", "pKi_D3"]].astype(float)
    settings = payload["parameters"]
    print(f"Fitting holonomy frames: n-loc={settings['n_loc']}, ridge={settings['ridge']:g}", flush=True)
    frames, valid = fit_binding_frames(coordinates, labels[["pKi_D2", "pKi_D3"]].to_numpy(),
                                      settings["n_loc"], settings["ridge"], settings["min_rank_ratio"])
    angles = cycle_rotation_angles(cycles, frames, valid, neighbors, settings["min_overlap"])
    print("Ranking cycles", flush=True)
    cycles = rank_cycles(cycles, angles, args.count)
    output.mkdir(parents=True, exist_ok=True)
    for path, rows in zip(paths, export_rows(hubs, cycles, labels)):
        pd.DataFrame(rows).to_csv(path, index=False, mode="w" if args.overwrite else "x")
        print(f"Saved {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
