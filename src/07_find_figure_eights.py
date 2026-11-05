# 07_find_figure_eights.py

import argparse
from collections import defaultdict
import hashlib
from importlib import import_module
from heapq import heapify, heappop, heappush
import json
import math
from pathlib import Path

import numpy as np

from holonomy import cycle_rotation_angles, fit_binding_frames


CYCLES_FILE = "data/processed/layer18/q50_pca_whiten_k50_d2_d3_m50_increase5pct_colored_cycles_3-4.json"


def validate_cycles(payload, edge_data=None, molecule_ids=None):
    if edge_data is None or molecule_ids is None:
        graph_path = Path(payload["colored_graph_file"])
        if hashlib.sha256(graph_path.read_bytes()).hexdigest() != payload["colored_graph_sha256"]:
            raise ValueError("Cycles do not match the colored graph")
        graph_module = import_module("06_find_colored_cycles")
        *arrays, graph_ids, _ = graph_module.load_graph(graph_path)
        _, _, edge_data = graph_module.build_adjacency(*arrays)
        molecule_ids = graph_ids.tolist()
    if not isinstance(molecule_ids, list) or not molecule_ids:
        raise ValueError("Molecule IDs must be a nonempty list")
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in molecule_ids):
        raise ValueError("Molecule IDs must be nonempty strings")
    node_count = len(molecule_ids)
    if len(set(molecule_ids)) != node_count:
        raise ValueError("Molecule IDs must be unique")
    cycles = payload["cycles"]
    if not isinstance(cycles, list):
        raise ValueError("Cycles must be a list")

    cycle_ids, cycle_keys = set(), set()
    for cycle in cycles:
        cycle_id = cycle["cycle_id"]
        nodes, colors = cycle["nodes"], cycle["colors"]
        if type(cycle_id) is not int or cycle_id < 1 or cycle_id in cycle_ids:
            raise ValueError("Cycle IDs must be unique positive integers")
        cycle_ids.add(cycle_id)
        if not isinstance(nodes, list) or len(nodes) < 4 or nodes[0] != nodes[-1]:
            raise ValueError(f"Cycle {cycle_id} must close and have at least three edges")
        if any(type(node) is not int or not 0 <= node < node_count for node in nodes):
            raise ValueError(f"Cycle {cycle_id} has invalid node indices")
        vertices = nodes[:-1]
        length = len(vertices)
        if len(set(vertices)) != length:
            raise ValueError(f"Cycle {cycle_id} repeats a molecule before closing")
        if type(cycle["length"]) is not int or cycle["length"] != length:
            raise ValueError(f"Cycle {cycle_id} has an inconsistent length")
        if not isinstance(colors, list) or len(colors) != length:
            raise ValueError(f"Cycle {cycle_id} has an inconsistent color count")
        if any(type(color) is not int or color not in (0, 1) for color in colors) or set(colors) != {0, 1}:
            raise ValueError(f"Cycle {cycle_id} must contain red and blue edges encoded as 1 and 0")
        first = vertices.index(min(vertices))
        cycle_key = tuple(zip(vertices[first:] + vertices[:first], colors[first:] + colors[:first]))
        if cycle_key in cycle_keys:
            raise ValueError("Input contains duplicate directed colored cycles under rotation")
        cycle_keys.add(cycle_key)
        linear_flips = sum(first != second for first, second in zip(colors[:-1], colors[1:]))
        cyclic_flips = sum(color != colors[(index + 1) % length] for index, color in enumerate(colors))
        if linear_flips != 1 or cyclic_flips != 2 or colors[0] != 1 or colors[-1] != 0:
            raise ValueError(f"Cycle {cycle_id} must have one red segment followed by one blue segment")
        if cycle["linear_flips"] != linear_flips or cycle["cyclic_color_transitions"] != cyclic_flips:
            raise ValueError(f"Cycle {cycle_id} has inconsistent color-transition counts")
        if cycle["color_pattern"] != "".join("R" if color else "B" for color in colors):
            raise ValueError(f"Cycle {cycle_id} has an inconsistent color pattern")
        if cycle["red_length"] != sum(colors) or cycle["blue_length"] != length - sum(colors):
            raise ValueError(f"Cycle {cycle_id} has inconsistent red/blue lengths")
        values = cycle["pki_changes"]
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"Cycle {cycle_id} has an inconsistent pKi-change count")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"Cycle {cycle_id} must have positive finite target-specific pKi changes")
        for edge, value in zip(zip(nodes[:-1], nodes[1:], colors), values):
            if edge_data.get(edge) != value:
                raise ValueError(f"Cycle edge or pKi change differs from the colored graph: {edge}")
    return cycles, molecule_ids


def cycle_boundary_strength(cycle):
    nodes, colors = cycle["nodes"], cycle["colors"]
    transitions = []
    for index, color in enumerate(colors):
        previous = (index - 1) % len(colors)
        if colors[previous] == color:
            continue
        transitions.append({
            "node": nodes[index], "incoming_edge_index": previous, "outgoing_edge_index": index,
            "incoming_edge": nodes[previous:previous + 2], "outgoing_edge": nodes[index:index + 2],
            "incoming_color": colors[previous], "outgoing_color": color,
            "pki_change": min(cycle["pki_changes"][previous], cycle["pki_changes"][index]),
        })
    if not transitions:
        raise ValueError("Boundary strength requires a cycle with both colors")
    return {
        "pki_change": min(transition["pki_change"] for transition in transitions),
        "transitions": transitions,
    }


def figure_eight_indices(cycles, max_per_hub=None, twist_angles=None, progress=None):
    if max_per_hub is not None and (type(max_per_hub) is not int or max_per_hub < 1):
        raise ValueError("Maximum per hub must be a positive integer")
    if twist_angles is not None:
        twist_angles = np.asarray(twist_angles, dtype=float)
        if twist_angles.shape != (len(cycles),) or np.isinf(twist_angles).any():
            raise ValueError("Twist angles must align with cycles; use NaN for unevaluable rotations")
        if np.any((twist_angles < 0) | (twist_angles > 180)):
            raise ValueError("Rotation angles must be between zero and 180 degrees")
    at_hub = defaultdict(list)
    for index, cycle in enumerate(cycles):
        if twist_angles is None or np.isfinite(twist_angles[index]):
            for node in cycle["nodes"][:-1]:
                at_hub[node].append(index)
    for completed, hub in enumerate(sorted(at_hub), 1):
        indices = sorted(at_hub[hub], key=lambda index: (
            cycles[index]["length"] if twist_angles is None else -twist_angles[index], cycles[index]["cycle_id"]))
        common = set(cycles[indices[0]]["nodes"])
        for index in indices[1:]:
            common.intersection_update(cycles[index]["nodes"])
            if len(common) == 1:
                break
        if len(common) == 1:
            def candidate(first, second):
                left, right = indices[first], indices[second]
                identifiers = sorted((cycles[left]["cycle_id"], cycles[right]["cycle_id"]))
                if twist_angles is None:
                    priority = (cycles[left]["length"] + cycles[right]["length"],
                                max(cycles[left]["length"], cycles[right]["length"]))
                else:
                    priority = (-twist_angles[left], -twist_angles[right])
                return (*priority, *identifiers, first, second)

            heap = [candidate(first, first + 1) for first in range(len(indices) - 1)]
            heapify(heap)
            saved = 0
            while heap:
                *_, first, second = heappop(heap)
                left, right = indices[first], indices[second]
                if not any(node != hub and node in cycles[right]["nodes"] for node in cycles[left]["nodes"][:-1]):
                    yield hub, left, right
                    saved += 1
                    if max_per_hub is not None and saved == max_per_hub:
                        break
                if second + 1 < len(indices):
                    heappush(heap, candidate(first, second + 1))
        if progress is not None:
            progress(completed, len(at_hub))


def pair_record(cycles, molecule_ids, hub, first, second, strengths):
    lobe1, lobe2 = sorted((cycles[first], cycles[second]), key=lambda cycle: cycle["cycle_id"])
    for lobe in (lobe1, lobe2):
        if lobe["cycle_id"] not in strengths:
            strengths[lobe["cycle_id"]] = cycle_boundary_strength(lobe)
    strength1, strength2 = strengths[lobe1["cycle_id"]], strengths[lobe2["cycle_id"]]
    return {
        "hub": hub, "hub_molecule_id": molecule_ids[hub],
        "cycle_ids": [lobe1["cycle_id"], lobe2["cycle_id"]],
        "total_length": lobe1["length"] + lobe2["length"],
        "distinct_molecule_count": lobe1["length"] + lobe2["length"] - 1,
        "lobe1_hub_position": lobe1["nodes"].index(hub),
        "lobe2_hub_position": lobe2["nodes"].index(hub),
        "lobe1": lobe1, "lobe2": lobe2,
        "boundary_strength": {
            "pki_change": min(strength1["pki_change"], strength2["pki_change"]),
            "transition_count": len(strength1["transitions"]) + len(strength2["transitions"]),
            "lobe1": strength1, "lobe2": strength2,
        },
    }


def rank_boundary_strengths(matches):
    ordered = sorted(matches, key=lambda match: (-match["boundary_strength"]["pki_change"], match["figure_eight_id"]))
    for rank, match in enumerate(ordered, 1):
        match["boundary_strength"]["rank"] = rank


def find_figure_eights(cycles, molecule_ids, max_per_hub=100, twist_angles=None, progress=None):
    strengths, matches = {}, []
    for hub, first, second in figure_eight_indices(cycles, max_per_hub, twist_angles, progress):
        match = pair_record(cycles, molecule_ids, hub, first, second, strengths)
        if twist_angles is not None:
            ordered = sorted((first, second), key=lambda index: cycles[index]["cycle_id"])
            match["lobe_twist_degrees"] = [float(twist_angles[index]) for index in ordered]
        matches.append(match)
    matches.sort(key=lambda match: (
        *((-max(match["lobe_twist_degrees"]), -min(match["lobe_twist_degrees"])) if twist_angles is not None else ()),
        match["total_length"], max(match["lobe1"]["length"], match["lobe2"]["length"]),
        match["hub"], tuple(match["cycle_ids"]),
    ))
    for index, match in enumerate(matches, 1):
        match["figure_eight_id"] = index
    rank_boundary_strengths(matches)
    return matches


def validate_pair_records(records, cycles, molecule_ids):
    by_id = {cycle["cycle_id"]: index for index, cycle in enumerate(cycles)}
    expected, strengths, seen = [], {}, set()
    for identifier, record in enumerate(records, 1):
        first_id, second_id = record["cycle_ids"]
        if first_id >= second_id or (first_id, second_id) in seen:
            raise ValueError("Figure-eight cycle pairs must be ordered and unique")
        seen.add((first_id, second_id))
        first, second = by_id[first_id], by_id[second_id]
        shared = set(cycles[first]["nodes"]) & set(cycles[second]["nodes"])
        if shared != {record["hub"]}:
            raise ValueError("Figure-eight lobes must share exactly their hub")
        match = pair_record(cycles, molecule_ids, record["hub"], first, second, strengths)
        match["figure_eight_id"] = identifier
        if "lobe_twist_degrees" in record:
            angles = record["lobe_twist_degrees"]
            if len(angles) != 2 or any(not math.isfinite(angle) or not 0 <= angle <= 180 for angle in angles):
                raise ValueError("Figure-eight rotations must be finite angles between zero and 180")
            match["lobe_twist_degrees"] = angles
        expected.append(match)
    rank_boundary_strengths(expected)
    if records != expected:
        raise ValueError("Figure-eight pairs or boundary strengths differ from the supplied cycles")
    return expected


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=Path, default=repository_root / CYCLES_FILE)
    parser.add_argument("--pca", type=Path, default=repository_root / "data/processed/layer18/q50_pca_whiten.npz")
    parser.add_argument("--max-per-hub", type=int, default=100)
    parser.add_argument("--n-loc", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--min-overlap", type=float, default=0.1)
    parser.add_argument("--min-rank-ratio", type=float, default=0.01)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_per_hub < 1:
        parser.error("--max-per-hub must be positive")
    if args.n_loc < 3 or not np.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--n-loc must be at least three and --ridge must be finite and positive")
    if not 0 < args.min_overlap < 1 or not 0 < args.min_rank_ratio < 1:
        parser.error("Overlap and rank thresholds must be between zero and one")
    output_path = args.out or args.cycles.with_name(f"{args.cycles.stem}_figure_eights_top{args.max_per_hub}_twist.json")
    if output_path.suffix != ".json" or output_path.resolve() in {args.cycles.resolve(), args.pca.resolve()}:
        parser.error("Output must be a JSON file distinct from the inputs")
    partial_path = output_path.with_suffix(".json.partial")
    if not args.overwrite and (output_path.exists() or partial_path.exists()):
        parser.error(f"Output or partial output already exists: {output_path}; use --out or --overwrite")
    print("Loading and validating cycles", flush=True)
    input_hash = file_hash(args.cycles)
    with args.cycles.open() as stream:
        source = json.load(stream)
    graph_path = Path(source["colored_graph_file"])
    cycles, molecule_ids = validate_cycles(source)
    pca_hash = file_hash(args.pca)
    with np.load(args.pca, allow_pickle=False) as archive:
        if not np.array_equal(archive["molecule_ids"], molecule_ids):
            raise ValueError("PCA coordinates and cycle molecules have different orders")
        coordinates = archive["Xw"]
    with np.load(graph_path, allow_pickle=False) as archive:
        neighbors, distances = archive["neighbors"], archive["distances"]
        metadata = json.loads(archive["metadata_json"].item())
    expected_neighbors, expected_distances = import_module("04_build_knn_graph").knn_graph(coordinates, neighbors.shape[1])
    if not np.array_equal(neighbors, expected_neighbors) or not np.allclose(distances, expected_distances, rtol=1e-5, atol=1e-6):
        raise ValueError("PCA coordinates do not reproduce the colored kNN graph")
    values, label_ids = import_module("05_color_knn_edges").load_binding_values(
        Path(metadata["labels_file"]), Path(metadata["smiles_file"]))
    if label_ids.tolist() != molecule_ids:
        raise ValueError("Binding labels and cycle molecules have different orders")
    print(f"Fitting holonomy frames: n-loc={args.n_loc}, ridge={args.ridge:g}", flush=True)
    frames, frame_valid = fit_binding_frames(coordinates, values, args.n_loc, args.ridge, args.min_rank_ratio)
    print("Computing cycle rotations", flush=True)
    angles = cycle_rotation_angles(cycles, frames, frame_valid, neighbors, args.min_overlap)
    print("Ranking figure-eights", flush=True)

    def progress(completed, total):
        if completed % 100 == 0 or completed == total:
            print(f"Hubs processed: {completed}/{total}", flush=True)

    matches = find_figure_eights(cycles, molecule_ids, args.max_per_hub, angles, progress)
    payload = {
        "cycles_file": str(args.cycles.resolve()), "cycles_sha256": input_hash,
        "colored_graph_file": source["colored_graph_file"], "colored_graph_sha256": source["colored_graph_sha256"],
        "pca_file": str(args.pca.resolve()), "pca_sha256": pca_hash,
        "parameters": dict(max_per_hub=args.max_per_hub, n_loc=args.n_loc, ridge=args.ridge,
                           min_overlap=args.min_overlap, min_rank_ratio=args.min_rank_ratio),
        "molecule_ids": molecule_ids, "figure_eights": matches,
    }
    hashes = {args.cycles: input_hash, graph_path: source["colored_graph_sha256"], args.pca: pca_hash}
    hashes.update({Path(metadata[f"{name}_file"]): metadata[f"{name}_sha256"] for name in ("knn", "labels", "smiles")})
    if any(file_hash(path) != digest for path, digest in hashes.items()):
        raise ValueError("Inputs changed during figure-eight matching")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("w" if args.overwrite else "x") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output appeared during matching: {output_path}")
    partial_path.replace(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
