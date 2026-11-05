# 06_find_colored_cycles.py

import argparse
from collections import deque
import hashlib
from importlib import import_module
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


COLORED_FILE = "data/processed/layer18/q50_pca_whiten_k50_d2_d3_m50_increase5pct_colored.npz"


def load_graph(path):
    coloring = import_module("05_color_knn_edges")
    with np.load(path, allow_pickle=False) as archive:
        neighbors, distances = archive["neighbors"], archive["distances"]
        red_keep, blue_keep = archive["red_keep"], archive["blue_keep"]
        d2_changes, d3_changes = archive["d2_changes"], archive["d3_changes"]
        molecule_ids = archive["molecule_ids"].astype(str)
        metadata = json.loads(archive["metadata_json"].item())
    if metadata.get("color_targets") != {"red": "pKi_D2", "blue": "pKi_D3"}:
        raise ValueError("Expected an independent D2/D3 colored graph from Step 05")
    for name in ("knn", "labels", "smiles"):
        if hashlib.sha256(Path(metadata[f"{name}_file"]).read_bytes()).hexdigest() != metadata[f"{name}_sha256"]:
            raise ValueError(f"Colored graph does not match its {name} input")
    values, label_ids = coloring.load_binding_values(Path(metadata["labels_file"]), Path(metadata["smiles_file"]))
    if not np.array_equal(molecule_ids, label_ids):
        raise ValueError("Graph and labels have different molecule orders")
    with np.load(metadata["knn_file"], allow_pickle=False) as archive:
        if not np.array_equal(molecule_ids, archive["molecule_ids"]):
            raise ValueError("Colored graph and kNN input have different molecule orders")
        if not np.array_equal(neighbors, archive["neighbors"]) or not np.array_equal(distances, archive["distances"]):
            raise ValueError("Colored graph differs from its kNN input")
    changes, retained = coloring.color_edges(values, neighbors, metadata["min_increase_percent"])
    for index, (keep, change) in enumerate(((red_keep, d2_changes), (blue_keep, d3_changes))):
        if keep.dtype != np.bool_ or not np.array_equal(keep, retained[:, :, index]):
            raise ValueError("Retained edges do not match the target-specific threshold")
        if np.iscomplexobj(change) or not np.array_equal(change, changes[:, :, index]):
            raise ValueError("Edge changes do not match the binding labels")
    return neighbors, red_keep, blue_keep, d2_changes, d3_changes, molecule_ids, metadata


def build_adjacency(neighbors, red_keep, blue_keep, d2_changes, d3_changes):
    node_count = len(neighbors)
    outgoing = {color: [[] for _ in range(node_count)] for color in (0, 1)}
    incoming = {color: [[] for _ in range(node_count)] for color in (0, 1)}
    edge_data = {}
    for color, keep, changes in ((1, red_keep, d2_changes), (0, blue_keep, d3_changes)):
        for source, column in zip(*np.nonzero(keep)):
            source, target = int(source), int(neighbors[source, column])
            outgoing[color][source].append(target)
            incoming[color][target].append(source)
            edge_data[source, target, color] = float(changes[source, column])
    return outgoing, incoming, edge_data


def possible_hubs(outgoing, incoming):
    node_count = len(outgoing[0])
    rows, columns = [], []
    for color in (0, 1):
        for source, targets in enumerate(outgoing[color]):
            rows.extend([source] * len(targets))
            columns.extend(targets)
    graph = csr_matrix((np.ones(len(rows), dtype=np.uint8), (rows, columns)), shape=(node_count, node_count))
    component_count, labels = connected_components(graph, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=component_count)
    hubs = []
    for node in range(node_count):
        if sizes[labels[node]] < 5:
            continue
        outgoing_neighbors = {target for color in (0, 1) for target in outgoing[color][node] if labels[target] == labels[node]}
        incoming_neighbors = {source for color in (0, 1) for source in incoming[color][node] if labels[source] == labels[node]}
        if len(outgoing_neighbors) >= 2 and len(incoming_neighbors) >= 2:
            hubs.append(node)
    return hubs, labels, sizes


def color_return_distances(incoming, start, component_labels, max_depth):
    distances = [[max_depth + 1] * len(component_labels) for _ in range(2)]
    distances[0][start] = 0
    queue = deque([(start, 0)])
    while queue:
        target, phase = queue.popleft()
        distance = distances[phase][target] + 1
        if distance > max_depth:
            continue
        for source in incoming[phase][target]:
            if component_labels[source] != component_labels[start]:
                continue
            for previous_phase in ((0, 1) if phase == 0 else (1,)):
                if distance < distances[previous_phase][source]:
                    distances[previous_phase][source] = distance
                    queue.append((source, previous_phase))
    return distances


def canonical_cycle(nodes, colors):
    edges = list(zip(nodes[:-1], colors))
    red_start = next(index for index, color in enumerate(colors) if color == 1 and colors[index - 1] == 0)
    return tuple(edges[red_start:] + edges[:red_start])


def find_cycles(outgoing, incoming, min_length, max_length, seed, progress=None):
    if min_length < 3 or max_length < min_length:
        raise ValueError("Cycle lengths must satisfy 3 <= min_length <= max_length")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    node_count = len(outgoing[1])
    hubs, component_labels, component_sizes = possible_hubs(outgoing, incoming)
    hub_components = {component_labels[hub] for hub in hubs}
    is_hub = [False] * node_count
    for hub in hubs:
        is_hub[hub] = True
    generator = random.Random(seed)
    successors = {color: [] for color in (0, 1)}
    for color in (0, 1):
        for node in range(node_count):
            targets = sorted((target, color) for target in outgoing[color][node]
                             if component_labels[target] == component_labels[node])
            generator.shuffle(targets)
            successors[color].append(targets)
    starts = [node for node in range(node_count) if component_labels[node] in hub_components
              and successors[1][node] and any(component_labels[source] == component_labels[node]
                                             for source in incoming[0][node])]
    red_successors = [successors[1][node] + successors[0][node] for node in range(node_count)]
    if progress is not None:
        progress(0, len(starts))
    for completed, start in enumerate(starts, 1):
        limit = min(max_length, int(component_sizes[component_labels[start]]))
        distances = color_return_distances(incoming, start, component_labels, limit - 1)
        path, colors, stack = [start], [], [iter(successors[1][start])]
        visited = bytearray(node_count)
        visited[start] = 1
        hub_count = int(is_hub[start])
        while stack:
            try:
                target, color = next(stack[-1])
            except StopIteration:
                stack.pop()
                node = path.pop()
                visited[node] = 0
                hub_count -= is_hub[node]
                if colors:
                    colors.pop()
                continue
            length = len(path)
            if target == start:
                if color == 0 and min_length <= length <= limit and hub_count:
                    yield tuple(zip(path, colors + [0]))
                continue
            if visited[target]:
                continue
            if length + distances[color][target] > limit:
                continue
            path.append(target)
            colors.append(color)
            visited[target] = 1
            hub_count += is_hub[target]
            stack.append(iter(red_successors[target] if color else successors[0][target]))
        if progress is not None:
            progress(completed, len(starts))


def cycle_record(cycle, edge_data, cycle_id):
    nodes = [node for node, color in cycle]
    nodes.append(nodes[0])
    colors = [color for node, color in cycle]
    return {
        "cycle_id": cycle_id, "nodes": nodes, "colors": colors,
        "color_pattern": "".join("R" if color else "B" for color in colors),
        "length": len(cycle), "red_length": sum(colors), "blue_length": len(colors) - sum(colors),
        "linear_flips": sum(first != second for first, second in zip(colors[:-1], colors[1:])),
        "cyclic_color_transitions": sum(color != colors[(index + 1) % len(colors)] for index, color in enumerate(colors)),
        "pki_changes": [edge_data[source, target, color] for source, target, color in zip(nodes[:-1], nodes[1:], colors)],
    }


def main():
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--colored", type=Path, default=repository_root / COLORED_FILE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.min_length < 3 or args.max_length < args.min_length:
        parser.error("Lengths must satisfy 3 <= min-length <= max-length")
    if args.seed < 0:
        parser.error("Seed must be nonnegative")
    output_path = args.out or args.colored.with_name(f"{args.colored.stem}_cycles_{args.min_length}-{args.max_length}.json")
    if output_path.suffix != ".json" or output_path.resolve() == args.colored.resolve():
        parser.error("Output must be a JSON file distinct from the colored graph")
    partial_path = output_path.with_suffix(".json.partial")
    if not args.overwrite and (output_path.exists() or partial_path.exists()):
        parser.error(f"Output or partial output already exists: {output_path}; use --out or --overwrite")
    input_hash = hashlib.sha256(args.colored.read_bytes()).hexdigest()
    neighbors, red_keep, blue_keep, d2_changes, d3_changes, _, _ = load_graph(args.colored)
    outgoing, incoming, edge_data = build_adjacency(neighbors, red_keep, blue_keep, d2_changes, d3_changes)
    payload = {
        "colored_graph_file": str(args.colored.resolve()),
        "colored_graph_sha256": input_hash,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    started = time.monotonic()
    with partial_path.open("w" if args.overwrite else "x") as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False)[:-2] + ',\n  "cycles": [\n')

        def progress(completed, total):
            if completed % 100 == 0 or completed == total:
                stream.flush()
                print(f"Starts: {completed}/{total}; cycles saved: {saved}; elapsed seconds: {time.monotonic() - started:.1f}", flush=True)

        for saved, cycle in enumerate(find_cycles(outgoing, incoming, args.min_length, args.max_length, args.seed, progress), 1):
            if saved > 1:
                stream.write(",\n")
            stream.write(json.dumps(cycle_record(cycle, edge_data, saved), separators=(",", ":"), allow_nan=False))
        if hashlib.sha256(args.colored.read_bytes()).hexdigest() != input_hash:
            raise ValueError("Colored graph changed during the search")
        stream.write("\n  ]\n}\n")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output appeared during the search: {output_path}")
    partial_path.replace(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
