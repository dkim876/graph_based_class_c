# 08_plot_figure_eights.py

import argparse
from collections import Counter
from importlib import import_module
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator


EDGE_COLORS = {1: "tab:red", 0: "tab:blue"}


def rotation_key(pair):
    angles = pair["lobe_twist_degrees"]
    return -max(angles), -min(angles), pair["figure_eight_id"]


def select_groups(pairs, count):
    ordered = sorted(pairs, key=rotation_key)
    distinct_cycles, distinct_hubs, used_cycles, used_hubs = [], [], set(), set()
    for pair in ordered:
        cycles = {frozenset(pair[name]["nodes"][:-1]) for name in ("lobe1", "lobe2")}
        if len(distinct_cycles) < count and not used_cycles.intersection(cycles):
            distinct_cycles.append(pair)
            used_cycles.update(cycles)
        if len(distinct_hubs) < count and pair["hub"] not in used_hubs:
            distinct_hubs.append(pair)
            used_hubs.add(pair["hub"])
        if len(distinct_cycles) == len(distinct_hubs) == count:
            break
    longest = sorted(pairs, key=lambda pair: (-pair["total_length"], *rotation_key(pair)))
    return dict(strongest_rotation=ordered[:count], distinct_cycles=distinct_cycles,
                longest=longest[:count], distinct_hubs=distinct_hubs)


def plot_hub_counts(pairs, output, dpi):
    counts = sorted(Counter(pair["hub"] for pair in pairs).items(), key=lambda item: (-item[1], item[0]))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(range(len(counts)), [count for _, count in counts], width=1)
    step = max(1, math.ceil(len(counts) / 16))
    positions = list(range(0, len(counts), step))
    if positions[-1] != len(counts) - 1:
        positions.append(len(counts) - 1)
    axis.set_xticks(positions, [str(counts[index][0]) for index in positions], rotation=45, ha="right")
    axis.set(title="Saved figure-eights per hub", xlabel="Hub", ylabel="Saved figure-eights")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.margins(x=0, y=0.08)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def lobe_positions(cycle, hub, side):
    vertices = cycle["nodes"][:-1]
    hub_position = vertices.index(hub)
    ordered = vertices[hub_position:] + vertices[:hub_position]
    radius = 1.15 + 0.30 * (len(vertices) - 3)
    first_angle = 0 if side < 0 else math.pi
    positions = {hub: (0.0, 0.0)}
    for index, node in enumerate(ordered[1:], start=1):
        angle = first_angle + 2 * math.pi * index / len(ordered)
        positions[node] = (side * radius + radius * math.cos(angle), radius * math.sin(angle))
    return positions, radius


def draw_pair(axis, pair):
    hub = pair["hub"]
    left, left_radius = lobe_positions(pair["lobe1"], hub, -1)
    right, right_radius = lobe_positions(pair["lobe2"], hub, 1)
    positions = {**left, **right}
    radius = max(left_radius, right_radius)
    axis.set(xlim=(-2 * left_radius - 0.4, 2 * right_radius + 0.4),
             ylim=(-radius - 0.4, radius + 0.8), aspect="equal")
    axis.axis("off")
    patches = {}
    for node, position in positions.items():
        patch = Circle(position, radius=0.26 if node == hub else 0.23,
                       facecolor="gold" if node == hub else "white", edgecolor="black", linewidth=1, zorder=3)
        axis.add_patch(patch)
        patches[node] = patch
        axis.text(*position, str(node), ha="center", va="center", fontsize=10, zorder=4)
    for name, angle, x in zip(("lobe1", "lobe2"), pair["lobe_twist_degrees"], (-left_radius, right_radius)):
        cycle = pair[name]
        vertices = cycle["nodes"][:-1]
        center_x = sum(positions[node][0] for node in vertices) / len(vertices)
        center_y = sum(positions[node][1] for node in vertices) / len(vertices)
        axis.text(center_x, center_y, f"Cycle {cycle['cycle_id']}", fontsize=9, ha="center", va="center", zorder=4)
        edges = zip(cycle["nodes"][:-1], cycle["nodes"][1:], cycle["colors"], cycle["pki_changes"])
        for source, target, color, weight in edges:
            axis.add_patch(FancyArrowPatch(positions[source], positions[target], patchA=patches[source],
                                           patchB=patches[target], shrinkA=3, shrinkB=3, arrowstyle="-|>",
                                           mutation_scale=15, linewidth=1.5, color=EDGE_COLORS[color], zorder=2))
            midpoint = tuple((first + second) / 2 for first, second in zip(positions[source], positions[target]))
            axis.text(*midpoint, f"{weight:.2f}", ha="center", va="center", fontsize=8, color=EDGE_COLORS[color],
                      zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1))
        axis.text(x, radius + 0.48, f"Twist {angle:.2f}°", ha="center", va="center", fontsize=11)


def plot_group(pairs, output, title, dpi):
    columns = min(3, len(pairs))
    rows = math.ceil(len(pairs) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows + 0.8), squeeze=False)
    figure.suptitle(title)
    for axis, pair in zip(axes.flat, pairs):
        draw_pair(axis, pair)
    for axis in list(axes.flat)[len(pairs):]:
        axis.axis("off")
    figure.legend(handles=[Line2D([], [], color=EDGE_COLORS[1], label="D2"),
                           Line2D([], [], color=EDGE_COLORS[0], label="D3"),
                           Line2D([], [], linestyle="none", marker="o", markerfacecolor="gold",
                                  markeredgecolor="black", label="Hub")], loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parents[1]
    pairing = import_module("07_find_figure_eights")
    cycles_path = root / pairing.CYCLES_FILE
    default_pairs = cycles_path.with_name(cycles_path.stem + "_figure_eights_top100_twist.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-eights", type=Path, default=default_pairs)
    parser.add_argument("--out-dir", type=Path, default=root / "figures")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.dpi < 1:
        parser.error("Count and DPI must be positive")
    definitions = [("strongest_rotation", "02", "Most twist"), ("distinct_cycles", "03", "Distinct cycles"),
                   ("longest", "04", "Longest figure-eights"), ("distinct_hubs", "05", "Distinct hubs")]
    hub_path = args.out_dir / "01_hub_counts.png"
    plots = [(group, args.out_dir / f"{prefix}_{group}.png", title) for group, prefix, title in definitions]
    if not args.overwrite and any(path.exists() for path in [hub_path, *(path for _, path, _ in plots)]):
        parser.error("Output files already exist; use --overwrite")
    print("Loading figure-eights", flush=True)
    with args.figure_eights.open() as stream:
        payload = json.load(stream)
    for name in ("cycles", "colored_graph", "pca"):
        if pairing.file_hash(Path(payload[f"{name}_file"])) != payload[f"{name}_sha256"]:
            raise ValueError(f"Figure-eights do not match the {name} input")
    pairs = payload["figure_eights"]
    if not pairs:
        parser.error("No figure-eight pairs are available")
    for pair in pairs:
        angles = pair["lobe_twist_degrees"]
        if len(angles) != 2 or any(not math.isfinite(angle) or not 0 <= angle <= 180 for angle in angles):
            raise ValueError("Figure-eights require two valid rotation angles")
    groups = select_groups(pairs, args.count)
    selected_cycles = {cycle["cycle_id"]: cycle for group in groups.values() for pair in group
                       for cycle in (pair["lobe1"], pair["lobe2"])}
    _, identifiers = pairing.validate_cycles(dict(cycles=list(selected_cycles.values()),
                                                  colored_graph_file=payload["colored_graph_file"],
                                                  colored_graph_sha256=payload["colored_graph_sha256"]))
    if identifiers != payload["molecule_ids"]:
        raise ValueError("Figure-eights and graph molecules have different orders")
    for group in groups.values():
        for pair in group:
            if set(pair["lobe1"]["nodes"]) & set(pair["lobe2"]["nodes"]) != {pair["hub"]}:
                raise ValueError("Figure-eight lobes must share exactly their hub")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    plot_hub_counts(pairs, hub_path, args.dpi)
    print(f"Saved: {hub_path}")
    for group, path, title in plots:
        plot_group(groups[group], path, title, args.dpi)
        print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
