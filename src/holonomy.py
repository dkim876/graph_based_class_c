# holonomy.py

import numpy as np
from scipy.spatial import cKDTree


def fit_binding_frames(coordinates, values, n_loc=256, ridge=10.0, min_rank_ratio=0.01):
    if np.iscomplexobj(coordinates) or np.iscomplexobj(values):
        raise ValueError("Coordinates and binding values must be real")
    coordinates, values = np.asarray(coordinates, dtype=float), np.asarray(values, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2 or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite with at least two dimensions")
    node_count, dimension = coordinates.shape
    if values.shape != (node_count, 2) or not np.isfinite(values).all():
        raise ValueError("Binding values must contain aligned finite D2/D3 values")
    if not dimension < n_loc < node_count or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("Require dimension < n-loc < node count and positive finite ridge")
    if not 0 < min_rank_ratio < 1:
        raise ValueError("Minimum rank ratio must be between zero and one")
    _, nearest = cKDTree(coordinates).query(coordinates, k=n_loc + 1)
    gradients = np.empty((node_count, dimension, 2))
    regularizer = ridge * np.eye(dimension)
    for node, row in enumerate(nearest):
        indices = row[row != node][:n_loc]
        local_coordinates, local_values = coordinates[indices], values[indices]
        centered_coordinates = local_coordinates - local_coordinates.mean(axis=0)
        centered_values = local_values - local_values.mean(axis=0)
        gradients[node] = np.linalg.solve(centered_coordinates.T @ centered_coordinates + regularizer,
                                          centered_coordinates.T @ centered_values)
    if not np.isfinite(gradients).all():
        raise ValueError("Fitted gradients must be finite")
    singular_values = np.linalg.svd(gradients, compute_uv=False)
    largest, smallest = singular_values[:, 0] ** 2, singular_values[:, 1] ** 2
    valid = (largest > 1e-12) & (smallest / np.maximum(largest, 1e-12) >= min_rank_ratio)
    valid &= (singular_values[:, 1] > 1e-3 * np.maximum(singular_values[:, 0], 1e-12))
    valid &= np.all(np.linalg.norm(gradients, axis=1) > 1e-12, axis=1)
    frames, triangular = np.linalg.qr(gradients)
    frames *= np.where(np.diagonal(triangular, axis1=-2, axis2=-1) < 0, -1.0, 1.0)[:, None, :]
    return frames, valid


def cycle_rotation_angles(cycles, frames, frame_valid, neighbors, min_overlap=0.1):
    if not 0 < min_overlap < 1:
        raise ValueError("Minimum overlap must be between zero and one")
    node_count, k = neighbors.shape
    if frames.ndim != 3 or frames.shape[0] != node_count or frames.shape[2] != 2:
        raise ValueError("Frames must have shape (nodes, dimensions, 2)")
    if frame_valid.shape != (node_count,) or frame_valid.dtype != np.bool_:
        raise ValueError("Frame validity must be an aligned boolean array")
    if not np.issubdtype(neighbors.dtype, np.integer) or np.any(neighbors < 0) or np.any(neighbors >= node_count):
        raise ValueError("Invalid neighbor indices")
    selected = frames[frame_valid]
    if not np.isfinite(selected).all() or not np.allclose(np.swapaxes(selected, 1, 2) @ selected, np.eye(2), atol=1e-8, rtol=0):
        raise ValueError("Valid frames must be finite and orthonormal")
    safe_frames = np.where(frame_valid[:, None, None], frames, 0.0)
    cross = np.einsum("nkqa,nqb->nkab", safe_frames[neighbors], safe_frames)
    left, singular_values, right = np.linalg.svd(cross)
    transport = (left @ right).reshape(-1, 2, 2)
    edge_valid = (frame_valid[:, None] & frame_valid[neighbors] & (singular_values[:, :, -1] > min_overlap)).ravel()
    edge_indices = np.full((node_count, node_count), -1, dtype=np.int32)
    edge_indices[np.arange(node_count)[:, None], neighbors] = np.arange(node_count * k).reshape(node_count, k)
    lengths = np.fromiter((len(cycle["nodes"]) - 1 for cycle in cycles), dtype=int, count=len(cycles))
    angles = np.full(len(cycles), np.nan)
    for length in np.unique(lengths):
        indices = np.flatnonzero(lengths == length)
        for offset in range(0, len(indices), 20000):
            batch = indices[offset:offset + 20000]
            nodes = np.array([cycles[index]["nodes"] for index in batch])
            edges = edge_indices[nodes[:, :-1], nodes[:, 1:]]
            if np.any(edges < 0):
                raise ValueError("Cycle contains an edge missing from the kNN graph")
            product = np.broadcast_to(np.eye(2), (len(batch), 2, 2)).copy()
            for column in range(length):
                product = transport[edges[:, column]] @ product
            valid = edge_valid[edges].all(axis=1) & (np.linalg.det(product) > 0)
            angles[batch[valid]] = np.degrees(np.arccos(np.clip(np.trace(product[valid], axis1=1, axis2=2) / 2, -1, 1)))
    return angles
