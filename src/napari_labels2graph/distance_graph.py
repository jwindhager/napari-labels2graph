from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist, squareform


def _expand_euclidean(
    labels_img: np.ndarray, expansion_dist: float
) -> np.ndarray:
    dists, (i, j) = distance_transform_edt(
        labels_img == 0, return_indices=True
    )
    return np.where(dists <= expansion_dist, labels_img[i, j], labels_img)


def _to_triu_indices(k: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    # https://stackoverflow.com/a/27088560 (optimized for numpy arrays)
    # Briefly, the formulas were derived using triangular numbers/roots
    i = n - 2 - np.floor((-8 * k + 4 * n * (n - 1) - 7) ** 0.5 / 2 - 0.5)
    j = k + i + 1 - (n * (n - 1) - (n - i) * (n - i - 1)) // 2
    return i.astype(int), j.astype(int)


def get_centroid_distance_neighbors(
    labels: np.ndarray,
    centroids: np.ndarray,
    dist_metric: str = "euclidean",
    max_dist: float = 0.0,
    max_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    condensed_dists = pdist(centroids, metric=dist_metric)
    if max_k > 0:
        max_k = min(max_k, len(labels) - 1)
        dist_mat = squareform(condensed_dists, checks=False).astype(float)
        np.fill_diagonal(dist_mat, np.inf)
        knn_ind = np.argpartition(dist_mat, max_k - 1)[:, :max_k]
        if max_dist > 0.0:
            knn_dists = np.take_along_axis(dist_mat, knn_ind, -1)
            ind1, ind2 = np.nonzero(knn_dists <= max_dist)
            ind2 = knn_ind[(ind1, ind2)]
        else:
            ind1 = np.repeat(np.arange(len(labels)), max_k)
            ind2 = np.ravel(knn_ind)
        dists = dist_mat[(ind1, ind2)]
    elif max_dist > 0.0:
        (condensed_ind,) = np.nonzero(condensed_dists <= max_dist)
        ind1, ind2 = _to_triu_indices(condensed_ind, len(labels))
        dists = condensed_dists[condensed_ind]
        ind1, ind2, dists = (
            np.concatenate((ind1, ind2)),
            np.concatenate((ind2, ind1)),
            np.concatenate((dists, dists)),
        )
    else:
        raise ValueError("Either `max_distance` or `max_k` must be specified.")
    neighbors = np.column_stack((labels[ind1], labels[ind2]))
    return neighbors, dists


def get_euclidean_contour_distance_neighbors(
    labels_img: np.ndarray, max_dist: float = 0.0, max_k: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.unique(labels_img)
    labels = labels[labels != 0]
    labels1, labels2, dists = [], [], []
    dmax = int(np.ceil(max_dist)) if max_dist > 0.0 else None
    for label in labels:
        if dmax is not None:
            patch_ind = np.nonzero(labels_img == label)
            patch_slices = tuple(
                slice(
                    max(0, np.amin(ind) - dmax),
                    min(labels_img.shape[dim], np.amax(ind) + dmax + 1),
                )
                for dim, ind in enumerate(patch_ind)
            )
            patch = labels_img[patch_slices]
            patch_labels = np.unique(patch)
            patch_labels = patch_labels[patch_labels != 0]
            current_neighbor_labels = patch_labels[patch_labels != label]
        else:
            patch = labels_img
            current_neighbor_labels = labels[labels != label]
        current_dist_transform = distance_transform_edt(patch != label)
        current_dists = np.array(
            [
                np.amin(current_dist_transform[patch == neighbor_label])
                for neighbor_label in current_neighbor_labels
            ]
        )
        if max_dist > 0.0:
            current_neighbor_labels = current_neighbor_labels[
                current_dists <= max_dist
            ]
            current_dists = current_dists[current_dists <= max_dist]
        if max_k > 0 and len(current_neighbor_labels) > max_k:
            knn_ind = np.argpartition(current_dists, max_k - 1)[:max_k]
            current_neighbor_labels = current_neighbor_labels[knn_ind]
            current_dists = current_dists[knn_ind]
        labels1 += [label] * len(current_neighbor_labels)
        labels2 += current_neighbor_labels.tolist()
        dists += current_dists.tolist()
    neighbors = np.column_stack((labels1, labels2))
    return neighbors, dists


def get_euclidean_expansion_distance_neighbors(
    labels_img: np.ndarray,
    expansion_dist: float,
    post_expansion_connectivity: int = 4,
) -> np.ndarray:
    if post_expansion_connectivity == 4:
        max_dist = 1.0
    elif post_expansion_connectivity == 8:
        max_dist = 1.5  # 2**0.5 + eps
    else:
        raise ValueError("`post_expansion_connectivity` must be 4 or 8.")
    expanded_labels_img = _expand_euclidean(labels_img, expansion_dist)
    neighbors, _ = get_euclidean_contour_distance_neighbors(
        expanded_labels_img, max_dist=max_dist
    )
    return neighbors
