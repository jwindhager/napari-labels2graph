from enum import Enum

import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.layers import Graph as GraphLayer
from napari.layers import Labels as LabelsLayer
from napari.viewer import Viewer
from napari_graph import DirectedGraph, UndirectedGraph
from skimage.measure import regionprops

from .delaunay_graph import get_delaunay_neighbors
from .distance_graph import (
    get_centroid_distance_neighbors,
    get_euclidean_contour_distance_neighbors,
    get_euclidean_expansion_distance_neighbors,
)


class DistanceType(Enum):
    centroid_distance = 0
    contour_distance = 1
    expansion_distance = 2


@magic_factory(call_button="Create graph")
def make_delaunay_graph_layer(
    viewer: Viewer,
    labels_layer: LabelsLayer,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str = "Qbb Qc Qz Q12",
) -> None:
    labels_img = np.asarray(labels_layer.data)
    props = regionprops(labels_img)
    labels = np.array([p.label for p in props])
    centroids = pd.DataFrame([p.centroid for p in props], index=labels)
    neighbors = get_delaunay_neighbors(
        labels,
        centroids,
        furthest_site=furthest_site,
        incremental=incremental,
        qhull_options=qhull_options,
    )
    graph = UndirectedGraph(edges=neighbors, coords=centroids)
    graph_layer = GraphLayer(graph)
    viewer.add_layer(graph_layer)


@magic_factory(
    call_button="Create graph",
    distance_metric={"choices": ["euclidean"]},
    maximum_distance={"min": 0.0},
    maximum_k_neighbors={"min": 0},
    post_expansion_connectivity={"choices": [4, 8]},
)
def make_distance_graph_layer(
    viewer: Viewer,
    labels_layer: LabelsLayer,
    distance_type: DistanceType = DistanceType.centroid_distance,
    distance_metric: str = "euclidean",
    maximum_distance: float = 0.0,
    maximum_k_neighbors: int = 0,
    post_expansion_connectivity: int = 4,
) -> None:
    labels_img = np.asarray(labels_layer.data)
    props = regionprops(labels_img)
    labels = np.array([p.label for p in props])
    centroids = pd.DataFrame([p.centroid for p in props], index=labels)
    if distance_type == DistanceType.centroid_distance:
        neighbors, _ = get_centroid_distance_neighbors(
            labels,
            centroids.to_numpy(),
            dist_metric=distance_metric,
            max_dist=maximum_distance,
            max_k=maximum_k_neighbors,
        )
    elif distance_type == DistanceType.contour_distance:
        if distance_metric != "euclidean":
            raise ValueError("Contour distance requires euclidean metric.")
        neighbors, _ = get_euclidean_contour_distance_neighbors(
            labels_img, max_dist=maximum_distance, max_k=maximum_k_neighbors
        )
    elif distance_type == DistanceType.expansion_distance:
        if distance_metric != "euclidean":
            raise ValueError("Expansion distance requires euclidean metric.")
        if maximum_distance <= 0.0:
            raise ValueError("Expansion distance requires maximum distance.")
        if maximum_k_neighbors > 0:
            raise ValueError("Expansion distance does not support maximum k.")
        neighbors = get_euclidean_expansion_distance_neighbors(
            labels_img,
            expansion_dist=maximum_distance,
            post_expansion_connectivity=post_expansion_connectivity,
        )
    else:
        raise NotImplementedError()
    if maximum_k_neighbors > 0:
        graph = DirectedGraph(edges=neighbors, coords=centroids)
    else:
        graph = UndirectedGraph(edges=neighbors, coords=centroids)
    graph_layer = GraphLayer(graph)
    viewer.add_layer(graph_layer)
