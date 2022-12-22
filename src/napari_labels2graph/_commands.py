from enum import Enum

import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari.layers import Graph as GraphLayer
from napari.layers import Labels as LabelsLayer
from napari.viewer import Viewer
from napari_graph import DirectedGraph, UndirectedGraph
from skimage.measure import regionprops

from .distance_graph import (
    get_centroid_distance_neighbors,
    get_euclidean_contour_distance_neighbors,
    get_euclidean_expansion_distance_neighbors,
)


class DistanceType(Enum):
    centroid_distance = 0
    contour_distance = 1
    expansion_distance = 2


@magic_factory
def make_delaunay_graph_layer(
    viewer: Viewer,
    labels_layer: LabelsLayer,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str = "Qbb Qc Qz Q12",
) -> None:
    raise NotImplementedError()  # TODO


@magic_factory(distance_metric={"choices": ["euclidean"]})
def make_distance_graph_layer(
    viewer: Viewer,
    labels_layer: LabelsLayer,
    distance_type: DistanceType = DistanceType.centroid_distance,
    distance_metric: str = "euclidean",
    max_distance: float = 0.0,
    max_k: int = 0,
) -> None:
    labels_img = np.asarray(labels_layer.data)
    props = regionprops(labels_img)
    labels = np.array([p.label for p in props])
    centroids = np.array([p.centroid for p in props])
    if distance_type == DistanceType.centroid_distance:
        neighbors, _ = get_centroid_distance_neighbors(
            labels,
            centroids,
            distance_metric=distance_metric,
            max_distance=max_distance,
            max_k=max_k,
        )
    elif distance_type == DistanceType.contour_distance:
        if distance_metric != "euclidean":
            raise ValueError("Contour distance requires euclidean metric.")
        neighbors, _ = get_euclidean_contour_distance_neighbors(
            labels_img, max_distance=max_distance, max_k=max_k
        )
    elif distance_type == DistanceType.expansion_distance:
        if distance_metric != "euclidean":
            raise ValueError("Expansion distance requires euclidean metric.")
        if max_distance <= 0.0:
            raise ValueError("Expansion distance requires maximum distance.")
        if max_k > 0:
            raise ValueError("Expansion distance does not support maximum k.")
        neighbors = get_euclidean_expansion_distance_neighbors(
            labels_img, expansion_dist=max_distance
        )
    else:
        raise NotImplementedError()
    coords = pd.DataFrame(centroids, index=labels)
    if max_k > 0:
        graph = DirectedGraph(edges=neighbors, coords=coords)
    else:
        graph = UndirectedGraph(edges=neighbors, coords=coords)
    graph_layer = GraphLayer(graph)
    viewer.add_layer(graph_layer)
