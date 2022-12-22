import numpy as np
from scipy.spatial import Delaunay


def get_delaunay_neighbors(
    labels: np.ndarray,
    centroids: np.ndarray,
    furthest_site: bool = False,
    incremental: bool = False,
    qhull_options: str = "Qbb Qc Qz Q12",
) -> np.ndarray:
    tri = Delaunay(
        centroids,
        furthest_site=furthest_site,
        incremental=incremental,
        qhull_options=qhull_options,
    )
    neighbors = np.unique(
        [
            sorted([labels[i], labels[(i + 1) % len(simplex)]])
            for simplex in tri.simplices
            for i in range(len(simplex))
        ],
        axis=0,
    )
    neighbors = np.vstack((neighbors, neighbors[:, ::-1]))
    return neighbors
