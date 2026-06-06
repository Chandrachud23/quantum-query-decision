"""Low-noise island detection by pruning outlier qubits/edges from the coupling graph."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import networkx as nx


@dataclass
class Island:
    qubits: list[int]
    edges: list[tuple[int, int]]
    mean_edge_error: float

    @property
    def size(self) -> int:
        return len(self.qubits)


def _zscore_outliers(scores: dict, z_thresh: float) -> set:
    """Keys whose error is more than z_thresh stddevs above the mean."""
    vals = [v for v in scores.values() if v < 1.0] or list(scores.values())
    mu = statistics.mean(vals)
    sigma = statistics.pstdev(vals) or 1e-12
    return {k for k, v in scores.items() if (v - mu) / sigma > z_thresh}


def find_islands(coupling_map: list[tuple[int, int]],
                 edge_errors: dict[tuple[int, int], float],
                 qubit_ro_errors: dict[int, float],
                 z_thresh: float = 1.0) -> list[Island]:
    """Prune outliers, return connected components of the surviving graph, largest first."""
    bad_qubits = _zscore_outliers(qubit_ro_errors, z_thresh)
    bad_edges = _zscore_outliers(edge_errors, z_thresh)

    g = nx.Graph()
    for (u, v) in coupling_map:
        if (u, v) in bad_edges or (v, u) in bad_edges:
            continue
        if u in bad_qubits or v in bad_qubits:
            continue
        g.add_edge(u, v)

    islands: list[Island] = []
    for comp in nx.connected_components(g):
        nodes = sorted(comp)
        sub = g.subgraph(nodes)
        e = [tuple(sorted(x)) for x in sub.edges()]
        errs = [edge_errors.get(x, edge_errors.get(x[::-1], 0.0)) for x in e]
        mean_err = sum(errs) / len(errs) if errs else 0.0
        islands.append(Island(qubits=nodes, edges=e, mean_edge_error=mean_err))

    islands.sort(key=lambda isl: isl.size, reverse=True)
    return islands


def largest_island_capacity(islands: list[Island]) -> int:
    """Width of the largest island."""
    return max((isl.size for isl in islands), default=0)
