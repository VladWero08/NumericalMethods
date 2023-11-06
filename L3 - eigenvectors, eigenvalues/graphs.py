import networkx
import pickle
import numpy as np
import typing as t
import matplotlib.pyplot as plt

def read_adjacency_matrix(graphs_file_path: str) -> t.Optional[list]:
    """Given a path to a `pickle` file that contains adjancency
    matrixs, load that pickle and return the array of matrixs."""

    try:
        with open(graphs_file_path, 'rb') as graphs_input:
            graph_adjacency_matrixs = pickle.load(graphs_input)
            return graph_adjacency_matrixs
    except Exception as e:
        print(f"Error occured while loading pickle {graphs_file_path}")

def display_graph(graph: networkx.classes.Graph) -> None:
    """Display given graph with `networkx.draw` and `matplotlib.pyplot`."""
    networkx.draw(graph, with_labels=True)
    plt.show()

def get_maximum_clique_in_graph(graph: networkx.classes.Graph) -> set:
    """Get the estimated value of the `maximum clique` that
    can be found in a graph."""
    maximum_clique = networkx.algorithms.approximation.max_clique(graph)
    return maximum_clique

def get_eigenvalues_in_graph(graph_adjacency_matrix: list) -> dict:
    """Get the unique eigenvalues, the most significant and
    least significant eigenvalue from a graph, represented by its
    adjacency matrix."""
    eigen = np.linalg.eigh(graph_adjacency_matrix)
    eigenvalues = eigen[0].round(decimals=4)

    eigenvalue_max = np.argmax(abs(eigenvalues))
    eigenvalue_min = np.argmin(abs(eigenvalues))
    eigenvalues_unique = np.unique(eigenvalues)

    return {
        "unique": eigenvalues_unique,
        "max": eigenvalue_max,
        "min": eigenvalue_min
    }

def check_graph_complete(eigenvalues: list) -> str:
    return "True" if len(eigenvalues) == 2 else "False"

def check_graph_bipartite(min_eigenvalue: float, max_eigenvalue: float) -> str:
    return "True" if min_eigenvalue == -1 * max_eigenvalue else "False"

def solve_graph(graph: networkx.classes.Graph, matrix: list) -> None:
    """Given a `graph` and its `adjacency matrix`, check if the
    graph is complete, bipartite and display its maximum clique."""
    graph_eigenvalues = get_eigenvalues_in_graph(matrix)
    graph_max_clique = get_maximum_clique_in_graph(graph)

    is_graph_complete = check_graph_complete(graph_eigenvalues["unique"])
    is_graph_bipartite = check_graph_bipartite(graph_eigenvalues["min"], graph_eigenvalues["max"])

    print(f"{graph_index + 1} graph:")
    print(f"Complete? {is_graph_complete}")
    print(f"Bipartite? {is_graph_bipartite}")
    print(f"Maximum clique: {graph_max_clique}")
    print()


if __name__ == "__main__":
    matrixs = read_adjacency_matrix("graphs.pickle")
    graphs_networkx = [networkx.convert_matrix.from_numpy_array(matrix) for matrix in matrixs]

    for graph_index, graph in enumerate(graphs_networkx):
        solve_graph(graph, matrixs[graph_index])