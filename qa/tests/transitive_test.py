import pytest
import time
from LaSSI.HOnK.conceptnet.transitive_closure import floyd_warshall, DSU

def sort_adjacency_list(adj):
    for k, v in adj.items():
        adj[k] = sorted(v)

def timer(callback):
    start_time = time.perf_counter()
    callback()
    print(f"Test took {(time.perf_counter() - start_time) * 1000:.3f} ms")


def test_floyd_1():
    adjacency_list = {   ## just to see if it works with directed graph
        "A": ["B"],
        "B": ["C"],
        "C": []
    }
    expected_result = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": []
    }

    def f():
        floyd_warshall(adjacency_list)

    timer(f)
    sort_adjacency_list(adjacency_list)
    assert adjacency_list == expected_result


def test_floyd_2():
    adjacency_list = {
        "A": ["B", "C"],
        "B": ["A"],
        "C": ["A"]
    }
    expected_result = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"]
    }

    def f():
        floyd_warshall(adjacency_list)

    timer(f)
    sort_adjacency_list(adjacency_list)
    assert adjacency_list == expected_result

def test_DSU_1():
    adjacency_list = {
        "A": ["B", "C"],
        "B": ["A"],
        "C": ["A"]
    }
    expected_result = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"]
    }

    def f():
        DSU(adjacency_list)

    timer(f)
    sort_adjacency_list(adjacency_list)
    assert adjacency_list == expected_result


def test_DSU_2():
    adjacency_list = {
        "A": ["B"],
        "B": ["A", "C"],
        "C": ["B", "D"],
        "D": ["C"],
        "E": ["F"],
        "F": ["E", "G"],
        "G": ["F", "H"],
        "H": ["G", "I"],
        "I": ["H"],
        "J": ["K"],
        "K": ["J", "L"],
        "L": ["K"]
    }

    expected_result = {
        "A": ["B", "C", "D"],
        "B": ["A", "C", "D"],
        "C": ["A", "B", "D"],
        "D": ["A", "B", "C"],
        "E": ["F", "G", "H", "I"],
        "F": ["E", "G", "H", "I"],
        "G": ["E", "F", "H", "I"],
        "H": ["E", "F", "G", "I"],
        "I": ["E", "F", "G", "H"],
        "J": ["K", "L"],
        "K": ["J", "L"],
        "L": ["J", "K"]
    }

    def f():
        DSU(adjacency_list)

    timer(f)
    sort_adjacency_list(adjacency_list)
    assert adjacency_list == expected_result

