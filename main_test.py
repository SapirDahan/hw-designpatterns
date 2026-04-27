# Unit tests for the MST system.
# Covers all 8 combinations: 2 algorithms x 2 input types x 2 output types.

import pytest
from main import mst, kruskal, prim, TotalWeight, EdgeList

# ---------------------------------------------------------------------------
# Shared test graph (used in all tests):
#
#   0 --1-- 1
#   |       |
#   4       2
#   |       |
#   2 --3-- 3
#
# The unique MST uses edges: 0-1 (w=1), 1-3 (w=2), 2-3 (w=3) -> total weight = 6
# ---------------------------------------------------------------------------

# Same graph as adjacency list (both directions listed for undirected edges)
ADJ = {0: [(1,1),(2,4)], 1: [(0,1),(3,2)], 2: [(0,4),(3,3)], 3: [(1,2),(2,3)]}

# Same graph as a weight matrix (0 = no edge)
MAT = [[0,1,4,0],[1,0,0,2],[4,0,0,3],[0,2,3,0]]

EXPECTED_WEIGHT = 6.0
EXPECTED_EDGES  = [(0,1,1), (1,3,2), (2,3,3)]  # each tuple is (node_a, node_b, weight), smaller node first

# ---------------------------------------------------------------------------
# A second, larger graph used in the additional correctness checks.
# 5 nodes, so the MST must have exactly 4 edges and span all 5 nodes.
#
#   0 --2-- 1 --3-- 2
#   |               |
#   5               1
#   |               |
#   4 ------4------ 3
#
# MST edges: (2-3, w=1), (0-1, w=2), (1-2, w=3), (3-4, w=4) -> total weight = 10
# ---------------------------------------------------------------------------

ADJ2 = {0: [(1,2),(4,5)], 1: [(0,2),(2,3)], 2: [(1,3),(3,1)], 3: [(2,1),(4,4)], 4: [(0,5),(3,4)]}
MAT2 = [[0,2,0,0,5],[2,0,3,0,0],[0,3,0,1,0],[0,0,1,0,4],[5,0,0,4,0]]

EXPECTED_WEIGHT2 = 10.0
EXPECTED_EDGES2  = [(0,1,2), (1,2,3), (2,3,1), (3,4,4)]  # each tuple is (node_a, node_b, weight), smaller node first


def sort_edges_for_comparison(edges):
    """
    Makes edges comparable regardless of direction.
    (u,v,w) and (v,u,w) are the same undirected edge, so we always put
    the smaller node first and then sort the whole list.
    """
    return sorted((min(u,v), max(u,v), w) for u,v,w in edges)


# ---------------------------------------------------------------------------
# The 8 required combinations
# ---------------------------------------------------------------------------

class TestKruskalAdjacencyListTotalWeight:
    def test_weight(self):
        # Kruskal + adjacency list + weight-only output
        assert mst(kruskal, ADJ, TotalWeight) == EXPECTED_WEIGHT
        assert mst(kruskal, ADJ2, TotalWeight) == EXPECTED_WEIGHT2

class TestKruskalAdjacencyListEdgeList:
    def test_edges(self):
        # Kruskal + adjacency list + full edge list output
        assert sort_edges_for_comparison(mst(kruskal, ADJ, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES)
        assert sort_edges_for_comparison(mst(kruskal, ADJ2, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES2)

class TestKruskalWeightMatrixTotalWeight:
    def test_weight(self):
        # Kruskal + weight matrix + weight-only output
        assert mst(kruskal, MAT, TotalWeight) == EXPECTED_WEIGHT
        assert mst(kruskal, MAT2, TotalWeight) == EXPECTED_WEIGHT2

class TestKruskalWeightMatrixEdgeList:
    def test_edges(self):
        # Kruskal + weight matrix + full edge list output
        assert sort_edges_for_comparison(mst(kruskal, MAT, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES)
        assert sort_edges_for_comparison(mst(kruskal, MAT2, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES2)

class TestPrimAdjacencyListTotalWeight:
    def test_weight(self):
        # Prim + adjacency list + weight-only output
        assert mst(prim, ADJ, TotalWeight) == EXPECTED_WEIGHT
        assert mst(prim, ADJ2, TotalWeight) == EXPECTED_WEIGHT2

class TestPrimAdjacencyListEdgeList:
    def test_edges(self):
        # Prim + adjacency list + full edge list output
        assert sort_edges_for_comparison(mst(prim, ADJ, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES)
        assert sort_edges_for_comparison(mst(prim, ADJ2, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES2)

class TestPrimWeightMatrixTotalWeight:
    def test_weight(self):
        # Prim + weight matrix + weight-only output
        assert mst(prim, MAT, TotalWeight) == EXPECTED_WEIGHT
        assert mst(prim, MAT2, TotalWeight) == EXPECTED_WEIGHT2

class TestPrimWeightMatrixEdgeList:
    def test_edges(self):
        # Prim + weight matrix + full edge list output
        assert sort_edges_for_comparison(mst(prim, MAT, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES)
        assert sort_edges_for_comparison(mst(prim, MAT2, EdgeList)) == sort_edges_for_comparison(EXPECTED_EDGES2)


# ---------------------------------------------------------------------------
# Additional correctness checks
# ---------------------------------------------------------------------------

class TestEdgeCount:
    # A spanning tree on N nodes always has exactly N-1 edges.
    # Tested on both graphs: 4-node (expect 3 edges) and 5-node (expect 4 edges).
    def test_kruskal_adj(self):
        assert len(mst(kruskal, ADJ, EdgeList)) == len(ADJ) - 1
        assert len(mst(kruskal, ADJ2, EdgeList)) == len(ADJ2) - 1

    def test_kruskal_mat(self):
        assert len(mst(kruskal, MAT, EdgeList)) == len(MAT) - 1
        assert len(mst(kruskal, MAT2, EdgeList)) == len(MAT2) - 1

    def test_prim_adj(self):
        assert len(mst(prim, ADJ, EdgeList)) == len(ADJ) - 1
        assert len(mst(prim, ADJ2, EdgeList)) == len(ADJ2) - 1

    def test_prim_mat(self):
        assert len(mst(prim, MAT, EdgeList)) == len(MAT) - 1
        assert len(mst(prim, MAT2, EdgeList)) == len(MAT2) - 1


class TestAllNodesSpanned:
    # Every node in the graph must appear in at least one MST edge.
    # Tested on both graphs: 4-node and 5-node.
    def test_kruskal_adj(self):
        for graph in [ADJ, ADJ2]:
            edges = mst(kruskal, graph, EdgeList)
            nodes_in_mst = {u for u,v,w in edges} | {v for u,v,w in edges}
            assert nodes_in_mst == set(graph.keys())

    def test_kruskal_mat(self):
        for graph in [MAT, MAT2]:
            edges = mst(kruskal, graph, EdgeList)
            nodes_in_mst = {u for u,v,w in edges} | {v for u,v,w in edges}
            assert nodes_in_mst == set(range(len(graph)))

    def test_prim_adj(self):
        for graph in [ADJ, ADJ2]:
            edges = mst(prim, graph, EdgeList)
            nodes_in_mst = {u for u,v,w in edges} | {v for u,v,w in edges}
            assert nodes_in_mst == set(graph.keys())

    def test_prim_mat(self):
        for graph in [MAT, MAT2]:
            edges = mst(prim, graph, EdgeList)
            nodes_in_mst = {u for u,v,w in edges} | {v for u,v,w in edges}
            assert nodes_in_mst == set(range(len(graph)))


class TestTotalWeightMatchesEdgeListSum:
    # TotalWeight and EdgeList must agree on the total weight.
    # Tested on both graphs: 4-node (expect 6) and 5-node (expect 10).
    def test_kruskal_adj(self):
        for graph in [ADJ, ADJ2]:
            assert mst(kruskal, graph, TotalWeight) == sum(w for _,_,w in mst(kruskal, graph, EdgeList))

    def test_prim_adj(self):
        for graph in [ADJ, ADJ2]:
            assert mst(prim, graph, TotalWeight) == sum(w for _,_,w in mst(prim, graph, EdgeList))

    def test_kruskal_mat(self):
        for graph in [MAT, MAT2]:
            assert mst(kruskal, graph, TotalWeight) == sum(w for _,_,w in mst(kruskal, graph, EdgeList))

    def test_prim_mat(self):
        for graph in [MAT, MAT2]:
            assert mst(prim, graph, TotalWeight) == sum(w for _,_,w in mst(prim, graph, EdgeList))