"""
This file solves the Minimum Spanning Tree (MST) problem.
Given a weighted graph, it finds the subset of edges that connects all nodes
with the minimum total weight.

The system is built to be flexible -- it automatically handles two input formats,
supports two different output formats, and can run two different algorithms.
All combinations work without any changes to the core logic.
"""

from abc import ABC, abstractmethod
import heapq
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# MSTBuilder: manages what gets stored while the algorithm runs.
# The algorithm only calls add_edge() -- it never knows what is being stored.
# ---------------------------------------------------------------------------

class MSTBuilder(ABC):
    """Abstract base: defines the interface every builder must implement."""

    @abstractmethod
    def new_mst(self) -> Any:
        """Create and return a fresh empty MST state."""

    @abstractmethod
    def add_edge(self, mst: Any, u, v, weight: float) -> Any:
        """Add edge (u, v, weight) to the MST state and return it."""

    @abstractmethod
    def get_weight(self, mst: Any) -> float:
        """Return the current total weight stored in the MST state."""


class MSTBuilderKeepingWeight(MSTBuilder):
    """
    Lightweight builder -- stores only the running total weight.
    State is a one-element list so it stays mutable:
    the algorithm never needs to rebind the variable, it just updates mst[0].

    >>> b = MSTBuilderKeepingWeight()
    >>> mst = b.new_mst()
    >>> _ = b.add_edge(mst, 'A', 'B', 3.0)
    >>> _ = b.add_edge(mst, 'B', 'C', 5.0)
    >>> b.get_weight(mst)
    8.0
    """

    def new_mst(self) -> List:
        return [0.0] # [total_weight]

    def add_edge(self, mst: List, u, v, weight: float) -> List:
        mst[0] += weight # accumulate weight in place
        return mst

    def get_weight(self, mst: List) -> float:
        return mst[0]


class MSTBuilderKeepingEdges(MSTBuilderKeepingWeight):
    """
    Full builder -- stores the total weight AND the list of edges.
    State is [total_weight, edges_list].

    >>> b = MSTBuilderKeepingEdges()
    >>> mst = b.new_mst()
    >>> _ = b.add_edge(mst, 'A', 'B', 3.0)
    >>> _ = b.add_edge(mst, 'B', 'C', 5.0)
    >>> b.get_weight(mst)
    8.0
    >>> mst[1]  # index 1 holds the edges list
    [('A', 'B', 3.0), ('B', 'C', 5.0)]
    """

    def new_mst(self) -> List:
        return [0.0, []] # [total_weight, edges_list]

    def add_edge(self, mst: List, u, v, weight: float) -> List:
        mst[0] += weight # update running total
        mst[1].append((u, v, weight)) # record the edge
        return mst


# ---------------------------------------------------------------------------
# OutputType: decides which builder to use and how to extract the final answer.
# Each subclass picks the cheapest builder that is sufficient for its output.
# ---------------------------------------------------------------------------

class OutputType(ABC):
    """Abstract base: defines create_builder and extract_output."""

    @classmethod
    def create_builder(cls) -> MSTBuilder:
        raise NotImplementedError("Choose a specific output type")

    @classmethod
    def extract_output(cls, mst: Any) -> Any:
        raise NotImplementedError("Choose a specific output type")


class TotalWeight(OutputType):
    """
    Only needs the total weight -> uses the lightweight builder (no edge list allocated).
    This satisfies the requirement to save runtime when only the weight is needed.
    """

    @classmethod
    def create_builder(cls) -> MSTBuilder:
        return MSTBuilderKeepingWeight()

    @classmethod
    def extract_output(cls, mst: Any) -> float:
        return mst[0] # return just the total weight


class EdgeList(OutputType):
    """
    Needs all edges -> uses the full builder that records every edge.
    Output is sorted so results are deterministic and easy to compare.
    """

    @classmethod
    def create_builder(cls) -> MSTBuilder:
        return MSTBuilderKeepingEdges() # full: records every edge

    @classmethod
    def extract_output(cls, mst: Any) -> List[Tuple]:
        return sorted(mst[1]) # return sorted list of (u, v, weight)


# ---------------------------------------------------------------------------
# Input converters -- called automatically inside mst(), not by the user.
# Both return the same internal format: (set of nodes, adjacency dict).
# ---------------------------------------------------------------------------

def _adjacency_list_to_internal(graph: dict) -> Tuple[Set, Dict]:
    """
    Convert { node: [(neighbor, weight), ...] } -> (nodes, adj_dict).
    The input should already list both directions for each undirected edge.
    """
    nodes: Set = set()
    adj: Dict = {}
    for u, neighbors in graph.items():
        nodes.add(u)
        adj.setdefault(u, [])  # make sure every node has an entry
        for v, w in neighbors:
            nodes.add(v)
            adj.setdefault(v, [])
            adj[u].append((v, w)) # add the directed edge u -> v
    return nodes, adj


def _weight_matrix_to_internal(matrix) -> Tuple[Set, Dict]:
    """
    Convert a symmetric weight matrix -> (nodes, adj_dict).
    Nodes are integers 0..n-1.  matrix[i][j] = 0 means no edge.
    """
    n = len(matrix)
    nodes: Set = set(range(n))
    adj: Dict = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n): # upper triangle only (undirected)
            if matrix[i][j] != 0:
                w = matrix[i][j]
                adj[i].append((j, w)) # add edge in both directions
                adj[j].append((i, w))
    return nodes, adj


# ---------------------------------------------------------------------------
# Algorithms
# Each algorithm receives a builder and a graph (nodes + adj dict).
# It calls builder.add_edge() for every MST edge -- nothing else.
# ---------------------------------------------------------------------------

def kruskal(builder: MSTBuilder, nodes: Set, adj: Dict) -> Any:
    """
    Find MST using Kruskal's algorithm.
    Strategy: sort all edges by weight, then greedily add edges that
    don't form a cycle (checked with Union-Find).

    >>> b = MSTBuilderKeepingEdges()
    >>> nodes = {0, 1, 2, 3}
    >>> adj = {0: [(1,1),(2,4)], 1: [(0,1),(3,2)], 2: [(0,4),(3,3)], 3: [(1,2),(2,3)]}
    >>> result = kruskal(b, nodes, adj)
    >>> result[0]
    6.0
    >>> sorted(result[1])
    [(0, 1, 1), (1, 3, 2), (2, 3, 3)]
    """
    # --- collect unique undirected edges (each pair appears once) ---
    seen: Set = set()
    edges: List = []
    for u in adj:
        for v, w in adj[u]:
            key = (min(u, v), max(u, v)) # canonical form so (0,1) == (1,0)
            if key not in seen:
                seen.add(key)
                edges.append((w, u, v)) # store as (weight, u, v) for easy sorting
    edges.sort() # cheapest edges first

    # --- Union-Find data structure ---
    parent = {n: n for n in nodes} # each node starts as its own root
    rank = {n: 0 for n in nodes} # used to keep the tree shallow

    def find(x):
        """Return the root of x's component (with path compression)."""
        while parent[x] != x:
            parent[x] = parent[parent[x]] # skip one level (iterative compression)
            x = parent[x]
        return x

    def union(x, y) -> bool:
        """Merge x and y's components. Returns False if already in the same one."""
        px, py = find(x), find(y)
        if px == py:
            return False # adding this edge would create a cycle
        if rank[px] < rank[py]:
            px, py = py, px # always attach smaller tree under larger
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1 # only increase rank on tie
        return True

    # --- build the MST ---
    mst = builder.new_mst()
    for w, u, v in edges:
        if union(u, v): # safe to add (no cycle)
            builder.add_edge(mst, u, v, w)
    return mst


def prim(builder: MSTBuilder, nodes: Set, adj: Dict) -> Any:
    """
    Find MST using Prim's algorithm.
    Strategy: start from min(nodes), always expand via the cheapest
    edge that connects the visited set to an unvisited node.

    >>> b = MSTBuilderKeepingEdges()
    >>> nodes = {0, 1, 2, 3}
    >>> adj = {0: [(1,1),(2,4)], 1: [(0,1),(3,2)], 2: [(0,4),(3,3)], 3: [(1,2),(2,3)]}
    >>> result = prim(b, nodes, adj)
    >>> result[0]
    6.0
    >>> sorted(result[1])
    [(0, 1, 1), (1, 3, 2), (3, 2, 3)]
    """
    mst = builder.new_mst()
    if not nodes:
        return mst # nothing to do for an empty graph

    start   = min(nodes) # deterministic starting point
    visited: Set = {start}
    # heap entries: (weight, from_node, to_node)
    heap = [(w, start, v) for v, w in adj.get(start, [])]
    heapq.heapify(heap)

    while heap and len(visited) < len(nodes):
        w, u, v = heapq.heappop(heap) # pop the cheapest available edge
        if v in visited:
            continue # skip: to_node already in the tree
        visited.add(v)
        builder.add_edge(mst, u, v, w) # this edge is part of the MST
        for neighbor, weight in adj.get(v, []):
            if neighbor not in visited:
                heapq.heappush(heap, (weight, v, neighbor)) # explore new edges

    return mst


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mst(algorithm, graph, outputtype: OutputType = EdgeList) -> Any:
    """
    Find the Minimum Spanning Tree of a graph.

    :param algorithm:  kruskal or prim
    :param graph:      adjacency list (dict) or weight matrix (list of lists)
    :param outputtype: TotalWeight or EdgeList  (default: EdgeList)

    The input format is detected automatically -- no extra parameter needed.

    Example graph (4 nodes, unique MST of weight 6):

        0 --1-- 1
        |       |
        4       2
        |       |
        2 --3-- 3

    --- Adjacency list input, TotalWeight output ---
    >>> adj = {0: [(1,1),(2,4)], 1: [(0,1),(3,2)], 2: [(0,4),(3,3)], 3: [(1,2),(2,3)]}
    >>> mst(kruskal, adj, TotalWeight)
    6.0
    >>> mst(prim, adj, TotalWeight)
    6.0

    --- Adjacency list input, EdgeList output ---
    >>> mst(kruskal, adj, EdgeList)
    [(0, 1, 1), (1, 3, 2), (2, 3, 3)]
    >>> mst(prim, adj, EdgeList)
    [(0, 1, 1), (1, 3, 2), (3, 2, 3)]

    --- Weight matrix input, TotalWeight output ---
    >>> mat = [[0,1,4,0],[1,0,0,2],[4,0,0,3],[0,2,3,0]]
    >>> mst(kruskal, mat, TotalWeight)
    6.0
    >>> mst(prim, mat, TotalWeight)
    6.0

    --- Weight matrix input, EdgeList output ---
    >>> mst(kruskal, mat, EdgeList)
    [(0, 1, 1), (1, 3, 2), (2, 3, 3)]
    >>> mst(prim, mat, EdgeList)
    [(0, 1, 1), (1, 3, 2), (3, 2, 3)]
    """
    # auto-detect input type: dict = adjacency list, anything else = weight matrix
    if isinstance(graph, dict):
        nodes, adj = _adjacency_list_to_internal(graph)
    else:
        nodes, adj = _weight_matrix_to_internal(graph)

    builder = outputtype.create_builder() # OutputType picks the right builder
    result  = algorithm(builder, nodes, adj) # algorithm fills the builder
    return outputtype.extract_output(result) # OutputType extracts the final answer


if __name__ == '__main__':
    import doctest
    print(doctest.testmod(verbose=True))