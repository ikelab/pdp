import numpy as np
import igraph


def build_trip_time_table(n, ET):
    """
    n: number of nodes
    ET = [(i, j, t), ...]: list of specification of (time t on edge from i to j)    
    """
    E, T = zip(*[((x, y), z) for x, y, z in ET])
    
    G = igraph.Graph(n, list(E), True, edge_attrs={'time': T})
    
    return np.array(G.shortest_paths(range(n), range(n), 'time'), float)


if __name__ == '__main__':
    EW = [(0, 1, 3), (1, 2, 3), (2, 0, 3)]
    t = build_trip_time_table(3, EW)
    print(t)
