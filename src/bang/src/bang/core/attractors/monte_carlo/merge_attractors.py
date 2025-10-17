from collections import defaultdict

import numpy as np


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)


def merge_attractors(data, threshold=0.15):
    print(data[9].shape)
    attractors = set()

    trajectory_len, trajectory_count, state_size = data.shape

    for trajectory in range(trajectory_count):
        histogram = defaultdict(int)
        for step in range(trajectory_len):
            histogram[tuple(data[step, trajectory])] += 1

        attractors.update([node for node in histogram if histogram[node] >= threshold * trajectory_len])
        attractors.add(tuple(data[-1, trajectory]))

    return [np.array(x) for x in attractors]

