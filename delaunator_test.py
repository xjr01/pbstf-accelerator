import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time

from delaunator_2d import get_local_mesh

N = 600
x = ti.field(dtype=tm.vec2, shape=(1, N))
n_neighbor = ti.field(dtype=int, shape=1)
local_mesh_neighbors = ti.field(dtype=int, shape=(1, N))

@ti.kernel
def test_delaunator():
	for i in range(N):
		x[0, i][0] = ti.random() * 2. - 1.
		x[0, i][1] = ti.random() * 2. - 1.
	n_neighbor[0] = N
	for i in range(1):
		get_local_mesh(i, n_neighbor, x, local_mesh_neighbors)

if __name__ == '__main__':
	for test in range(5):
		st_time = time.time()
		test_delaunator()
		print('time:', time.time() - st_time)
		positions = x.to_numpy().squeeze(0)
		ring_ids = local_mesh_neighbors.to_numpy().squeeze(0)
		print(positions.shape, positions.dtype, ring_ids.shape, ring_ids.dtype)
		_, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ring_ids = np.concatenate([ring_ids, np.zeros(1, dtype=int)], axis=0)
		ring_ids[n_neighbor[0]] = ring_ids[0]
		sorted_positions = positions[ring_ids[:n_neighbor[0] + 1], :]
		ax.plot(sorted_positions[:, 0], sorted_positions[:, 1])
		ax.scatter(positions[:, 0], positions[:, 1], s=2., c='red')
		ax.scatter([0], [0], s=2., c='red')
		plt.show()