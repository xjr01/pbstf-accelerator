import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
import os

ti.init(arch=ti.cuda)

N = ti.field(dtype=int, shape=())
mass = ti.field(dtype=float, shape=())
rest_density = 1.	# g/cm^3
x_min, x_max = -5., 5.
y_min, y_max = -5., 5.
particle_radius = ti.field(dtype=float, shape=())
kernel_radius = ti.field(dtype=float, shape=())

Nmax, Cmax = 10000, 50000
positions = ti.field(dtype=tm.vec2, shape=Nmax)
velocities = ti.field(dtype=tm.vec2, shape=Nmax)
densities = ti.field(dtype=float, shape=Nmax)
on_surface = ti.field(dtype=ti.i8, shape=Nmax)
normals = ti.field(dtype=tm.vec2, shape=Nmax)
delta_positions = ti.field(dtype=tm.vec2, shape=Nmax)

grid_size = [int(np.ceil(np.sqrt(Nmax))) * 5, int(np.ceil(np.sqrt(Nmax))) * 5]
grid_offset = ti.field(dtype=int, shape=grid_size)
grid_cnt = ti.field(dtype=int, shape=grid_size)
sorted_id = ti.field(dtype=int, shape=Nmax)


############## Initializations ##############

@ti.kernel
def init_square_droplet(sq_xmin: float, sq_xmax: float, sq_ymin: float, sq_ymax: float, x_resolution: int):
	N[None] = 0
	spacing = (sq_xmax - sq_xmin) / x_resolution
	particle_radius[None] = .5 * spacing
	kernel_radius[None] = 3. * spacing
	y, y_odd = sq_ymin, False
	while y <= sq_ymax:
		x = sq_xmin + y_odd * spacing * .5
		while x <= sq_xmax:
			positions[N[None]] = tm.vec2(x, y)
			velocities[N[None]] = tm.vec2(0, 0)
			N[None] += 1
			x += spacing
		y += spacing * tm.sqrt(3.) * .5
		y_odd = not y_odd

@ti.kernel
def init_neighbor_searcher():
	x_size = int((x_max - x_min) // kernel_radius[None]) + 1
	y_size = int((y_max - y_min) // kernel_radius[None]) + 1
	for i in range(x_size):
		for j in range(y_size):
			grid_cnt[i, j] = 0
	for i in range(N[None]):
		x, y = positions[i][0], positions[i][1]
		idx, idy = int((x - x_min) // kernel_radius[None]), int((y - y_min) // kernel_radius[None])
		if 0 <= idx < x_size and 0 <= idy < y_size:
			grid_cnt[idx, idy] += 1
	for _ in range(1):
		for i in range(x_size):
			for j in range(y_size):
				pre_i, pre_j = i, j
				pre_j -= 1
				if pre_j < 0:
					pre_j = y_size - 1
					pre_i -= 1
				if pre_i < 0:
					grid_offset[i, j] = 0
					continue
				grid_offset[i, j] = grid_offset[pre_i, pre_j] + grid_cnt[pre_i, pre_j]
	for i in range(x_size):
		for j in range(y_size):
			grid_cnt[i, j] = 0
	for _ in range(1):
		for i in range(N[None]):
			x, y = positions[i][0], positions[i][1]
			idx, idy = int((x - x_min) // kernel_radius[None]), int((y - y_min) // kernel_radius[None])
			if 0 <= idx < x_size and 0 <= idy < y_size:
				sorted_id[grid_offset[idx, idy] + grid_cnt[idx, idy]] = i
				grid_cnt[idx, idy] += 1


############## Simulations ##############

@ti.func
def cubic_spline_kernel(r: float, h: float) -> float:
	x = r / h
	a = 1. / (tm.pi * h) * 40. / 7.
	res = 0.
	if x < .5:
		res = a * (6. * x * x * (x - 1) + 1)
	elif x < 1.:
		res = a * 2. * (1 - x) ** 3
	return res

@ti.func
def cubic_spline_derivative(r: float, h: float) -> float:
	x = r / h
	a = 1. / (tm.pi * h) * 40. / 7.
	b = 6. / h * a
	res = 0.
	if x < .5:
		res = b * x * (3. * x - 2.)
	elif x < 1.:
		res = b * (1. - x) * (x - 1.)
	return res

@ti.kernel
def get_densities():
	for i in range(N[None]):
		densities[i] = 0.
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len > kernel_radius[None]:
						continue
					densities[i] += mass[None] * cubic_spline_kernel(pij_len, kernel_radius[None])

@ti.kernel
def apply_density_constraints(epsilon: float):
	for i in range(N[None]):
		p = positions[i]
		denominator = epsilon
		grad_i = tm.vec2(0, 0)
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len > kernel_radius[None] or j == i:
						continue
					grad_j = mass[None] / rest_density * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
					grad_i -= grad_j
					denominator += tm.length(grad_j) ** 2
		denominator += tm.length(grad_i) ** 2
		lmd = -(densities[i] / rest_density - 1.) / denominator
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len > kernel_radius[None] or j == i:
						continue
					grad_j = mass[None] / rest_density * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
					delta_positions[j] += lmd * grad_j
		delta_positions[i] += lmd * grad_i

@ti.kernel
def update_positions():
	for i in range(N[None]):
		positions[i] += delta_positions[i]
	delta_positions.fill(0)


############## Visualizations ##############

@ti.kernel
def get_np_positions(np_positions: ti.types.ndarray()):
	for i in range(N[None]):
		np_positions[i, 0] = positions[i][0]
		np_positions[i, 1] = positions[i][1]

def show_particles(p, save_file=None):
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.scatter(p[:, 0], p[:, 1], s=1.)
	ax.set_aspect('equal', adjustable='datalim')
	ax.set_xlim(x_min, x_max)
	ax.set_ylim(y_min, y_max)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
		plt.clf()
	plt.close(fig)


if __name__ == '__main__':
	init_square_droplet(-3., 3., -3., 3., 50)
	init_neighbor_searcher()
	mass[None] = 1.
	get_densities()
	mass[None] /= densities.to_numpy().max()
	
	os.makedirs('output', exist_ok=True)
	vis_p = np.zeros((N[None], 2))
	get_np_positions(vis_p)
	show_particles(vis_p, f'output/particles_0.png')
	print(f'Iteration 0 saved.')
	for iter in range(40):
		get_densities()
		constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
		print((constraints ** 2).mean(), constraints.max(), constraints.min(), densities.to_numpy()[:N[None]].min())
		apply_density_constraints(40.)
		update_positions()
		init_neighbor_searcher()
		
		get_np_positions(vis_p)
		show_particles(vis_p, f'output/particles_{iter + 1}.png')
		print(f'Iteration {iter + 1} saved.')
	get_densities()
	constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
	print((constraints ** 2).mean(), constraints.max(), constraints.min(), densities.to_numpy()[:N[None]].min())