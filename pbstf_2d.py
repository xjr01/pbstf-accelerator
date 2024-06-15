import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time

ti.init(arch=ti.cuda, short_circuit_operators=True)

N = ti.field(dtype=int, shape=())
mass = ti.field(dtype=float, shape=())
rest_density = 1.	# g/cm^3
x_min, x_max = -5., 5.
y_min, y_max = -5., 5.
particle_radius = ti.field(dtype=float, shape=())
kernel_radius = ti.field(dtype=float, shape=())
dt = 1. / 30.

density_eps = 200.
distance_eps = 60.
surface_eps = 10.

Nmax, Cmax = 10000, 50000
Nangle = 360
unit_angle = 2. * tm.pi / Nangle
illuminated_threshold = 1. / 6.
positions = ti.field(dtype=tm.vec2, shape=Nmax)
velocities = ti.field(dtype=tm.vec2, shape=Nmax)
densities = ti.field(dtype=float, shape=Nmax)
normals = ti.field(dtype=tm.vec2, shape=Nmax)
blocked = ti.field(dtype=int, shape=(Nmax, Nangle + 1))
on_surface = ti.field(dtype=ti.i8, shape=Nmax)
local_mesh_neighbors = ti.field(dtype=int, shape=(Nmax, 2))
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
def get_surface_normal():
	blocked.fill(0)
	for i in range(N[None]):
		normals[i] = tm.vec2(0, 0)
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len > kernel_radius[None] or i == j:
						continue
					normals[i] += mass[None] / densities[i] * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
					block_radius = ti.min(particle_radius[None], pij_len * .5)
					angle = tm.atan2(pij[1], pij[0])
					dangle = tm.asin(block_radius / pij_len)
					st_angle = angle - dangle + (tm.pi * 2. if angle - dangle < -tm.pi else 0.)
					en_angle = angle + dangle - (tm.pi * 2. if angle + dangle > tm.pi else 0.)
					st = int((st_angle + tm.pi) // unit_angle)
					en = int(tm.ceil((en_angle + tm.pi) / unit_angle))
					if st < en:
						blocked[i, st] += 1
						blocked[i, en] -= 1
					else:
						blocked[i, st] += 1
						blocked[i, Nangle] -= 1
						blocked[i, 0] += 1
						blocked[i, en] -= 1
		illuminated_cnt = 0
		for j in range(Nangle):
			if j > 0:
				blocked[i, j] += blocked[i, j - 1]
			if blocked[i, j] == 0:
				illuminated_cnt += 1
		on_surface[i] = ti.i8(1 if illuminated_cnt >= illuminated_threshold * Nangle else 0)

@ti.kernel
def get_local_mesh():
	local_mesh_neighbors.fill(-1)
	for i in range(N[None]):
		if on_surface[i] == 0:
			continue
		tangent = tm.normalize(tm.vec2(normals[i][1], -normals[i][0]))
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len > kernel_radius[None] or i == j or on_surface[j] == 0 or tm.dot(normals[i], normals[j]) <= 0:
						continue
					if tm.dot(pij, tangent) < 0. and (local_mesh_neighbors[i, 0] == -1 or tm.dot(positions[local_mesh_neighbors[i, 0]] - p, tangent) < tm.dot(pij, tangent)):
						local_mesh_neighbors[i, 0] = j
					if tm.dot(pij, tangent) > 0. and (local_mesh_neighbors[i, 1] == -1 or tm.dot(positions[local_mesh_neighbors[i, 1]] - p, tangent) > tm.dot(pij, tangent)):
						local_mesh_neighbors[i, 1] = j

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
def apply_distance_constraints(epsilon: float) -> float:
	constraint_sos = 0.
	for i in range(N[None]):
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len < particle_radius[None] * 2. and i < j:
						c = pij_len - particle_radius[None] * 2.
						constraint_sos += c ** 2
						grad_j = pij / pij_len
						grad_i = -grad_j
						lmd = -c / (epsilon + 2.)
						delta_positions[i] += lmd * grad_i
						delta_positions[j] += lmd * grad_j
	return constraint_sos

@ti.kernel
def apply_surface_constraints(epsilon: float) -> float:
	constraint_sos = 0.
	for i in range(N[None]):
		if on_surface[i] == 0:
			continue
		p = positions[i]
		pi0 = positions[local_mesh_neighbors[i, 0]] - p
		pi1 = positions[local_mesh_neighbors[i, 1]] - p
		pi0_len = tm.length(pi0)
		pi1_len = tm.length(pi1)
		c = pi0_len + pi1_len
		constraint_sos += c ** 2
		grad_0 = pi0 / pi0_len
		grad_1 = pi1 / pi1_len
		grad_i = -grad_0 - grad_1
		lmd = -c / (epsilon + 2. + tm.length(grad_i) ** 2)
		delta_positions[i] += lmd * grad_i
		delta_positions[local_mesh_neighbors[i, 0]] += lmd * grad_0
		delta_positions[local_mesh_neighbors[i, 1]] += lmd * grad_1
	return constraint_sos

@ti.kernel
def update_positions():
	for i in range(N[None]):
		positions[i] += delta_positions[i]
		velocities[i] += delta_positions[i] / dt
	delta_positions.fill(0)

@ti.kernel
def advance():
	for i in range(N[None]):
		positions[i] += velocities[i] * dt


############## Visualizations ##############

@ti.kernel
def get_np_positions(np_positions: ti.types.ndarray()):
	for i in range(N[None]):
		np_positions[i, 0] = positions[i][0]
		np_positions[i, 1] = positions[i][1]

@ti.kernel
def get_np_surface_data(np_normals: ti.types.ndarray(), np_on_surface: ti.types.ndarray(), np_local_mesh: ti.types.ndarray()):
	for i in range(N[None]):
		np_normals[i, 0] = normals[i][0]
		np_normals[i, 1] = normals[i][1]
		np_on_surface[i] = on_surface[i]
		np_local_mesh[i, 0] = local_mesh_neighbors[i, 0]
		np_local_mesh[i, 1] = local_mesh_neighbors[i, 1]

def show_particles(p, n=None, on_surf=None, local_mesh=None, save_file=None):
	fig, ax = plt.subplots(figsize=(5, 5))
	ax.scatter(p[:, 0], p[:, 1], s=1., c=on_surf.astype(float), cmap='jet')
	if n is not None:
		normal_scale = ((n ** 2).sum(axis=-1) ** .5).max()
		n = n[on_surf.astype(bool)] * kernel_radius[None] * 2. / normal_scale
		drawn_normals = np.stack([p[on_surf.astype(bool)], p[on_surf.astype(bool)] + n], axis=1)
		lc = LineCollection(drawn_normals, color='blue', linewidth=.5)
		ax.add_collection(lc)
	if local_mesh is not None:
		has_meshes_0 = local_mesh[:, 0] != -1
		drawn_meshes_0 = np.stack([p[has_meshes_0, :], p[local_mesh[has_meshes_0, 0], :]], axis=1)
		has_meshes_1 = local_mesh[:, 1] != -1
		drawn_meshes_1 = np.stack([p[has_meshes_1, :], p[local_mesh[has_meshes_1, 1], :]], axis=1)
		lc0 = LineCollection(drawn_meshes_0, color='green', linewidth=.5)
		lc1 = LineCollection(drawn_meshes_1, color='green', linewidth=.5)
		ax.add_collection(lc0)
		ax.add_collection(lc1)
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
	print('particle number:', N[None])
	init_neighbor_searcher()
	mass[None] = 1.
	get_densities()
	mass[None] /= densities.to_numpy().max()
	
	os.makedirs('output', exist_ok=True)
	
	get_densities()
	get_surface_normal()
	get_local_mesh()
	vis_p = np.zeros((N[None], 2))
	normal_p = np.zeros((N[None], 2))
	on_surf = np.zeros(N[None], dtype=np.int8)
	local_mesh = np.zeros((N[None], 2), dtype=np.int32)
	get_np_positions(vis_p)
	get_np_surface_data(normal_p, on_surf, local_mesh)
	show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'output/particles_0.png')
	print(f'Frame 0 written.')
	max_iter = 10000
	constraint_sos = np.zeros(max_iter + 1)
	for frame in range(1):
		advance()
		init_neighbor_searcher()
		
		st_time = time.time()
		for iter in range(max_iter):
			get_densities()
			get_surface_normal()
			get_local_mesh()
			constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
			apply_density_constraints(density_eps)
			distance_constriant = apply_distance_constraints(distance_eps)
			surface_constraint = apply_surface_constraints(surface_eps)
			constraint_sos[iter] = (constraints ** 2).sum() + distance_constriant + surface_constraint
			if iter % 1000 == 0:
				print(f'Iteration {iter}: {constraint_sos[iter]}, time: {time.time() - st_time}')
				st_time = time.time()
			update_positions()
			init_neighbor_searcher()
		get_densities()
		get_surface_normal()
		get_local_mesh()
		constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
		distance_constriant = apply_distance_constraints(distance_eps)
		surface_constraint = apply_surface_constraints(surface_eps)
		constraint_sos[max_iter] = (constraints ** 2).sum() + distance_constriant + surface_constraint
		print(f'Iteration {max_iter}: {constraint_sos[max_iter]}, time: {time.time() - st_time}')
		
		get_np_positions(vis_p)
		get_np_surface_data(normal_p, on_surf, local_mesh)
		show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'output/particles_{frame + 1}.png')
		plt.plot(np.linspace(0, max_iter, max_iter + 1), constraint_sos)
		plt.xlabel('iteration')
		plt.ylabel('constraint')
		plt.savefig(f'output/plot_{frame + 1}.png')
		print(f'Frame {frame + 1} written.')