import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time

from pbstf_2d import *


############## Solver ##############

density_weight = 1.
distance_weight = 2.
surface_weight = .6

density_sos = ti.field(dtype=float, shape=Nmax)
distance_sos = ti.field(dtype=float, shape=Nmax)
surface_sum = ti.field(dtype=float, shape=Nmax)

@ti.kernel
def get_unscaled_proximal(b: ti.template()) -> tm.vec3:
	'''
	Calculates -(grad C)^T C - (grad C_surf)^T 1 and stores in b.
	'''
	distance_sos.fill(0)
	b.fill(0)
	for i in range(N[None]):
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		# density & distance constraint
		density_c = density_weight * (densities[i] / rest_density - 1.)
		density_sos[i] = (density_c / density_weight) ** 2
		grad_density_i = tm.vec2(0, 0)
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for j_id in range(grid_cnt[gi, gj]):
					j = sorted_id[grid_offset[gi, gj] + j_id]
					pj = positions[j]
					pij = pj - p
					pij_len = tm.length(pij)
					if pij_len > kernel_radius[None] or j == i:
						continue
					grad_density_j = density_weight * mass[None] / rest_density * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
					grad_density_i -= grad_density_j
					b[j] -= grad_density_j * density_c
		b[i] -= grad_density_i * density_c
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
						distance_c = distance_weight * (pij_len - particle_radius[None] * 2.)
						distance_sos[i] += (distance_c / distance_weight) ** 2
						grad_distance_j = distance_weight * pij / pij_len
						grad_distance_i = -grad_distance_j
						b[i] -= grad_distance_i * distance_c
						b[j] -= grad_distance_j * distance_c
	for i in range(N[None]):
		p = positions[i]
		# surface constraint
		pi0 = positions[local_mesh_neighbors[i, 0]] - p if local_mesh_neighbors[i, 0] >= 0 else tm.vec2(0, 0)
		pi1 = positions[local_mesh_neighbors[i, 1]] - p if local_mesh_neighbors[i, 1] >= 0 else tm.vec2(0, 0)
		pi0_len = tm.length(pi0)
		pi1_len = tm.length(pi1)
		surface_sum[i] = pi0_len + pi1_len
		grad_surface_0 = surface_weight * pi0 / pi0_len if local_mesh_neighbors[i, 0] >= 0 else tm.vec2(0, 0)
		grad_surface_1 = surface_weight * pi1 / pi1_len if local_mesh_neighbors[i, 1] >= 0 else tm.vec2(0, 0)
		grad_surface_i = -grad_surface_0 - grad_surface_1
		if local_mesh_neighbors[i, 0] >= 0:
			b[local_mesh_neighbors[i, 0]] -= grad_surface_0
		if local_mesh_neighbors[i, 1] >= 0:
			b[local_mesh_neighbors[i, 1]] -= grad_surface_1
		b[i] -= grad_surface_i
	c_density = 0.
	c_distance = 0.
	c_surface = 0.
	for i in range(N[None]):
		c_density += density_sos[i]
		c_distance += distance_sos[i]
		c_surface += surface_sum[i]
	return tm.vec3(c_density, c_distance, c_surface)

@ti.kernel
def scale(a: ti.template(), c: float):
	'''
	Apply a *= c
	'''
	for i in range(N[None]):
		a[i] *= c

def proximal_solve():
	constraint_packed = get_unscaled_proximal(delta_positions)
	scale(delta_positions, dt * dt / mass[None])
	return constraint_packed[0], constraint_packed[1], constraint_packed[2]


if __name__ == '__main__':
	init_square_droplet(-3., 3., -3., 3., 50)
	print('particle number:', N[None])
	init_neighbor_searcher()
	mass[None] = 1.
	get_densities()
	mass[None] /= densities.to_numpy().max()
	
	dir_name = 'output'
	os.makedirs(dir_name, exist_ok=True)
	
	get_densities()
	get_surface_normal()
	get_local_mesh()
	vis_p = np.zeros((N[None], 2))
	normal_p = np.zeros((N[None], 2))
	on_surf = np.zeros(N[None], dtype=np.int8)
	local_mesh = np.zeros((N[None], 2), dtype=np.int32)
	get_np_positions(vis_p)
	get_np_surface_data(normal_p, on_surf, local_mesh)
	show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'{dir_name}/particles_0.png')
	print(f'Frame 0 written.')
	max_iter = 8000
	constraint_sos = np.zeros(max_iter + 1)
	for frame in range(1):
		advance()
		init_neighbor_searcher()
		
		tot_time = 0.
		acc_time = 0.
		for iter in range(max_iter):
			st_time = time.time()
			get_densities()
			get_surface_normal()
			get_local_mesh()
			density_constraint, distance_constriant, surface_constraint = proximal_solve()
			constraint_sos[iter] = density_constraint + distance_constriant + surface_constraint
			ti.sync()
			acc_time += time.time() - st_time
			if iter % 1000 == 0:
				print(f'Iteration {iter}: {constraint_sos[iter]} = {density_constraint} + {distance_constriant} + {surface_constraint}, time: {acc_time}')
				tot_time += acc_time
				get_np_positions(vis_p)
				get_np_surface_data(normal_p, on_surf, local_mesh)
				show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'{dir_name}/particles_iteration_{iter}.png')
				acc_time = 0.
			st_time = time.time()
			update_positions()
			init_neighbor_searcher()
			ti.sync()
			acc_time += time.time() - st_time
		st_time = time.time()
		get_densities()
		get_surface_normal()
		get_local_mesh()
		ti.sync()
		acc_time += time.time() - st_time
		density_constraint, distance_constriant, surface_constraint = proximal_solve()
		constraint_sos[max_iter] = density_constraint + distance_constriant + surface_constraint
		print(f'Iteration {max_iter}: {constraint_sos[max_iter]} = {density_constraint} + {distance_constriant} + {surface_constraint}, time: {acc_time}')
		tot_time += acc_time
		
		get_np_positions(vis_p)
		get_np_surface_data(normal_p, on_surf, local_mesh)
		show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'{dir_name}/particles_{frame + 1}.png')
		plt.plot(np.linspace(0, max_iter, max_iter + 1), constraint_sos[:max_iter + 1])
		plt.xlabel('iteration')
		plt.ylabel('constraint')
		plt.savefig(f'{dir_name}/plot_{frame + 1}.png')
		plt.clf()
		print(f'Frame {frame + 1} written. Total time: {tot_time}')