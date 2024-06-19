import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time

from pbstf_3d import *


############## Solver ##############

density_weight = .4
distance_weight = 1.
surface_weight = .5

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
	# density constraint
	for i in range(N[None]):
		p = positions[i]
		idx, idy, idz = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None]), int((p[2] - z_min) // kernel_radius[None])
		density_c = density_weight * (densities[i] / rest_density - 1.)
		density_sos[i] = (density_c / density_weight) ** 2
		grad_density_i = tm.vec3(0, 0, 0)
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for gk in range(max(0, idz - 1), min(grid_size[2] - 1, idz + 1) + 1):
					for j_id in range(grid_cnt[gi, gj, gk]):
						j = sorted_id[grid_offset[gi, gj, gk] + j_id]
						pj = positions[j]
						pij = pj - p
						pij_len = tm.length(pij)
						if pij_len > kernel_radius[None] or j == i:
							continue
						grad_density_j = density_weight * mass[None] / rest_density * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
						grad_density_i -= grad_density_j
						b[j] -= grad_density_j * density_c
		b[i] -= grad_density_i * density_c
	# distance constraint
	for i in range(N[None]):
		p = positions[i]
		idx, idy, idz = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None]), int((p[2] - z_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for gk in range(max(0, idz - 1), min(grid_size[2] - 1, idz + 1) + 1):
					for j_id in range(grid_cnt[gi, gj, gk]):
						j = sorted_id[grid_offset[gi, gj, gk] + j_id]
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
	# surface constraint
	for i in range(N[None]):
		if on_surface[i] == 0 or n_neighbors[i] < 3:
			continue
		surface_sum[i] = 0.
		grad_i = tm.vec3(0, 0, 0)
		for j in range(n_neighbors[i]):
			t0 = i
			t1 = local_mesh_neighbors[i, j]
			t2 = local_mesh_neighbors[i, j + 1] if j < n_neighbors[i] - 1 else local_mesh_neighbors[i, 0]
			surface_sum[i] += triangle_area(t0, t1, t2)
			grad_i += surface_weight * triangle_area_gradient(t0, t1, t2)
			b[t1] -= surface_weight * triangle_area_gradient(t1, t2, t0)
			b[t2] -= surface_weight * triangle_area_gradient(t2, t0, t1)
		b[i] -= grad_i
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
	init_square_droplet(-3., 3., -3., 3., -3., 3., 30)
	print('particle number:', N[None])
	init_neighbor_searcher()
	mass[None] = 1.
	get_densities()
	mass[None] /= densities.to_numpy().max()
	
	dir_name = 'output_3d'
	os.makedirs(dir_name, exist_ok=True)
	
	get_densities()
	get_surface_normal()
	get_local_meshes()
	vis_p = np.zeros((N[None], 3))
	local_mesh = np.zeros((N[None] * N_neighbor, 3), dtype=np.int32)
	tri_cnt = get_visualization_data(vis_p, local_mesh)
	export_obj(vis_p, local_mesh[:tri_cnt, :], f'{dir_name}/particles_0.obj')
	print(f'Frame 0 written.')
	max_iter = 20
	constraint_sos = np.zeros(max_iter + 1)
	for frame in range(100):
		advance()
		init_neighbor_searcher()
		
		tot_time, acc_time = 0., 0.
		for iter in range(max_iter):
			st_time = time.time()
			get_densities()
			get_surface_normal()
			get_local_meshes()
			density_constraint, distance_constriant, surface_constraint = proximal_solve()
			ti.sync()
			acc_time += time.time() - st_time
			constraint_sos[iter] = density_constraint + distance_constriant + surface_constraint
			if iter % 100 == 0:
				print(f'Iteration {iter}: {constraint_sos[iter]} = {density_constraint} + {distance_constriant} + {surface_constraint}, time: {acc_time}')
				tot_time += acc_time
				# tri_cnt = get_visualization_data(vis_p, local_mesh)
				# export_obj(vis_p, local_mesh[:tri_cnt, :], f'{dir_name}/particles_iteration_{iter}.obj')
				acc_time = 0.
			st_time = time.time()
			update_positions()
			init_neighbor_searcher()
			ti.sync()
			acc_time += time.time() - st_time
		st_time = time.time()
		get_densities()
		get_surface_normal()
		get_local_meshes()
		ti.sync()
		acc_time += time.time() - st_time
		density_constraint, distance_constriant, surface_constraint = proximal_solve()
		constraint_sos[max_iter] = density_constraint + distance_constriant + surface_constraint
		print(f'Iteration {max_iter}: {constraint_sos[max_iter]} = {density_constraint} + {distance_constriant} + {surface_constraint}, time: {acc_time}')
		tot_time += acc_time
		
		tri_cnt = get_visualization_data(vis_p, local_mesh)
		export_obj(vis_p, local_mesh[:tri_cnt, :], f'{dir_name}/particles_{frame + 1}.obj')
		plt.plot(np.linspace(0, max_iter, max_iter + 1), constraint_sos)
		plt.xlabel('iteration')
		plt.ylabel('constraint')
		plt.savefig(f'{dir_name}/plot_{frame + 1}.png')
		plt.clf()
		print(f'Frame {frame + 1} written. Total time: {tot_time}')