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
surface_weight = .1


@ti.kernel
def mat_mul_vec(x: ti.template(), Qx: ti.template()):
	'''
	Calculates (grad C)^T(grad C)x and stores in Qx.
	'''
	Qx.fill(0)
	for i in range(N[None]):
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		# calculate grad density constraint * solver_x
		grad_density_x = 0.
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
					grad_density_x += tm.dot(grad_density_j, x[j])
		grad_density_x += tm.dot(grad_density_i, x[i])
		# accumulate for solver_Qx
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
					Qx[j] += grad_density_j * grad_density_x
					if pij_len < particle_radius[None] * 2. and i < j:
						grad_distance_j = distance_weight * pij / pij_len
						grad_distance_i = -grad_distance_j
						grad_distance_x = tm.dot(grad_distance_i, x[i]) + tm.dot(grad_distance_j, x[j])
						Qx[i] += grad_distance_i * grad_distance_x
						Qx[j] += grad_distance_j * grad_distance_x
		Qx[i] += grad_density_i * grad_density_x

@ti.kernel
def get_RHS(b: ti.template()):
	'''
	Calculates -(grad C)^T C and stores in b.
	'''
	b.fill(0)
	for i in range(N[None]):
		p = positions[i]
		idx, idy = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None])
		# density & distance constraint
		density_c = density_weight * (densities[i] / rest_density - 1.)
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
					if pij_len < particle_radius[None] * 2. and i < j:
						distance_c = distance_weight * (pij_len - particle_radius[None] * 2.)
						grad_distance_j = distance_weight * pij / pij_len
						grad_distance_i = -grad_distance_j
						b[i] -= grad_distance_i * distance_c
						b[j] -= grad_distance_j * distance_c
		b[i] -= grad_density_i * density_c
		# surface constraint
		pi0 = positions[local_mesh_neighbors[i, 0]] - p if local_mesh_neighbors[i, 0] >= 0 else tm.vec2(0, 0)
		pi1 = positions[local_mesh_neighbors[i, 1]] - p if local_mesh_neighbors[i, 1] >= 0 else tm.vec2(0, 0)
		pi0_len = tm.length(pi0)
		pi1_len = tm.length(pi1)
		grad_surface_0 = surface_weight * pi0 / pi0_len if local_mesh_neighbors[i, 0] >= 0 else tm.vec2(0, 0)
		grad_surface_1 = surface_weight * pi1 / pi1_len if local_mesh_neighbors[i, 1] >= 0 else tm.vec2(0, 0)
		grad_surface_i = -grad_surface_0 - grad_surface_1
		if local_mesh_neighbors[i, 0] >= 0:
			b[local_mesh_neighbors[i, 0]] -= grad_surface_0
		if local_mesh_neighbors[i, 1] >= 0:
			b[local_mesh_neighbors[i, 1]] -= grad_surface_1
		b[i] -= grad_surface_i

@ti.kernel
def dot_prod(x: ti.template(), y: ti.template()) -> float:
	'''
	Returns the dot product of x and y.
	'''
	s = 0.
	for i in range(N[None]):
		s += tm.dot(x[i], y[i])
	return s

@ti.kernel
def accumulate(a: ti.template(), scale_a: float, b: ti.template(), scale_b: float):
	'''
	Apply a = a * scale_a + b * scale_b
	'''
	for i in range(N[None]):
		a[i] = a[i] * scale_a + b[i] * scale_b

@ti.kernel
def norm(x: ti.template()) -> float:
	'''
	Returns the 2 norm of x.
	'''
	s = 0.
	for i in range(N[None]):
		s += tm.length(x[i]) ** 2
	return tm.sqrt(s)

@ti.kernel
def max_delta_position() -> float:
	res = 0.
	for i in range(N[None]):
		ti.atomic_max(res, tm.length(delta_positions[i]))
	return res

solver_b = ti.field(dtype=tm.vec2, shape=Nmax)
solver_gk = ti.field(dtype=tm.vec2, shape=Nmax)
solver_dk = ti.field(dtype=tm.vec2, shape=Nmax)
solver_Qdk = ti.field(dtype=tm.vec2, shape=Nmax)

def conjugate_direction_solve(tolerance=1e-2):
	delta_positions.fill(0)
	get_RHS(solver_b)
	mat_mul_vec(delta_positions, solver_gk)
	accumulate(solver_gk, 1., solver_b, -1.)
	if norm(solver_gk) <= tolerance:
		return 0
	solver_dk.fill(0)
	accumulate(solver_dk, 1., solver_gk, -1.)
	for iter in range(N[None] * 2):
		mat_mul_vec(solver_dk, solver_Qdk)
		dQd = dot_prod(solver_dk, solver_Qdk)
		alpha = -dot_prod(solver_gk, solver_dk) / dQd
		accumulate(delta_positions, 1., solver_dk, alpha)
		mat_mul_vec(delta_positions, solver_gk)
		accumulate(solver_gk, 1., solver_b, -1.)
		# print(f'{norm(solver_gk)} at iteration {iter + 1}')
		if norm(solver_gk) <= tolerance or max_delta_position() >= particle_radius[None] * .4:
			return iter + 1
		beta = dot_prod(solver_gk, solver_Qdk) / dQd
		accumulate(solver_dk, beta, solver_gk, -1.)
	return Nmax * 2


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
	max_iter = 700
	constraint_sos = np.zeros(max_iter + 1)
	for frame in range(1):
		advance()
		init_neighbor_searcher()
		
		acc_time, acc_iter = 0., 0
		for iter in range(max_iter):
			st_time = time.time()
			get_densities()
			get_surface_normal()
			get_local_mesh()
			acc_time += time.time() - st_time
			constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
			apply_density_constraints(density_eps)
			distance_constriant = apply_distance_constraints(distance_eps)
			surface_constraint = apply_surface_constraints(surface_eps)
			constraint_sos[iter] = (constraints ** 2).sum() + distance_constriant + surface_constraint
			if iter % 100 == 0:
				print(f'Iteration {iter}: {constraint_sos[iter]} = {(constraints ** 2).sum()} + {distance_constriant} + {surface_constraint}, time: {acc_time}, iter: {acc_iter}')
				get_np_positions(vis_p)
				get_np_surface_data(normal_p, on_surf, local_mesh)
				show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'output/particles_iteration_{iter}.png')
				acc_time, acc_iter = 0., 0
			st_time = time.time()
			cur_iter = conjugate_direction_solve()
			if cur_iter == 0:
				max_iter = iter
				break
			acc_iter += cur_iter
			# print(f'max: {max_delta_position()}\t{particle_radius[None] * 2.}')
			update_positions()
			init_neighbor_searcher()
			acc_time += time.time() - st_time
		st_time = time.time()
		get_densities()
		get_surface_normal()
		get_local_mesh()
		acc_time += time.time() - st_time
		constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
		distance_constriant = apply_distance_constraints(distance_eps)
		surface_constraint = apply_surface_constraints(surface_eps)
		constraint_sos[max_iter] = (constraints ** 2).sum() + distance_constriant + surface_constraint
		print(f'Iteration {max_iter}: {constraint_sos[max_iter]} = {(constraints ** 2).sum()} + {distance_constriant} + {surface_constraint}, time: {acc_time}, iter: {acc_iter}')
		
		get_np_positions(vis_p)
		get_np_surface_data(normal_p, on_surf, local_mesh)
		show_particles(vis_p, n=normal_p, on_surf=on_surf, local_mesh=local_mesh, save_file=f'output/particles_{frame + 1}.png')
		plt.plot(np.linspace(0, max_iter, max_iter + 1), constraint_sos[:max_iter + 1])
		plt.xlabel('iteration')
		plt.ylabel('constraint')
		plt.savefig(f'output/plot_{frame + 1}.png')
		plt.clf()
		print(f'Frame {frame + 1} written.')