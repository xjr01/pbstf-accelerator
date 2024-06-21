import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time

from delaunator_2d import cmd_args, Nmax, N_neighbor, get_local_mesh

N = ti.field(dtype=int, shape=())
mass = ti.field(dtype=float, shape=())
rest_density = 1.	# g/cm^3
x_min, x_max = -5., 5.
y_min, y_max = -5., 5.
z_min, z_max = -5., 5.
particle_radius = ti.field(dtype=float, shape=())
kernel_radius = ti.field(dtype=float, shape=())
dt = 1. / 30.

density_eps = 500.
distance_eps = 40.
surface_eps = 1.

Cmax = 500000
Ntheta, Nphi = 18, 36
unit_theta, unit_phi = tm.pi / Ntheta, 2. * tm.pi / Nphi
illuminated_threshold = 1. / 9.
positions = ti.field(dtype=tm.vec3, shape=Nmax)
velocities = ti.field(dtype=tm.vec3, shape=Nmax)
densities = ti.field(dtype=float, shape=Nmax)
normals = ti.field(dtype=tm.vec3, shape=Nmax)
blocked = ti.field(dtype=int, shape=(Nmax, Ntheta + 1, Nphi + 1))
on_surface = ti.field(dtype=ti.i8, shape=Nmax)
projected_positions = ti.field(dtype=tm.vec2, shape=(Nmax, N_neighbor))
neighbors_id = ti.field(dtype=int, shape=(Nmax, N_neighbor))
n_neighbors = ti.field(dtype=int, shape=Nmax)
local_mesh_neighbors = ti.field(dtype=int, shape=(Nmax, N_neighbor))
surface_gradient = ti.field(dtype=tm.vec3, shape=(Nmax, N_neighbor))
delta_positions = ti.field(dtype=tm.vec3, shape=Nmax)

grid_size = [int(np.ceil(np.sqrt(Nmax))), int(np.ceil(np.sqrt(Nmax))), int(np.ceil(np.sqrt(Nmax)))]
grid_offset = ti.field(dtype=int, shape=grid_size)
grid_cnt = ti.field(dtype=int, shape=grid_size)
sorted_id = ti.field(dtype=int, shape=Nmax)


############## Initializations ##############

@ti.kernel
def init_square_droplet(sq_xmin: float, sq_xmax: float, sq_ymin: float, sq_ymax: float, sq_zmin: float, sq_zmax: float, x_resolution: int):
	N[None] = 0
	spacing = (sq_xmax - sq_xmin) / x_resolution
	particle_radius[None] = .5 * spacing
	kernel_radius[None] = 3. * spacing
	y = sq_ymin
	while y <= sq_ymax:
		z = sq_zmin
		while z <= sq_zmax:
			x = sq_xmin
			while x <= sq_xmax:
				positions[N[None]] = tm.vec3(x, y, z)
				velocities[N[None]] = tm.vec3(0, 0, 0)
				N[None] += 1
				x += spacing
			z += spacing
		y += spacing

@ti.kernel
def init_droplets_colliding(x_resolution: int):
	N[None] = 0
	spacing = 2. / x_resolution
	particle_radius[None] = .5 * spacing
	kernel_radius[None] = 3. * spacing
	sq_xmin, sq_xmax = -1.5, -.5
	sq_ymin, sq_ymax = -.625, .375
	sq_zmin, sq_zmax = -.5, .5
	y = sq_ymin
	while y <= sq_ymax:
		z = sq_zmin
		while z <= sq_zmax:
			x = sq_xmin
			while x <= sq_xmax:
				positions[N[None]] = tm.vec3(x, y, z)
				velocities[N[None]] = tm.vec3(1, 0, 0)
				N[None] += 1
				x += spacing
			z += spacing
		y += spacing
	sq_xmin, sq_xmax = .5, 1.5
	sq_ymin, sq_ymax = -.375, .625
	y = sq_ymin
	while y <= sq_ymax:
		z = sq_zmin
		while z <= sq_zmax:
			x = sq_xmin
			while x <= sq_xmax:
				positions[N[None]] = tm.vec3(x, y, z)
				velocities[N[None]] = tm.vec3(-1, 0, 0)
				N[None] += 1
				x += spacing
			z += spacing
		y += spacing

@ti.kernel
def init_neighbor_searcher():
	x_size = int((x_max - x_min) // kernel_radius[None]) + 1
	y_size = int((y_max - y_min) // kernel_radius[None]) + 1
	z_size = int((z_max - z_min) // kernel_radius[None]) + 1
	for i in range(x_size):
		for j in range(y_size):
			for k in range(z_size):
				grid_cnt[i, j, k] = 0
	for i in range(N[None]):
		x, y, z = positions[i][0], positions[i][1], positions[i][2]
		idx, idy, idz = int((x - x_min) // kernel_radius[None]), int((y - y_min) // kernel_radius[None]), int((z - z_min) // kernel_radius[None])
		if 0 <= idx < x_size and 0 <= idy < y_size and 0 <= idz < z_size:
			grid_cnt[idx, idy, idz] += 1
	for _ in range(1):
		for i in range(x_size):
			for j in range(y_size):
				for k in range(z_size):
					pre_i, pre_j, pre_k = i, j, k
					pre_k -= 1
					if pre_k < 0:
						pre_k = z_size - 1
						pre_j -= 1
					if pre_j < 0:
						pre_j = y_size - 1
						pre_i -= 1
					if pre_i < 0:
						grid_offset[i, j, k] = 0
						continue
					grid_offset[i, j, k] = grid_offset[pre_i, pre_j, pre_k] + grid_cnt[pre_i, pre_j, pre_k]
	for i in range(x_size):
		for j in range(y_size):
			for k in range(z_size):
				grid_cnt[i, j, k] = 0
	for _ in range(1):
		for i in range(N[None]):
			x, y, z = positions[i][0], positions[i][1], positions[i][2]
			idx, idy, idz = int((x - x_min) // kernel_radius[None]), int((y - y_min) // kernel_radius[None]), int((z - z_min) // kernel_radius[None])
			if 0 <= idx < x_size and 0 <= idy < y_size and 0 <= idz < z_size:
				sorted_id[grid_offset[idx, idy, idz] + grid_cnt[idx, idy, idz]] = i
				grid_cnt[idx, idy, idz] += 1


############## Simulations ##############

@ti.func
def cubic_spline_kernel(r: float, h: float) -> float:
	x = r / h
	a = 1. / (tm.pi * h) * (8. / h)
	res = 0.
	if x < .5:
		res = a * (6. * x * x * (x - 1) + 1)
	elif x < 1.:
		res = a * 2. * (1 - x) ** 3
	return res

@ti.func
def cubic_spline_derivative(r: float, h: float) -> float:
	x = r / h
	a = 1. / (tm.pi * h) * (8. / h)
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
		idx, idy, idz = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None]), int((p[2] - z_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for gk in range(max(0, idz - 1), min(grid_size[2] - 1, idz + 1) + 1):
					for j_id in range(grid_cnt[gi, gj, gk]):
						j = sorted_id[grid_offset[gi, gj, gk] + j_id]
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
		normals[i] = tm.vec3(0, 0, 0)
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
						if pij_len > kernel_radius[None] or i == j:
							continue
						normals[i] += mass[None] / densities[i] * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
						block_radius = ti.min(particle_radius[None], pij_len * .5)
						dangle = tm.asin(block_radius / pij_len)
						theta = tm.acos(tm.clamp(pij[1] / pij_len, -1., 1.))
						phi = tm.atan2(pij[2], pij[0])
						st_theta = tm.min(int(tm.max(theta - dangle, 0.) // unit_theta), Ntheta - 1)
						en_theta = tm.min(int(tm.ceil(tm.min(theta + dangle, tm.pi) / unit_theta)), Ntheta)
						st_phi = tm.min(int((phi - dangle + (2. * tm.pi if phi - dangle < -tm.pi else 0.) + tm.pi) // unit_phi), Nphi - 1)
						en_phi = tm.min(int(tm.ceil((phi + dangle - (2. * tm.pi if phi + dangle > tm.pi else 0.) + tm.pi) / unit_phi)), Nphi)
						if st_phi < en_phi:
							blocked[i, st_theta, st_phi] += 1
							blocked[i, st_theta, en_phi] -= 1
							blocked[i, en_theta, st_phi] -= 1
							blocked[i, en_theta, en_phi] += 1
						else:
							blocked[i, st_theta, st_phi] += 1
							blocked[i, st_theta, Nphi] -= 1
							blocked[i, en_theta, st_phi] -= 1
							blocked[i, en_theta, Nphi] += 1
							blocked[i, st_theta, 0] += 1
							blocked[i, st_theta, en_phi] -= 1
							blocked[i, en_theta, 0] -= 1
							blocked[i, en_theta, en_phi] += 1
		illuminated_cnt, illuminated_tot = 0., 0.
		for j in range(Ntheta):
			w = tm.sin(unit_theta * (j + .5))
			for k in range(Nphi):
				if j > 0 and k > 0:
					blocked[i, j, k] += blocked[i, j - 1, k] + blocked[i, j, k - 1] - blocked[i, j - 1, k - 1]
				elif j > 0:
					blocked[i, j, k] += blocked[i, j - 1, k]
				elif k > 0:
					blocked[i, j, k] += blocked[i, j, k - 1]
				illuminated_cnt += w * (blocked[i, j, k] == 0)
				illuminated_tot += w
		on_surface[i] = ti.i8(1 if illuminated_cnt >= illuminated_threshold * illuminated_tot else 0)

@ti.kernel
def apply_density_constraints(epsilon: float):
	for i in range(N[None]):
		p = positions[i]
		denominator = epsilon
		grad_i = tm.vec3(0, 0, 0)
		idx, idy, idz = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None]), int((p[2] - z_min) // kernel_radius[None])
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
						grad_j = mass[None] / rest_density * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
						grad_i -= grad_j
						denominator += tm.length(grad_j) ** 2
		denominator += tm.length(grad_i) ** 2
		lmd = -(densities[i] / rest_density - 1.) / denominator
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
						grad_j = mass[None] / rest_density * cubic_spline_derivative(pij_len, kernel_radius[None]) * pij / pij_len
						delta_positions[j] += lmd * grad_j
		delta_positions[i] += lmd * grad_i

@ti.kernel
def apply_distance_constraints(epsilon: float) -> float:
	constraint_sos = 0.
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
							c = pij_len - particle_radius[None] * 2.
							constraint_sos += c ** 2
							grad_j = pij / pij_len
							grad_i = -grad_j
							lmd = -c / (epsilon + 2.)
							delta_positions[i] += lmd * grad_i
							delta_positions[j] += lmd * grad_j
	return constraint_sos

@ti.kernel
def get_local_meshes():
	for i in range(N[None]):
		if on_surface[i] == 0:
			continue
		p = positions[i]
		ni = tm.normalize(normals[i])
		x_axis = tm.vec3(1, 0, 0)
		if tm.length(tm.cross(x_axis, ni)) < 1e-5:
			x_axis = tm.vec3(0, 1, 0)
		x_axis = tm.normalize(tm.cross(x_axis, ni))
		y_axis = tm.normalize(tm.cross(ni, x_axis))
		n_neighbors[i] = 0
		idx, idy, idz = int((p[0] - x_min) // kernel_radius[None]), int((p[1] - y_min) // kernel_radius[None]), int((p[2] - z_min) // kernel_radius[None])
		for gi in range(max(0, idx - 1), min(grid_size[0] - 1, idx + 1) + 1):
			for gj in range(max(0, idy - 1), min(grid_size[1] - 1, idy + 1) + 1):
				for gk in range(max(0, idz - 1), min(grid_size[2] - 1, idz + 1) + 1):
					for j_id in range(grid_cnt[gi, gj, gk]):
						j = sorted_id[grid_offset[gi, gj, gk] + j_id]
						pj = positions[j]
						pij = pj - p
						pij_len = tm.length(pij)
						nj = tm.normalize(normals[j])
						if pij_len > kernel_radius[None] or i == j or on_surface[j] == 0 or\
							(tm.dot(ni, nj) <= tm.cos(tm.pi / 4.) and (tm.dot(ni - nj, pij) <= 0. or pij_len >= 4. * particle_radius[None])):
							continue
						neighbors_id[i, n_neighbors[i]] = j
						projected_pij = pij - tm.dot(pij, ni) * ni
						projected_positions[i, n_neighbors[i]] = tm.vec2(tm.dot(projected_pij, x_axis), tm.dot(projected_pij, y_axis))
						n_neighbors[i] += 1
		if n_neighbors[i] >= 3:
			get_local_mesh(i, n_neighbors, projected_positions, local_mesh_neighbors)
		for j in range(n_neighbors[i]):
			local_mesh_neighbors[i, j] = neighbors_id[i, local_mesh_neighbors[i, j]]

@ti.func
def triangle_area(a: int, b: int, c: int) -> float:
	return tm.length(tm.cross(positions[b] - positions[a], positions[c] - positions[a])) * .5

@ti.func
def triangle_area_gradient(a: int, b: int, c: int) -> tm.vec3:
	normal = tm.normalize(tm.cross(positions[b] - positions[a], positions[c] - positions[a]))
	return .5 * tm.cross(normal, positions[c] - positions[b])

@ti.kernel
def apply_surface_constraints(epsilon: float) -> float:
	constraint_sum = 0.
	surface_gradient.fill(0)
	for i in range(N[None]):
		if on_surface[i] == 0 or n_neighbors[i] < 3:
			continue
		c = 0.
		grad_i = tm.vec3(0, 0, 0)
		denominator = epsilon
		for j in range(n_neighbors[i]):
			t0 = i
			t1 = local_mesh_neighbors[i, j]
			t2 = local_mesh_neighbors[i, j + 1] if j < n_neighbors[i] - 1 else local_mesh_neighbors[i, 0]
			c += triangle_area(t0, t1, t2)
			grad_i += triangle_area_gradient(t0, t1, t2)
			surface_gradient[i, j] += triangle_area_gradient(t1, t2, t0)
			surface_gradient[i, j + 1 if j < n_neighbors[i] - 1 else 0] += triangle_area_gradient(t2, t0, t1)
		constraint_sum += c
		denominator += tm.length(grad_i) ** 2
		for j in range(n_neighbors[i]):
			denominator += tm.length(surface_gradient[i, j]) ** 2
		lmd = -c / denominator
		delta_positions[i] += lmd * grad_i
		for j in range(n_neighbors[i]):
			delta_positions[local_mesh_neighbors[i, j]] += lmd * surface_gradient[i, j]
	return constraint_sum

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
def get_visualization_data(np_positions: ti.types.ndarray(), local_mesh: ti.types.ndarray()) -> int:
	for i in range(N[None]):
		np_positions[i, 0] = positions[i][0]
		np_positions[i, 1] = positions[i][1]
		np_positions[i, 2] = positions[i][2]
	triangle_cnt = 0
	for _ in range(1):
		for i in range(N[None]):
			if not on_surface[i]:
				continue
			for j in range(n_neighbors[i]):
				local_mesh[triangle_cnt, 0] = i
				local_mesh[triangle_cnt, 1] = local_mesh_neighbors[i, j]
				local_mesh[triangle_cnt, 2] = local_mesh_neighbors[i, j + 1] if j < n_neighbors[i] - 1 else local_mesh_neighbors[i, 0]
				triangle_cnt += 1
	return triangle_cnt

def export_obj(p, local_mesh, save_file: str, particles_only=True):
	with open(save_file, 'w') as fd:
		for i in range(N[None]):
			fd.write(f'v {p[i, 0]} {p[i, 1]} {p[i, 2]}\n')
		if not particles_only:
			for i in range(local_mesh.shape[0]):
				fd.write(f'f {local_mesh[i, 0] + 1} {local_mesh[i, 1] + 1} {local_mesh[i, 2] + 1}\n')

@ti.kernel
def distance_to_perfect_ball() -> float:
	center = tm.vec3(0, 0, 0)
	cnt = 0
	for i in range(N[None]):
		if on_surface[i]:
			center += positions[i]
			cnt += 1
	center /= cnt
	avg_dist = 0.
	for i in range(N[None]):
		if on_surface[i]:
			avg_dist += tm.length(positions[i] - center)
	avg_dist /= cnt
	var_dist = 0.
	for i in range(N[None]):
		if on_surface[i]:
			var_dist += (tm.length(positions[i] - center) - avg_dist) ** 2
	var_dist /= cnt
	return var_dist

if __name__ == '__main__':
	if cmd_args.case == 0:
		init_square_droplet(-1., 1., -1., 1., -1., 1., 20)
	elif cmd_args.case == 1:
		init_droplets_colliding(35)
	else:
		raise NotImplementedError
	print('particle number:', N[None])
	init_neighbor_searcher()
	mass[None] = 1.
	get_densities()
	mass[None] /= densities.to_numpy().max()
	
	dir_name = cmd_args.dir
	os.makedirs(dir_name, exist_ok=True)
	
	get_densities()
	get_surface_normal()
	get_local_meshes()
	vis_p = np.zeros((N[None], 3))
	local_mesh = np.zeros((N[None] * N_neighbor, 3), dtype=np.int32)
	tri_cnt = get_visualization_data(vis_p, local_mesh)
	if cmd_args.frame > 1:
		export_obj(vis_p, local_mesh[:tri_cnt, :], os.path.join(dir_name, 'particles_0.obj'))
		print(f'Frame 0 written.')
	max_iter = cmd_args.iter
	constraint_sos = np.zeros(max_iter + 1)
	dist2ball = np.zeros(max_iter + 1)
	for frame in range(cmd_args.frame):
		advance()
		init_neighbor_searcher()
		
		tot_time, acc_time = 0., 0.
		for iter in range(max_iter):
			st_time = time.time()
			get_densities()
			get_surface_normal()
			get_local_meshes()
			ti.sync()
			acc_time += time.time() - st_time
			constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
			st_time = time.time()
			apply_density_constraints(density_eps)
			distance_constriant = apply_distance_constraints(distance_eps)
			surface_constraint = apply_surface_constraints(surface_eps)
			ti.sync()
			acc_time += time.time() - st_time
			constraint_sos[iter] = (constraints ** 2).sum() + distance_constriant + surface_constraint
			dist2ball[iter] = distance_to_perfect_ball()
			if iter % 100 == 0:
				print(f'Iteration {iter}: {constraint_sos[iter]} = {(constraints ** 2).sum()} + {distance_constriant} + {surface_constraint}, dist2ball: {dist2ball[iter]}, time: {acc_time}')
				tot_time += acc_time
				if cmd_args.frame == 1:
					tri_cnt = get_visualization_data(vis_p, local_mesh)
					export_obj(vis_p, local_mesh[:tri_cnt, :], os.path.join(dir_name, f'particles_iteration_{iter}.obj'))
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
		constraints = densities.to_numpy()[:N[None]] / rest_density - 1.
		distance_constriant = apply_distance_constraints(distance_eps)
		surface_constraint = apply_surface_constraints(surface_eps)
		constraint_sos[max_iter] = (constraints ** 2).sum() + distance_constriant + surface_constraint
		dist2ball[max_iter] = distance_to_perfect_ball()
		print(f'Iteration {max_iter}: {constraint_sos[max_iter]} = {(constraints ** 2).sum()} + {distance_constriant} + {surface_constraint}, dist2ball: {dist2ball[max_iter]}, time: {acc_time}')
		tot_time += acc_time
		
		tri_cnt = get_visualization_data(vis_p, local_mesh)
		export_obj(vis_p, local_mesh[:tri_cnt, :], os.path.join(dir_name, f'particles_{frame + 1}.obj' if cmd_args.frame > 1 else f'particles_iteration_{max_iter}.obj'))
		np.savez(os.path.join(dir_name, f'convergence_data_{frame + 1}.npz'), constraint_sos=constraint_sos, dist2ball=dist2ball, time=tot_time)
		print(f'Frame {frame + 1} written. Total time: {tot_time}')