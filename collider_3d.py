import taichi as ti
import taichi.math as tm
import numpy as np

from delaunator_2d import Nmax

@ti.data_oriented
class Collider:
	@ti.kernel
	def project(self, N: ti.template(), positions: ti.template(), t: float):
		pass

@ti.data_oriented
class BoundaryCollider(Collider):
	def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
		self.x_min, self.x_max = x_min, x_max
		self.y_min, self.y_max = y_min, y_max
		self.z_min, self.z_max = z_min, z_max
	
	@ti.kernel
	def project(self, N: ti.template(), positions: ti.template(), t: float):
		for i in range(N[None]):
			positions[i][0] = ti.min(ti.max(positions[i][0], self.x_min), self.x_max)
			positions[i][1] = ti.min(ti.max(positions[i][1], self.y_min), self.y_max)
			positions[i][2] = ti.min(ti.max(positions[i][2], self.z_min), self.z_max)

colliders = []
positions_before_project = ti.field(dtype=tm.vec3, shape=Nmax)

@ti.kernel
def copy_positions(positions: ti.template()):
	for i in positions:
		positions_before_project[i] = positions[i]

@ti.kernel
def update_velocities(N: ti.template(), positions_after_project: ti.template(), velocities: ti.template(), dt: float):
	for i in range(N[None]):
		velocities[i] += (positions_after_project[i] - positions_before_project[i]) / dt

def colliders_project(N, positions, velocities, t, dt):
	copy_positions(positions)
	for collider in colliders:
		collider.project(N, positions, t)
	update_velocities(N, positions, velocities, dt)