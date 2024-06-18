import taichi as ti
import taichi.math as tm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import time

ti.init(arch=ti.cuda, random_seed=42)

Nmax = 100000
N_neighbor = 800

chain_pre = ti.field(dtype=int, shape=(Nmax, N_neighbor))
chain_nxt = ti.field(dtype=int, shape=(Nmax, N_neighbor))
node_queue = ti.field(dtype=int, shape=(Nmax, N_neighbor * 3))

@ti.func
def angle_of_vec2(a: tm.vec2, b: tm.vec2) -> float:
	res = 0.
	a_len = tm.length(a)
	b_len = tm.length(b)
	if a_len > 0. and b_len > 0.:
		res = tm.acos(tm.clamp(tm.dot(a, b) / (a_len * b_len), -1., 1.))
	return res

@ti.func
def need_flip(cur_id: int, u: int, x: ti.template()) -> ti.i8:
	res = 0
	if angle_of_vec2(-x[cur_id, chain_pre[cur_id, u]], x[cur_id, u] - x[cur_id, chain_pre[cur_id, u]])\
		+ angle_of_vec2(-x[cur_id, chain_nxt[cur_id, u]], x[cur_id, u] - x[cur_id, chain_nxt[cur_id, u]]) > tm.pi:
		res = 1
	return res

@ti.func
def get_local_mesh(cur_id: int, n_neighbor: ti.template(), x: ti.template(), ring_ids: ti.template()):
	for i in range(n_neighbor[cur_id]):
		ring_ids[cur_id, i] = i
	# Sort by polar angle
	while True:
		has_swaped = 0
		for i in range(n_neighbor[cur_id] - 1):
			u = ring_ids[cur_id, i]
			u_nxt = ring_ids[cur_id, i + 1]
			angle = tm.atan2(x[cur_id, u][1], x[cur_id, u][0])
			angle_nxt = tm.atan2(x[cur_id, u_nxt][1], x[cur_id, u_nxt][0])
			if angle > angle_nxt or (angle == angle_nxt and tm.length(x[cur_id, u]) > tm.length(x[cur_id, u_nxt])):
				tmp = ring_ids[cur_id, i + 1]
				ring_ids[cur_id, i + 1] = ring_ids[cur_id, i]
				ring_ids[cur_id, i] = tmp
				has_swaped = 1
		if has_swaped == 0:
			break
	# Build chain
	for i in range(n_neighbor[cur_id] - 1):
		chain_nxt[cur_id, ring_ids[cur_id, i]] = ring_ids[cur_id, i + 1]
		chain_pre[cur_id, ring_ids[cur_id, i + 1]] = ring_ids[cur_id, i]
	chain_nxt[cur_id, ring_ids[cur_id, n_neighbor[cur_id] - 1]] = ring_ids[cur_id, 0]
	chain_pre[cur_id, ring_ids[cur_id, 0]] = ring_ids[cur_id, n_neighbor[cur_id] - 1]
	# BFS for edge flip
	q_st = 0
	q_en = 0
	for i in range(n_neighbor[cur_id]):
		if need_flip(cur_id, ring_ids[cur_id, i], x):
			node_queue[cur_id, q_en] = ring_ids[cur_id, i]
			q_en += 1
	while q_st < q_en:
		u = node_queue[cur_id, q_st]
		q_st += 1
		if chain_nxt[cur_id, u] == -1 or not need_flip(cur_id, u, x):
			continue
		chain_nxt[cur_id, chain_pre[cur_id, u]] = chain_nxt[cur_id, u]
		chain_pre[cur_id, chain_nxt[cur_id, u]] = chain_pre[cur_id, u]
		if need_flip(cur_id, chain_nxt[cur_id, u], x):
			node_queue[cur_id, q_en] = chain_nxt[cur_id, u]
			q_en += 1
		if need_flip(cur_id, chain_pre[cur_id, u], x):
			node_queue[cur_id, q_en] = chain_pre[cur_id, u]
			q_en += 1
		chain_nxt[cur_id, u] = chain_pre[cur_id, u] = -1
	# Get final ring and stores in ring_ids
	u = -1
	for _ in range(1):
		for i in range(n_neighbor[cur_id]):
			if chain_nxt[cur_id, i] != -1:
				u = i
				break
	n_neighbor[cur_id] = 0
	while True:
		ring_ids[cur_id, n_neighbor[cur_id]] = u
		n_neighbor[cur_id] += 1
		u = chain_nxt[cur_id, u]
		if u == ring_ids[cur_id, 0]:
			break