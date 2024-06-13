import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)

a = ti.field(dtype=ti.f32, shape=10)
N = ti.field(dtype=ti.i32, shape=())
b = ti.field(dtype=ti.f32, shape=(10, 20))
c = ti.field(dtype=tm.vec2, shape=(2, 3))

@ti.kernel
def clear_c():
	c.fill(tm.vec2(0, 0))

clear_c()
c.fill(1)
N[None] = 100
a[0] = 1.
b[0, 2] = 12.
print(a[0], N[None], b, c)