# Accelerate the Position-Based Surface Tension Flow

We reimplement and accelerate the optimization process in the position-based surface tension flow solver (see paper link: https://dl.acm.org/doi/pdf/10.1145/3550454.3555476).

## How to get the results

Environment requirement: Windows 10 with cuda support.

You need to install `taichi` package first with the following command:

```
python -m pip install taichi
```

Then go to the directory of this repository.

Reproduce the droplet shape optimization experiment:

```
python .\pbstf_3d.py --dir output_3d_org-0-static --case 0 --frame 1 --iter 3000
python .\pbstf_acc_3d.py --dir output_3d-0-static --case 0 --frame 1 --iter 3000
```

Reproduce the droplet oscillating simulation:

```
python .\pbstf_3d.py --dir output_3d_org-0-dynamic --case 0 --frame 100 --iter 100
python .\pbstf_acc_3d.py --dir output_3d-0-dynamic --case 0 --frame 100 --iter 35
```

Reproduce the droplet colliding simulation:

```
python .\pbstf_3d.py --dir output_3d_org-1-dynamic --case 1 --frame 500 --iter 100
python .\pbstf_acc_3d.py --dir output_3d-1-dynamic --case 1 --frame 500 --iter 35
```

A series of `.obj` files will be generated into the correpondent directory after you run a command above, visualize it anyway you want!