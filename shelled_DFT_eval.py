import gc
import numpy as np
import cupy as cp
import cupyx.scipy.special as mspecial
import json
import time
import math

xp = cp

data = np.load("eval_checkpoint.npz", allow_pickle=True)
centers = data["centers"]
N_a = data["N_a"]
N_b = data["N_b"]
C_a = data["C_a"]
C_b = data["C_b"]
total_ao = data["total_ao"]
shells = data["shells"].tolist()
A = data["A"]

C_a = xp.asarray(C_a)
C_b = xp.asarray(C_b)
A = xp.asarray(A)



print("Initializing Grid...")
padding = 10
grid_spacing = 0.2 #0.05
minx = float(xp.min(centers[:, 0]) - padding)
maxx = float(xp.max(centers[:, 0]) + padding)
miny = float(xp.min(centers[:, 1]) - padding)
maxy = float(xp.max(centers[:, 1]) + padding)
minz = float(xp.min(centers[:, 2]) - padding)
maxz = float(xp.max(centers[:, 2]) + padding)

x_space = math.ceil((maxx-minx)/grid_spacing)+1
y_space = math.ceil((maxy-miny)/grid_spacing)+1
z_space = math.ceil((maxz-minz)/grid_spacing)+1
print("X_space: ", x_space)
print("Y_space: ", y_space)
print("Z_space: ", z_space)
gridx = xp.linspace(minx, maxx, x_space)
gridy = xp.linspace(miny, maxy, y_space)
gridz = xp.linspace(minz, maxz, z_space)
X_shape = xp.size(gridx)
Y_shape = xp.size(gridy)
Z_shape = xp.size(gridz)
gridx = xp.broadcast_to(gridx[:, None, None], (X_shape, Y_shape, Z_shape))
gridy = xp.broadcast_to(gridy[None, :, None], (X_shape, Y_shape, Z_shape))
gridz = xp.broadcast_to(gridz[None, None, :], (X_shape, Y_shape, Z_shape))
grid = xp.stack((gridx, gridy, gridz), axis=-1, dtype=xp.float32)
grid_shape = grid.shape[:-1]
grid = grid.reshape(-1, 3)
grid_len = int(grid.shape[0])


print("Evaluating Grid...")
chunk_size = int(5e5)
max_N = int(max(N_a, N_b))
print("Orbitals: ", max_N)
C_a = C_a.astype(xp.float32)[:, :max_N].T
C_b = C_b.astype(xp.float32)[:, :max_N].T


xp.save("preprocess/grid.npy", grid)
total_file = np.lib.format.open_memmap(
    "preprocess/total_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(max_N, grid_len)
)

spin_file = np.lib.format.open_memmap(
    "preprocess/spin_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(max_N, grid_len)
)
write = 0
while write != grid_len:
    write_to = min(write + chunk_size, grid_len)
    ao_vals = xp.empty((total_ao, write_to-write))
    for shell in shells:
        powers = shell["components"]
        normalization = shell["normalization"].reshape(shell["n_components"], shell["n_exponents"])

        pos_diff = grid[write:write_to] - shell["atom_pos"]
        R2 = xp.sum(pos_diff ** 2, axis=-1)
        radial = xp.exp(-shell["exponents"][:, None] * R2[None, :])
        vals = (
                pos_diff[None, :, 0] ** powers[:, None, 0]
                * pos_diff[None, :, 1] ** powers[:, None, 1]
                * pos_diff[None, :, 2] ** powers[:, None, 2]
        )
        vals = normalization[:, :, None] * vals[:, None, :] * radial[None, :, :]
        vals = xp.einsum("kp,apg->kag",shell["coefficients"], vals)
        vals = vals.reshape(shell["n_ao"], -1)
        ao_vals[shell["ao_start"]:shell["ao_stop"], :] = vals

    ao_vals = A @ ao_vals
    psi_a = C_a @ ao_vals
    psi_b = C_b @ ao_vals
    mo_density_a = xp.real(psi_a.conj() * psi_a)
    mo_density_b = xp.real(psi_b.conj() * psi_b)

    occ_a = xp.zeros(max_N)
    occ_b = xp.zeros(max_N)
    occ_a[:int(N_a)] = 1.0
    occ_b[:int(N_b)] = 1.0
    occupied_density_a = occ_a[:, None] * mo_density_a
    occupied_density_b = occ_b[:, None] * mo_density_b
    orbital_total_density = occupied_density_a + occupied_density_b
    orbital_spin_density = occupied_density_a - occupied_density_b


    total_file[:, write:write_to] = xp.asnumpy(orbital_total_density)
    spin_file[:, write:write_to] = xp.asnumpy(orbital_spin_density)
    write = write_to