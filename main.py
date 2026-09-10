import numpy as np
import cupy as cp
import cupyx.scipy.special as mspecial
import basis_set_exchange as bse
import json
import ragged
import awkward as ak
import time

xp = cp

data = json.load(open("data.json"))

def combinations(added):
    total = []
    for n in range(added+1):
        for m in range(added+1):
            for l in range(added+1):
                if (l+m+n) == added:
                    small_list = [l, m, n]
                    total.append(small_list)
    return total

s_orb = [0, 0, 0]
p_orb = combinations(1)
d_orb = combinations(2)
f_orb = combinations(3)

def gaussian(pos, center, alpha, powers):
    dx = pos[0] - center[0]
    dy = pos[1] - center[1]
    dz = pos[2] - center[2]

    r2 = dx*dx + dy*dy + dz*dz

    return (
        dx**powers[0] *
        dy**powers[1] *
        dz**powers[2] *
        xp.exp(-alpha*r2)
    )
'''
atoms = ["11", "20", "3", "16", "10", "12", "17"]
centers = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [-1, 0, 0], [0, -1, 0]]
'''
atoms = ["1", "1"]
centers = [[0, 0, 0], [1, 0, 0]]

exp = []
coeffs = []
cen = []
pow = []

for i, atom in enumerate(atoms):
    atom_data = data[atom]
    for instance in atom_data:
        l = instance[0]
        exponents = instance[1]
        coefficients = instance[2]

        for j in range(len(l)):
            if l[j] == 0:
                exp.append(exponents)
                coeffs.append(coefficients[j])
                cen.append(centers[i])
                pow.append(s_orb)
            if l[j] == 1:
                for p in p_orb:
                    exp.append(exponents)
                    coeffs.append(coefficients[j])
                    cen.append(centers[i])
                    pow.append(p)

exp = xp.array(exp)
coeffs = xp.array(coeffs)
cen = xp.array(cen)
pow = xp.array(pow)
centers = xp.array(centers)
Z = xp.empty(len(atoms))
for i in range(len(atoms)):
    Z[i] = int(atoms[i])

exp_shape = exp.shape
cen = cen.repeat(exp_shape[1], axis=0).reshape((exp_shape[0], exp_shape[1], 3))
pow = pow.repeat(exp_shape[1], axis=0).reshape((exp_shape[0], exp_shape[1], 3))

exp = exp.ravel()
coeffs = coeffs.ravel()
cen = cen.reshape((exp_shape[0]*exp_shape[1], 3))
pow = pow.reshape((exp_shape[0]*exp_shape[1], 3))

def double_factorial(x):
    term = xp.power(2, x/2)*mspecial.gamma(x/2 + 1)
    idxs = xp.where(x % 2.0 != 0.0)
    term[idxs] *= xp.sqrt(2/xp.pi)
    return term

def normal(alpha, pow):
    term = xp.power((4*alpha), xp.sum(pow, axis=1))
    term /= (double_factorial(2*pow[:, 0] - 1))
    term /= (double_factorial(2 * pow[:, 1] - 1))
    term /= (double_factorial(2 * pow[:, 2] - 1))
    term = xp.sqrt(term)
    return term * xp.power((2*alpha)/xp.pi, 0.75)

def overlap(alpha, beta, cen_a, cen_b, p1, p2):
    p = xp.add.outer(alpha, beta)
    P = xp.add.outer(alpha*cen_a, beta*cen_b)/p

    a = P - cen_a[:, None]
    b = P - cen_b[None, :]
    max1 = int(xp.max(p1))
    idxs_a = xp.arange(max1+1)
    idxs_a = xp.broadcast_to(idxs_a[None, None, :], (p.shape[0], p.shape[1], idxs_a.shape[0]))
    mask_a = (idxs_a <= p1[:, None, None])
    idx1, idx2, idx3 = xp.where(mask_a)
    u_plus_a = xp.zeros(mask_a.shape)
    u_plus_a[idx1, idx2, idx3] = mspecial.binom(p1[idx1], idxs_a[idx1, idx2, idx3]) * xp.power(a[idx1, idx2], p1[idx1] - idxs_a[idx1, idx2, idx3])
    max1 = int(xp.max(p2))
    idxs_b = xp.arange(max1 + 1)
    idxs_b = xp.broadcast_to(idxs_b[None, None, :], (p.shape[0], p.shape[1], idxs_b.shape[0]))
    mask_b = (idxs_b <= p2[None, :, None])
    idx1, idx2, idx3 = xp.where(mask_b)
    u_plus_b = xp.zeros(mask_b.shape)
    u_plus_b[idx1, idx2, idx3] = mspecial.binom(p2[idx2], idxs_b[idx1, idx2, idx3]) * xp.power(b[idx1, idx2], p2[idx2] - idxs_b[idx1, idx2, idx3])

    p = xp.add.outer(alpha, beta)
    outer_coeff = xp.subtract.outer(cen_a, cen_b, dtype=xp.float64)
    outer_coeff = xp.square(outer_coeff)
    outer_coeff *= xp.outer(alpha, beta)
    outer_coeff /= -p
    outer_coeff = xp.exp(outer_coeff)

    inner_coeff = u_plus_a[:, :, :, None]*u_plus_b[:, :, None, :]
    u_matrix = idxs_a[:, :, :, None]+idxs_b[:, :, None, :]
    int_matrix = xp.zeros_like(u_matrix, dtype=xp.float64)
    idxs = xp.where(u_matrix % 2.0 == 0.0)
    int_matrix[idxs] = double_factorial(u_matrix[idxs] - 1)
    int_matrix[idxs] /= xp.power(2, u_matrix[idxs] / 2)
    int_matrix[idxs] *= xp.sqrt(xp.pi)
    int_matrix[idxs] *= xp.power(p[idxs[:2]], -0.5 * (u_matrix[idxs] + 1))
    int_matrix *= inner_coeff
    int_matrix = xp.sum(int_matrix, axis=(-1, -2))
    return int_matrix*outer_coeff

def T_raw(exp, cen, pow, prev_overlap):
    result = -2*exp[None, :]*(2*pow[None, :] + 1)*prev_overlap
    result += 4*xp.square(exp[None, :])*overlap(exp, exp, cen, cen, pow, pow+2)
    if xp.any(pow >= 2):
        print("Entered")
        idxs = xp.where(pow >= 2)[0]
        result[:, idxs] += pow[idxs][None, :]*(pow[idxs][None, :] - 1)*overlap(exp, exp[idxs], cen, cen[idxs], pow, pow[idxs]-2)
    return result

def boys_large(m, t):
    term = mspecial.gammainc(m + 0.5, t)*mspecial.gamma(m + 0.5)
    term /= 2*xp.power(t, m + 0.5) + 1e-40
    return term

def boys_small(m, t):
    k = xp.arange(30)
    term = xp.power(-t[:, None], k[None, :])
    term /= mspecial.gamma(k[None, :] + 1)
    term1 = 2*m[:, None] + 2*k[None, :] + 1
    return xp.sum(term/term1, axis=-1)

def boys(m, t):
    result = xp.empty_like(t)
    mask = (t >= 1)
    result[mask] = boys_large(m[mask], t[mask])
    result[~mask] = boys_small(m[~mask], t[~mask])
    return result

def get_idx(arr1, arr2):
    mask1 = (arr1[:, 0][:, None] == arr2[:, 0][None, :])
    mask2 = (arr1[:, 1][:, None] == arr2[:, 1][None, :])
    mask = mask1 & mask2
    return xp.argwhere(mask)[:, 1]

def calc_E_1d(exp, cen, pow):
    alpha = xp.broadcast_to(exp[:, None], (exp.shape[0], exp.shape[0]))
    beta = xp.broadcast_to(exp[None, :], (exp.shape[0], exp.shape[0]))
    p = alpha + beta
    q = xp.outer(exp, exp)/p
    cen_sep = xp.subtract.outer(cen, cen).astype(xp.float64)

    prefactor = xp.square(cen_sep)
    prefactor *= -q
    prefactor = xp.exp(prefactor)

    s = pow.shape[0]
    pow_pairs = xp.stack((xp.broadcast_to(pow[:, None], (s, s)), xp.broadcast_to(pow[None, :], (s, s))), axis=-1)
    pow_iter_pairs = xp.zeros_like(pow_pairs)
    t_max = xp.add.outer(pow, pow)
    max_loop = int(xp.max(t_max))

    added_E_coeffs = []
    added_E_idxs = []

    base_idxs = xp.argwhere(xp.ones_like(t_max) == 1)

    added_E_coeffs.insert(0, prefactor.ravel()[:, None])
    added_E_idxs.insert(0, base_idxs)

    for i in range(max_loop):
        mask = (t_max > i)
        j_mask = mask & (pow_iter_pairs[:, :, 0] == pow_pairs[:, :, 0])
        i_mask = ~j_mask & mask

        pow_iter_pairs[i_mask, 0] += 1
        pow_iter_pairs[j_mask, 1] += 1

        idxs = xp.argwhere(mask)
        i_idxs = xp.argwhere(i_mask)
        j_idxs = xp.argwhere(j_mask)

        super_i_idx = get_idx(i_idxs, idxs)
        super_j_idx = get_idx(j_idxs, idxs)

        N = int(xp.sum(mask))
        store_E = xp.empty((N, i+2))

        N_i = int(xp.sum(i_mask))
        N_j = int(xp.sum(j_mask))

        prev_E = added_E_coeffs[-1]
        prev_idxs = added_E_idxs[-1]

        super_i_prev_idx = get_idx(i_idxs, prev_idxs)
        super_j_prev_idx = get_idx(j_idxs, prev_idxs)

        for need_t in range(i+2):
            i_term = xp.zeros(N_i)
            j_term = xp.zeros(N_j)

            if need_t <= i:
                i_term += -1*beta[i_mask]*cen_sep[i_mask]*prev_E[super_i_prev_idx, need_t]/p[i_mask]
                j_term += alpha[j_mask]*cen_sep[j_mask]*prev_E[super_j_prev_idx, need_t]/p[j_mask]

            if need_t > 0:
                i_term += prev_E[super_i_prev_idx, need_t-1]/(2*p[i_mask])
                j_term += prev_E[super_j_prev_idx, need_t-1]/(2*p[j_mask])

            if need_t < i:
                i_term += prev_E[super_i_prev_idx, need_t+1]*(need_t+1)
                j_term += prev_E[super_j_prev_idx, need_t+1]*(need_t+1)

            store_E[super_i_idx, need_t] = i_term
            store_E[super_j_idx, need_t] = j_term

        super_idx = get_idx(idxs, prev_idxs)

        added_E_coeffs[-1] = xp.delete(added_E_coeffs[-1], super_idx, axis=0)
        added_E_idxs[-1] = xp.delete(added_E_idxs[-1], super_idx, axis=0)

        added_E_coeffs.append(store_E)
        added_E_idxs.append(idxs)

    E_coeffs = []
    for i in range(s):
        E_coeffs.append([])

    for (e, idx) in zip(added_E_coeffs, added_E_idxs):
        for (e_val, idx_val) in zip(e, idx):
            E_coeffs[int(idx_val[0])].insert(int(idx_val[1]), e_val)

    return E_coeffs
    #return added_E_coeffs, added_E_idxs, prefactor

def calc_R(exp, cen, pow, centers):
    p = exp[:, None]+exp[None, :]
    P = ((exp[:, None]*cen)[:, None, :] + (exp[:, None]*cen)[None, :, :])/p[..., None]

    T = P[:, :, None, :] - centers[None, None, :, :]
    T = xp.sum(xp.square(T), axis=-1) * p[:, :, None]

    max_hermite = xp.sum(pow, axis=-1)
    max_hermite = max_hermite[:, None] + max_hermite[None, :]

    x_shape = pow[:, 0][:, None] + pow[:, 0][None, :] + 1
    y_shape = pow[:, 1][:, None] + pow[:, 1][None, :] + 1
    z_shape = pow[:, 2][:, None] + pow[:, 2][None, :] + 1

    R_matrix = []
    for i in range(len(exp)):
        R_row = []
        for j in range(len(exp)):
            x_len, y_len, z_len = int(x_shape[i, j]), int(y_shape[i, j]), int(z_shape[i, j])
            n_len = int(max_hermite[i, j]) + 1
            M = int(centers.shape[0])
            R = xp.empty((x_len, y_len, z_len, M, n_len))
            n_arr = xp.arange(n_len)
            n_arr = xp.broadcast_to(n_arr[None, :], (T.shape[2], n_len))
            boys_t = xp.broadcast_to(T[i, j, :, None], (T.shape[2], n_len))
            R[0, 0, 0, :, :] = xp.power(-2*p[i, j], n_arr)*boys(n_arr, boys_t)

            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        if (x != 0):
                            R[x, y, z, :, :-1] = (P[i, j, 0] - centers[:, 0])[:, None]*R[x-1, y, z, :, 1:]
                            if x > 1:
                                R[x, y, z, :, :-1] += (x-1)*R[x-2, y, z, :, 1:]
                        elif (y != 0):
                            R[x, y, z, :, :-1] = (P[i, j, 1] - centers[:, 1])[:, None] * R[x, y-1, z, :, 1:]
                            if y > 1:
                                R[x, y, z, :, :-1] += (y - 1) * R[x, y-2, z, :, 1:]
                        else:
                            R[x, y, z, :, :-1] = (P[i, j, 2] - centers[:, 2])[:, None] * R[x, y, z-1, :, 1:]
                            if z > 1:
                                R[x, y, z, :, :-1] += (z - 1) * R[x, y, z-2, :, 1:]

            R_row.append(R[..., 0])
        R_matrix.append(R_row)
    return R_matrix, p, P

def nuclear(Ex, Ey, Ez, R, p):
    nuclear = xp.empty((len(Ex), len(Ex), centers.shape[0]))

    for i in range(len(Ex)):
        for j in range(len(Ex[0])):
            V = Ex[i][j][:, None, None, None]*Ey[i][j][None, :, None, None]*Ez[i][j][None, None, :, None]*R[i][j]
            nuclear[i, j, :] = xp.sum(V, axis=(0, 1, 2))

    nuclear *= Z[None, None, :]
    nuclear = xp.sum(nuclear, axis=-1)
    nuclear *= -2*xp.pi/p

    return nuclear

def nuclear_repulsion(Z, centers):
    result = 0.0
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i < j:
                result += Z[i]*Z[j]/xp.linalg.norm(centers[i] - centers[j])
    return result

def calc_R_electron(pow, p, P):
    K = pow.shape[0]**2

    p_k = p.ravel()
    P_k = P.reshape(-1, 3)
    rho = p_k[:, None]*p_k[None, :]/(p_k[:, None] + p_k[None, :])
    R_ij = P_k[:, None, :] - P_k[None, :, :]
    T = xp.sum(xp.square(R_ij), axis=-1)*rho

    pow = pow[:, None, :] + pow[None, :, :]
    pow = pow.reshape(-1, 3)

    max_hermite = xp.sum(pow, axis=-1)
    max_hermite = max_hermite[:, None] + max_hermite[None, :]

    x_shape = pow[:, 0][:, None] + pow[:, 0][None, :] + 1
    y_shape = pow[:, 1][:, None] + pow[:, 1][None, :] + 1
    z_shape = pow[:, 2][:, None] + pow[:, 2][None, :] + 1

    R_matrix = []
    for i in range(K):
        print(f"{i}/{K}")
        R_row = []
        for j in range(K):
            x_len, y_len, z_len = int(x_shape[i, j]), int(y_shape[i, j]), int(z_shape[i, j])
            n_len = int(max_hermite[i, j]) + 1
            R = xp.empty((x_len, y_len, z_len, n_len))
            n_arr = xp.arange(n_len)
            boys_t = xp.broadcast_to(T[i, j][None], (n_len,))
            R[0, 0, 0, :] = xp.power(-2*rho[i, j], n_arr)*boys(n_arr, boys_t)

            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        if (x != 0):
                            R[x, y, z, :-1] = R_ij[i, j, 0]*R[x-1, y, z, 1:]
                            if x > 1:
                                R[x, y, z, :-1] += (x-1)*R[x-2, y, z, 1:]
                        elif (y != 0):
                            R[x, y, z, :-1] = R_ij[i, j, 1] * R[x, y-1, z, 1:]
                            if y > 1:
                                R[x, y, z, :-1] += (y - 1) * R[x, y-2, z, 1:]
                        else:
                            R[x, y, z, :-1] = R_ij[i, j, 2] * R[x, y, z-1, 1:]
                            if z > 1:
                                R[x, y, z, :-1] += (z - 1) * R[x, y, z-2, 1:]

            R_row.append(R[..., 0])
        R_matrix.append(R_row)
    return R_matrix, K, p_k

def elec_hermite_sum(Ex, Ey, Ez, R, K, p_k):
    S = xp.empty((K, K))
    N = int(xp.sqrt(K))
    for i in range(K):
        for j in range(K):
            a = i // N
            b = i % N
            c = j // N
            d = j % N

            Ex1 = Ex[a][b]
            Ey1 = Ey[a][b]
            Ez1 = Ez[a][b]
            Ex2 = Ex[c][d]
            Ey2 = Ey[c][d]
            Ez2 = Ez[c][d]

            term = (Ex1[:, None, None, None, None, None]
                    *Ey1[None, :, None, None, None, None]
                    *Ez1[None, None, :, None, None, None]
                    *Ex2[None, None, None, :, None, None]
                    *Ey2[None, None, None, None, :, None]
                    *Ez2[None, None, None, None, None, :])

            t = xp.arange(xp.size(Ex1))
            u = xp.arange(xp.size(Ey1))
            v = xp.arange(xp.size(Ez1))
            tau = xp.arange(xp.size(Ex2))
            phi = xp.arange(xp.size(Ey2))
            chi = xp.arange(xp.size(Ez2))

            term *= xp.power(-1, (tau[:, None, None] + phi[None, :, None] + chi[None, None, :])[None, None, None, ...])
            term *= R[i][j][
                t[:, None, None, None, None, None] + tau[None, None, None, :, None, None],
                u[None, :, None, None, None, None] + phi[None, None, None, None, :, None],
                v[None, None, :, None, None, None] + chi[None, None, None, None, None, :]
            ]
            S[i, j] = xp.sum(term)

    coeff = p_k[:, None]*p_k[None, :]*xp.sqrt(p_k[:, None] + p_k[None, :])
    coeff = 2*xp.power(xp.pi, 2.5)/coeff

    return coeff*S

print("Max exponent: ", xp.max(exp))
print("Min exponent: ", xp.min(exp))
print("Max position: ", xp.max(cen))
print("Min position: ", xp.min(cen))
print("Max power: ", xp.max(pow))
print("Min power: ", xp.min(pow), "\n")

print("Num Funcs: ", xp.size(exp), "\n")

print("Coeffs...")
normals = normal(exp, pow)
normals = xp.outer(normals, normals)
mult_coeffs = xp.outer(coeffs, coeffs)
'''
print("Overlap...")
overlapx = overlap(exp, exp, cen[:, 0], cen[:, 0], pow[:, 0], pow[:, 0])
overlapy = overlap(exp, exp, cen[:, 1], cen[:, 1], pow[:, 1], pow[:, 1])
overlapz = overlap(exp, exp, cen[:, 2], cen[:, 2], pow[:, 2], pow[:, 2])

print("T Matrix...")
T_x = T_raw(exp, cen[:, 0], pow[:, 0], overlapx)
T_y = T_raw(exp, cen[:, 1], pow[:, 1], overlapy)
T_z = T_raw(exp, cen[:, 2], pow[:, 2], overlapz)

print("Processing Overlap...")
overlaps = normals*mult_coeffs*overlapx*overlapy*overlapz
overlaps = overlaps.reshape(exp.shape[0], exp_shape[0], exp_shape[1])
overlaps = xp.sum(overlaps, axis=-1)
overlaps = overlaps.T
overlaps = overlaps.reshape(-1, exp_shape[0], exp_shape[1])
overlaps = xp.sum(overlaps, axis=-1)
overlaps = overlaps.T

print("Processing T Matrix...")
T_primitive = T_x*overlapy*overlapz + T_y*overlapx*overlapz + T_z*overlapx*overlapy
T_matrix = -0.5*normals*mult_coeffs*T_primitive
T_matrix = T_matrix.reshape(exp.shape[0], exp_shape[0], exp_shape[1])
T_matrix = xp.sum(T_matrix, axis=-1)
T_matrix = T_matrix.T
T_matrix = T_matrix.reshape(-1, exp_shape[0], exp_shape[1])
T_matrix = xp.sum(T_matrix, axis=-1)
T_matrix = T_matrix.T


print("Diagonalized Overlap: ", xp.isclose(xp.diag(overlaps), 1).all())
print("Symmetric Overlap: ", xp.isclose(overlaps, overlaps.T).all())
print("Symmetric T Matrix: ", xp.isclose(T_matrix, T_matrix.T).all())
'''
print("E Values...")
E_x_coeffs = calc_E_1d(exp, cen[:, 0], pow[:, 0])
E_y_coeffs = calc_E_1d(exp, cen[:, 1], pow[:, 1])
E_z_coeffs = calc_E_1d(exp, cen[:, 2], pow[:, 2])

R_matrix, p, P = calc_R(exp, cen, pow, centers)
'''
nuclear = normals*mult_coeffs*nuclear(E_x_coeffs, E_y_coeffs, E_z_coeffs, R_matrix, p)

nuclear = nuclear.reshape(exp.shape[0], exp_shape[0], exp_shape[1])
nuclear = xp.sum(nuclear, axis=-1)
nuclear = nuclear.T
nuclear = nuclear.reshape(-1, exp_shape[0], exp_shape[1])
nuclear = xp.sum(nuclear, axis=-1)
nuclear = nuclear.T

H = T_matrix + nuclear

print("Symmetric V Matrix: ", xp.isclose(nuclear, nuclear.T).all())
print("Symmetric H Matrix: ", xp.isclose(H, H.T).all())

E_NN = nuclear_repulsion(Z, centers)
'''
print("Electron Repulsion...")
R, K, p_k = calc_R_electron(pow, p, P)
S = elec_hermite_sum(E_x_coeffs, E_y_coeffs, E_z_coeffs, R, K, p_k)
S *= xp.outer(normals.ravel(), normals.ravel())*xp.outer(mult_coeffs.ravel(), mult_coeffs.ravel())
B = centers.shape[0]
L = xp.size(exp)//B
S = S.reshape(B, L, B, L, B, L, B, L)
S = xp.sum(S, axis=(1, 3, 5, 7))
