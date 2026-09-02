import numpy as np
import cupy as cp
import cupyx.scipy.special as mspecial
import basis_set_exchange as bse
import json

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
atoms = ["1", "40"]
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

def boys(m, t):
    term = mspecial.gammainc(m + 0.5, t)*mspecial.gamma(m + 0.5)
    term /= 2*xp.power(t, m + 0.5) + 1e-40
    return term

def R000(m, t, p):
    return xp.power(-2*p, m)*boys(m, t)

def R_t_plus_1(idxs, R_matrix, P_x, C_x):
    return idxs[1]*R_matrix[idxs[0]+1, idxs[1]-1, idxs[2], idxs[3]] + (P_x - C_x)*R_matrix[idxs[0]+1, idxs[1], idxs[2], idxs[3]]
def R_u_plus_1(idxs, R_matrix, P_x, C_x):
    return idxs[2]*R_matrix[idxs[0]+1, idxs[1], idxs[2]-1, idxs[3]] + (P_x - C_x)*R_matrix[idxs[0]+1, idxs[1], idxs[2], idxs[3]]
def R_v_plus_1(idxs, R_matrix, P_x, C_x):
    return idxs[3]*R_matrix[idxs[0]+1, idxs[1], idxs[2], idxs[3]-1] + (P_x - C_x)*R_matrix[idxs[0]+1, idxs[1], idxs[2], idxs[3]]

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
    pow_pairs = xp.stack((xp.broadcast_to(pow[:, None], (s, s)), xp.broadcast_to(pow[:, None], (s, s))), axis=-1)
    pow_iter_pairs = xp.zeros_like(pow_pairs)
    t_max = xp.add.outer(pow, pow)
    max_loop = int(xp.max(t_max))

    added_E_coeffs = []
    added_E_idxs = []

    for i in range(max_loop):
        mask = (t_max > i)
        j_mask = mask & (pow_iter_pairs[:, :, 0] == pow_pairs[:, :, 0])
        i_mask = ~j_mask & mask

        idxs = xp.argwhere(mask)
        i_idxs = xp.argwhere(i_mask)
        j_idxs = xp.argwhere(j_mask)

        super_i_idx = get_idx(i_idxs, idxs)
        super_j_idx = get_idx(j_idxs, idxs)

        N = int(xp.sum(mask))
        store_E = xp.empty((N, i+2))

        if i == 0:
            store_E[super_i_idx, 0] = -1*beta[i_mask]*cen_sep[i_mask]*prefactor[i_mask]/p[i_mask]
            store_E[super_j_idx, 0] = alpha[j_mask]*cen_sep[j_mask]*prefactor[j_mask]/p[j_mask]

            store_E[super_i_idx, 1] = prefactor[i_mask]/(2*p[i_mask])
            store_E[super_j_idx, 1] = prefactor[j_mask] / (2*p[j_mask])

            added_E_coeffs.append(store_E)
            added_E_idxs.append(idxs)
        else:
            prev_E = added_E_coeffs[-1]
            prev_idxs = added_E_idxs[-1]

            super_i_prev_idx = get_idx(i_idxs, prev_idxs)
            super_j_prev_idx = get_idx(j_idxs, prev_idxs)

            for need_t in range(i+2):
                i_term = -1*beta[i_mask]*cen_sep[i_mask]*prev_E[super_i_prev_idx, need_t]/p[i_mask]
                j_term = alpha[j_mask]*cen_sep[j_mask]*prev_E[super_j_prev_idx, need_t]/p[j_mask]

                if need_t > 0:
                    i_term += prev_E[super_i_prev_idx, need_t-1]/(2*p[i_mask])
                    j_term += prev_E[super_j_prev_idx, need_t-1]/(2*p[j_mask])

                if need_t < i+1:
                    i_term += prev_E[super_i_prev_idx, need_t+1]*(need_t+1)
                    j_term += prev_E[super_j_prev_idx, need_t+1]*(need_t+1)

                store_E[super_i_idx, need_t] = i_term
                store_E[super_j_idx, need_t] = j_term

                added_E_coeffs.append(store_E)
                added_E_idxs.append(idxs)

    base_idxs = xp.argwhere(xp.ones_like(t_max) == 1)
    seen = added_E_idxs[-1]
    size = len(added_E_idxs)

    for i in range(size-2):
        super_idx = get_idx(added_E_idxs[size-i-2], seen)
        added_E_coeffs[size-i-2] = xp.delete(added_E_coeffs[size-i-2], super_idx)
        added_E_idxs[size-i-2] = xp.delete(added_E_idxs[size-i-2], super_idx)
        seen = xp.concatenate((seen, added_E_idxs[size-i-2]))

    super_idx = xp.where((base_idxs == seen).all(axis=1))[0]
    base_idx = xp.delete(base_idxs, super_idx)

    added_E_coeffs.insert(0, prefactor[base_idx[:, 0], base_idx[:, 1]])
    added_E_idxs.insert(0, base_idx)

    return added_E_coeffs, added_E_idxs


'''
print("Max exponent: ", xp.max(exp))
print("Min exponent: ", xp.min(exp))
print("Max position: ", xp.max(cen))
print("Min position: ", xp.min(cen))
print("Max power: ", xp.max(pow))
print("Min power: ", xp.min(pow), "\n")

normals = normal(exp, pow)
normals = xp.outer(normals, normals)
mult_coeffs = xp.outer(coeffs, coeffs)

overlapx = overlap(exp, exp, cen[:, 0], cen[:, 0], pow[:, 0], pow[:, 0])
overlapy = overlap(exp, exp, cen[:, 1], cen[:, 1], pow[:, 1], pow[:, 1])
overlapz = overlap(exp, exp, cen[:, 2], cen[:, 2], pow[:, 2], pow[:, 2])

T_x = T_raw(exp, cen[:, 0], pow[:, 0], overlapx)
T_y = T_raw(exp, cen[:, 1], pow[:, 1], overlapy)
T_z = T_raw(exp, cen[:, 2], pow[:, 2], overlapz)


overlaps = normals*mult_coeffs*overlapx*overlapy*overlapz
overlaps = overlaps.reshape(exp.shape[0], exp_shape[0], exp_shape[1])
overlaps = xp.sum(overlaps, axis=-1)
overlaps = overlaps.T
overlaps = overlaps.reshape(-1, exp_shape[0], exp_shape[1])
overlaps = xp.sum(overlaps, axis=-1)
overlaps = overlaps.T

T_primitive = T_x*overlapy*overlapz + T_y*overlapx*overlapz + T_z*overlapx*overlapy
T_matrix = -0.5*normals*mult_coeffs*T_primitive
T_matrix = T_matrix.reshape(exp.shape[0], exp_shape[0], exp_shape[1])
T_matrix = xp.sum(T_matrix, axis=-1)
T_matrix = T_matrix.T
T_matrix = T_matrix.reshape(-1, exp_shape[0], exp_shape[1])
T_matrix = xp.sum(T_matrix, axis=-1)
T_matrix = T_matrix.T


print("Diagonalized Overlap: ", xp.isclose(xp.diag(overlaps), 1).all())
print("Symmetric T Matrix: ", xp.isclose(T_matrix, T_matrix.T).all())
'''
E_coeffs, E_idxs = calc_E_1d(exp, cen[:, 0], pow[:, 0])