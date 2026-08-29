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