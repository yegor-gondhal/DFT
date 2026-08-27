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

atoms = ["11", "13", "2", "20", "17"]
centers = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

exponents_total = []
coefficients_total = []
centers_total = []
powers = []

for i, atom in enumerate(atoms):
    atom_data = data[atom]
    for instance in atom_data:
        l = instance[0]
        exponents = instance[1]
        coefficients = instance[2]

        for j in range(len(l)):
            if l[j] == 0:
                exponents_total.append(exponents)
                coefficients_total.append(coefficients[j])
                centers_total.append(centers[i])
                powers.append(s_orb)
            if l[j] == 1:
                for pow in p_orb:
                    exponents_total.append(exponents)
                    coefficients_total.append(coefficients[j])
                    centers_total.append(centers[i])
                    powers.append(pow)

exponents_total = xp.array(exponents_total)
coefficients_total = xp.array(coefficients_total)
centers_total = xp.array(centers_total)
powers = xp.array(powers)

exp_shape = exponents_total.shape
centers_total = centers_total.repeat(exp_shape[1], axis=0).reshape((exp_shape[0], exp_shape[1], 3))
powers = powers.repeat(exp_shape[1], axis=0).reshape((exp_shape[0], exp_shape[1], 3))

exponents_total = exponents_total.ravel()
coefficients_total = coefficients_total.ravel()
centers_total = centers_total.reshape((exp_shape[0]*exp_shape[1], 3))
powers = powers.reshape((exp_shape[0]*exp_shape[1], 3))

def double_factorial(x):
    term = xp.power(2, x/2)*mspecial.gamma(x/2 + 1)
    idxs = xp.where(x % 2.0 != 0.0)
    term[idxs] *= xp.sqrt(2/xp.pi)
    return term

def normal(alpha, powers):
    term = xp.power((4*alpha), xp.sum(powers, axis=1))
    term /= (double_factorial(2*powers[:, 0] - 1))
    term /= (double_factorial(2 * powers[:, 1] - 1))
    term /= (double_factorial(2 * powers[:, 2] - 1))
    term = xp.sqrt(term)
    return term * xp.power((2*alpha)/xp.pi, 0.75)

def zero_zero_overlap(alpha, beta, cen_a, cen_b):
    mu = alpha*beta/(alpha+beta)
    R2 = xp.square(cen_a - cen_b)
    return xp.sqrt((xp.pi/(alpha+beta)))*xp.exp(-mu*R2)

def one_zero_overlap(alpha, beta, cen_a, cen_b, term):
    P = (alpha*cen_a + beta*cen_b)/(alpha+beta)
    return (P-cen_a)*term

def one_one_overlap(alpha, beta, cen_a, cen_b, term):
    P = (alpha*cen_a + beta*cen_b)/(alpha+beta)
    term1 = (P-cen_a)*(P-cen_b) + 1/(2*(alpha+beta))
    return term1*term

def all_cases(p1, p2, alpha, beta, cen_a, cen_b, term):
    one_zero_idx = xp.where((p1 == 1) & (p2 == 0))
    zero_one_idx = xp.where((p1 == 0) & (p2 == 1))
    one_one_idx = xp.where((p1 == 1) & (p2 == 1))

    term[one_zero_idx] = one_zero_overlap(alpha[one_zero_idx], beta[one_zero_idx], cen_a[one_zero_idx], cen_b[one_zero_idx], term[one_zero_idx])
    term[zero_one_idx] = one_zero_overlap(beta[zero_one_idx], alpha[zero_one_idx], cen_b[zero_one_idx], cen_a[zero_one_idx], term[zero_one_idx])
    term[one_one_idx] = one_one_overlap(alpha[one_one_idx], beta[one_one_idx], cen_a[one_one_idx], cen_b[one_one_idx], term[one_one_idx])
    return term

def overlap_1d(alpha, beta, cen_a, cen_b, p1, p2):
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

    inner_coeff = u_plus_a*u_plus_b
    u_matrix = idxs_a+idxs_b
    int_matrix = xp.zeros_like(u_matrix, dtype=xp.float64)
    idx1, idx2, idx3 = xp.where(u_matrix % 2.0 == 0.0)
    int_matrix[idx1, idx2, idx3] = double_factorial(u_matrix[idx1, idx2, idx3] - 1)
    int_matrix[idx1, idx2, idx3] /= xp.power(2, u_matrix[idx1, idx2, idx3] / 2)
    int_matrix[idx1, idx2, idx3] *= xp.sqrt(xp.pi)
    int_matrix[idx1, idx2, idx3] *= xp.power(p[idx1, idx2], -0.5 * (u_matrix[idx1, idx2, idx3] + 1))
    int_matrix *= inner_coeff
    int_matrix = xp.sum(int_matrix, axis=-1)
    return int_matrix*outer_coeff

def total_overlap(alpha, beta, cen_a, cen_b, power_a, power_b):
    term1 = zero_zero_overlap(alpha, beta, cen_a[:, :, 0], cen_b[:, :, 0])
    term2 = zero_zero_overlap(alpha, beta, cen_a[:, :, 1], cen_b[:, :, 1])
    term3 = zero_zero_overlap(alpha, beta, cen_a[:, :, 2], cen_b[:, :, 2])

    term1 = all_cases(power_a[:, :, 0], power_b[:, :, 0], alpha, beta, cen_a[:, :, 0], cen_b[:, :, 0], term1)
    term2 = all_cases(power_a[:, :, 1], power_b[:, :, 1], alpha, beta, cen_a[:, :, 1], cen_b[:, :, 1], term2)
    term3 = all_cases(power_a[:, :, 2], power_b[:, :, 2], alpha, beta, cen_a[:, :, 2], cen_b[:, :, 2], term3)
    return term1*term2*term3

def total_overlap1(alpha, beta, cen_a, cen_b, power_a, power_b):
    term1 = zero_zero_overlap(alpha, beta, cen_a, cen_b)

    term1 = all_cases(power_a, power_b, alpha, beta, cen_a, cen_b, term1)
    return term1

print("Max exponent: ", xp.max(exponents_total))
print("Min exponent: ", xp.min(exponents_total))
print("Max position: ", xp.max(centers_total))
print("Min position: ", xp.min(centers_total))
print("Max power: ", xp.max(powers))
print("Min power: ", xp.min(powers), "\n")

normals = normal(exponents_total, powers)
normals = xp.outer(normals, normals)
mult_coeffs = xp.outer(coefficients_total, coefficients_total)

exp1, exp2 = xp.meshgrid(exponents_total, exponents_total)
shape = centers_total.shape
c1 = xp.broadcast_to(centers_total[None, :, :], (shape[0], shape[0], shape[1]))
c2 = xp.broadcast_to(centers_total[:, None, :], (shape[0], shape[0], shape[1]))
shape = powers.shape
p1 = xp.broadcast_to(powers[None, :, :], (shape[0], shape[0], shape[1]))
p2 = xp.broadcast_to(powers[:, None, :], (shape[0], shape[0], shape[1]))

overlap = normals*mult_coeffs*total_overlap(exp1, exp2, c1, c2, p1, p2)
overlap = overlap.reshape(exponents_total.shape[0], exp_shape[0], exp_shape[1])
overlap = xp.sum(overlap, axis=-1)
overlap = overlap.T
overlap = overlap.reshape(-1, exp_shape[0], exp_shape[1])
overlap = xp.sum(overlap, axis=-1)
overlap_exp = overlap.T

print("Condition Number: ", xp.linalg.cond(overlap_exp))
print(xp.max(xp.abs(overlap_exp - overlap_exp.T)))
eigenvalues = xp.linalg.eigvalsh(overlap_exp)
print(eigenvalues)
print(xp.diag(overlap_exp), "\n\n")

overlap1 = overlap_1d(exponents_total, exponents_total, centers_total[:, 0], centers_total[:, 0], powers[:, 0], powers[:, 0])
overlap2 = overlap_1d(exponents_total, exponents_total, centers_total[:, 1], centers_total[:, 1], powers[:, 1], powers[:, 1])
overlap3 = overlap_1d(exponents_total, exponents_total, centers_total[:, 2], centers_total[:, 2], powers[:, 2], powers[:, 2])
overlap = normals*mult_coeffs*overlap1*overlap2*overlap3
overlap = overlap.reshape(exponents_total.shape[0], exp_shape[0], exp_shape[1])
overlap = xp.sum(overlap, axis=-1)
overlap = overlap.T
overlap = overlap.reshape(-1, exp_shape[0], exp_shape[1])
overlap = xp.sum(overlap, axis=-1)
overlap_true = overlap.T

print("Condition Number: ", xp.linalg.cond(overlap_true))
print(xp.max(xp.abs(overlap_true - overlap_true.T)))
eigenvalues = xp.linalg.eigvalsh(overlap_true)
print(eigenvalues)
print(xp.diag(overlap_true), "\n")

print("Max Absolute Difference: ", xp.max(xp.abs(overlap_true - overlap_exp)))
avg = (overlap_true + overlap_exp)/2 + 1e-50
print("Max Relative Difference: ", xp.max(xp.abs((overlap_true - overlap_exp)/avg)))

