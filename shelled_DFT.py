import gc
import numpy as np
import cupy as cp
import cupyx.scipy.special as mspecial
# import basis_set_exchange as bse
import json
# import ragged
# import awkward as ak
import time
import math

xp = cp

data = json.load(open("data1.json"))


def combinations(added):
    total = []
    for n in range(added + 1):
        for m in range(added + 1):
            for l in range(added + 1):
                if (l + m + n) == added:
                    small_list = [l, m, n]
                    total.append(small_list)
    return total


s_orb = [0, 0, 0]
p_orb = combinations(1)
d_orb = combinations(2)
f_orb = combinations(3)

s_orb = xp.array(s_orb)
p_orb = xp.array(p_orb)
d_orb = xp.array(d_orb)
f_orb = xp.array(f_orb)

'''
atoms = ["30", "30", "30"]
centers = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
'''
atoms = ["23"]
centers = [[0, 0, 0]]
unpaired_elec = [2]
molecular_charge = 0

unpaired_elec = xp.asarray(unpaired_elec)

exp = []
coeffs = []
cen = []
pow = []

shells = []

for i, atom in enumerate(atoms):
    atom_data = data[atom]

    for instance in atom_data:
        l = instance[0]
        exponents = instance[1]
        coefficients = instance[2]

        if len(l) == 1:
            shell = {
                "atom_idx": i,
                "atom_pos": xp.array(centers[i]),
                "angular_momentum": l[0],
                "exponents": xp.array(exponents),
                "coefficients": xp.array(coefficients),
            }
            shells.append(shell)
        elif len(l) == len(coefficients):
            for momentum, coef_row in zip(l, coefficients):
                shell = {
                    "atom_idx": i,
                    "atom_pos": xp.array(centers[i]),
                    "angular_momentum": momentum,
                    "exponents": xp.array(exponents),
                    "coefficients": xp.array([coef_row]),
                }
                shells.append(shell)
        else:
            raise ValueError("Something went wrong")

            '''
            if l == 0:
                exp.append(exponents)
                coeffs.append(coefficient)
                cen.append(centers[i])
                pow.append(s_orb)
            elif l == 1:
                for p in p_orb:
                    exp.append(exponents)
                    coeffs.append(coefficient)
                    cen.append(centers[i])
                    pow.append(p)
            elif l == 2:
                for d in d_orb:
                    exp.append(exponents)
                    coeffs.append(coefficient)
                    cen.append(centers[i])
                    pow.append(d)
            elif l == 3:
                for f in f_orb:
                    exp.append(exponents)
                    coeffs.append(coefficient)
                    cen.append(centers[i])
                    pow.append(f)
            '''


centers = xp.array(centers)
Z = xp.empty(len(atoms), xp.int32)
for i in range(len(atoms)):
    Z[i] = int(atoms[i])


def sym(matrix):
    return xp.isclose(matrix, matrix.T).all()


def double_factorial(x):
    term = xp.power(2, x / 2) * mspecial.gamma(x / 2 + 1)
    idxs = xp.where(x % 2.0 != 0.0)
    term[idxs] *= xp.sqrt(2 / xp.pi)
    return term


def normal(alpha, pow):
    term = xp.power((4 * alpha), xp.sum(pow, axis=1))
    term /= (double_factorial(2 * pow[:, 0] - 1))
    term /= (double_factorial(2 * pow[:, 1] - 1))
    term /= (double_factorial(2 * pow[:, 2] - 1))
    term = xp.sqrt(term)
    return term * xp.power((2 * alpha) / xp.pi, 0.75)


def overlap(alpha, beta, cen_a, cen_b, p1, p2):
    p = xp.add.outer(alpha, beta)
    P = xp.add.outer(alpha * cen_a, beta * cen_b) / p

    a = P - cen_a
    b = P - cen_b
    max1 = int(xp.max(p1))
    idxs_a = xp.arange(max1 + 1)
    idxs_a = xp.broadcast_to(idxs_a[None, None, :], (p.shape[0], p.shape[1], idxs_a.shape[0]))
    print(idxs_a.shape)
    print(p1.shape, "\n")
    mask_a = (idxs_a <= p1[:, None, None])
    idx1, idx2, idx3 = xp.where(mask_a)
    u_plus_a = xp.zeros(mask_a.shape)
    u_plus_a[idx1, idx2, idx3] = mspecial.binom(p1[idx1], idxs_a[idx1, idx2, idx3]) * xp.power(a[idx1, idx2],
                                                                                               p1[idx1] - idxs_a[
                                                                                                   idx1, idx2, idx3])
    max1 = int(xp.max(p2))
    idxs_b = xp.arange(max1 + 1)
    idxs_b = xp.broadcast_to(idxs_b[None, None, :], (p.shape[0], p.shape[1], idxs_b.shape[0]))
    mask_b = (idxs_b <= p2[None, :, None])
    idx1, idx2, idx3 = xp.where(mask_b)
    u_plus_b = xp.zeros(mask_b.shape)
    u_plus_b[idx1, idx2, idx3] = mspecial.binom(p2[idx2], idxs_b[idx1, idx2, idx3]) * xp.power(b[idx1, idx2],
                                                                                               p2[idx2] - idxs_b[
                                                                                                   idx1, idx2, idx3])
    #print("entered")
    p = xp.add.outer(alpha, beta)
    outer_coeff = cen_a - cen_b
    outer_coeff = xp.square(outer_coeff)
    outer_coeff = outer_coeff[None, None]*xp.outer(alpha, beta)
    outer_coeff /= -p
    outer_coeff = xp.exp(outer_coeff)

    inner_coeff = u_plus_a[:, :, :, None] * u_plus_b[:, :, None, :]
    u_matrix = idxs_a[:, :, :, None] + idxs_b[:, :, None, :]
    int_matrix = xp.zeros_like(u_matrix, dtype=xp.float64)
    idxs = xp.where(u_matrix % 2.0 == 0.0)
    int_matrix[idxs] = double_factorial(u_matrix[idxs] - 1)
    int_matrix[idxs] /= xp.power(2, u_matrix[idxs] / 2)
    int_matrix[idxs] *= xp.sqrt(xp.pi)
    int_matrix[idxs] *= xp.power(p[idxs[:2]], -0.5 * (u_matrix[idxs] + 1))
    int_matrix *= inner_coeff
    int_matrix = xp.sum(int_matrix, axis=(-1, -2))
    return int_matrix * outer_coeff


def T_raw(exp, cen, pow, prev_overlap):
    result = -2 * exp[None, :] * (2 * pow[None, :] + 1) * prev_overlap
    result += 4 * xp.square(exp[None, :]) * overlap(exp, exp, cen, cen, pow, pow + 2)
    if xp.any(pow >= 2):
        idxs = xp.where(pow >= 2)[0]
        result[:, idxs] += pow[idxs][None, :] * (pow[idxs][None, :] - 1) * overlap(exp, exp[idxs], cen, cen[idxs], pow,
                                                                                   pow[idxs] - 2)
    return result


def boys_large(m, t):
    term = mspecial.gammainc(m + 0.5, t) * mspecial.gamma(m + 0.5)
    term /= 2 * xp.power(t, m + 0.5) + 1e-40
    return term


def boys_small(m, t):
    k = xp.arange(30)
    term = xp.power(-t[:, None], k[None, :])
    term /= mspecial.gamma(k[None, :] + 1)
    term1 = 2 * m[:, None] + 2 * k[None, :] + 1
    return xp.sum(term / term1, axis=-1)


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
    q = xp.outer(exp, exp) / p
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
        store_E = xp.empty((N, i + 2))

        N_i = int(xp.sum(i_mask))
        N_j = int(xp.sum(j_mask))

        prev_E = added_E_coeffs[-1]
        prev_idxs = added_E_idxs[-1]

        super_i_prev_idx = get_idx(i_idxs, prev_idxs)
        super_j_prev_idx = get_idx(j_idxs, prev_idxs)

        for need_t in range(i + 2):
            i_term = xp.zeros(N_i)
            j_term = xp.zeros(N_j)

            if need_t <= i:
                i_term += -1 * beta[i_mask] * cen_sep[i_mask] * prev_E[super_i_prev_idx, need_t] / p[i_mask]
                j_term += alpha[j_mask] * cen_sep[j_mask] * prev_E[super_j_prev_idx, need_t] / p[j_mask]

            if need_t > 0:
                i_term += prev_E[super_i_prev_idx, need_t - 1] / (2 * p[i_mask])
                j_term += prev_E[super_j_prev_idx, need_t - 1] / (2 * p[j_mask])

            if need_t < i:
                i_term += prev_E[super_i_prev_idx, need_t + 1] * (need_t + 1)
                j_term += prev_E[super_j_prev_idx, need_t + 1] * (need_t + 1)

            store_E[super_i_idx, need_t] = i_term
            store_E[super_j_idx, need_t] = j_term

        super_idx = get_idx(idxs, prev_idxs)

        added_E_coeffs[-1] = xp.delete(added_E_coeffs[-1], super_idx, axis=0)
        added_E_idxs[-1] = xp.delete(added_E_idxs[-1], super_idx, axis=0)

        added_E_coeffs.append(store_E)
        added_E_idxs.append(idxs)

    E_coeffs = [
        [None for _ in range(s)]
        for _ in range(s)
    ]

    for e, idx in zip(added_E_coeffs, added_E_idxs):
        for e_val, idx_val in zip(e, idx):
            row = int(idx_val[0])
            col = int(idx_val[1])
            E_coeffs[row][col] = e_val

    return E_coeffs
    # return added_E_coeffs, added_E_idxs, prefactor


def calc_R(exp, cen, pow, centers):
    p = exp[:, None] + exp[None, :]
    P = ((exp[:, None] * cen)[:, None, :] + (exp[:, None] * cen)[None, :, :]) / p[..., None]

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
            R[0, 0, 0, :, :] = xp.power(-2 * p[i, j], n_arr) * boys(n_arr, boys_t)

            for x in range(x_len):
                for y in range(y_len):
                    for z in range(z_len):
                        if x == 0 and y == 0 and z == 0:
                            continue
                        if (x != 0):
                            R[x, y, z, :, :-1] = (P[i, j, 0] - centers[:, 0])[:, None] * R[x - 1, y, z, :, 1:]
                            if x > 1:
                                R[x, y, z, :, :-1] += (x - 1) * R[x - 2, y, z, :, 1:]
                        elif (y != 0):
                            R[x, y, z, :, :-1] = (P[i, j, 1] - centers[:, 1])[:, None] * R[x, y - 1, z, :, 1:]
                            if y > 1:
                                R[x, y, z, :, :-1] += (y - 1) * R[x, y - 2, z, :, 1:]
                        else:
                            R[x, y, z, :, :-1] = (P[i, j, 2] - centers[:, 2])[:, None] * R[x, y, z - 1, :, 1:]
                            if z > 1:
                                R[x, y, z, :, :-1] += (z - 1) * R[x, y, z - 2, :, 1:]

            R_row.append(R[..., 0])
        R_matrix.append(R_row)
    return R_matrix, p, P


def nuclear(Ex, Ey, Ez, R, p):
    nuclear = xp.empty((len(Ex), len(Ex), centers.shape[0]))

    for i in range(len(Ex)):
        for j in range(len(Ex[0])):
            V = Ex[i][j][:, None, None, None] * Ey[i][j][None, :, None, None] * Ez[i][j][None, None, :, None] * R[i][j]
            nuclear[i, j, :] = xp.sum(V, axis=(0, 1, 2))

    nuclear *= Z[None, None, :]
    nuclear = xp.sum(nuclear, axis=-1)
    nuclear *= -2 * xp.pi / p

    return nuclear


def nuclear_repulsion(Z, centers):
    result = 0.0
    for i in range(len(Z)):
        for j in range(len(Z)):
            if i < j:
                result += Z[i] * Z[j] / xp.linalg.norm(centers[i] - centers[j])
    return result


source = f"""
#include <math_constants.h>
#define max_conv 13
#define max_r 2048
#define max_e 7
#define max_boys 13
#define RIDX(x, y, z, n) (((x*y_len+y)*z_len+z)*n_len+n)

__device__ double taylor(int m, double T) {{
    double result = 1.0 / (2.0 * m + 1.0);
    double T_pow = 1.0;
    for (int k = 1; k < 40; ++k) {{
        T_pow *= -T/((double)k);
        result += T_pow/(2.0*m + 2.0*k + 1.0);
    }}
    return result;
}}


__device__ void boys(int max_m, double T, double* F) {{
    double exp_neg_T = exp(-T);
    if (T < 1e-14) {{
        for (int m = 0; m < max_m; ++m) {{
            F[m] = 1.0 / (2.0*m + 1.0);
        }}
    }}
    else if (T < 6) {{
        double val = taylor(max_m-1, T);
        F[max_m-1] = val;
        for (int m = max_m-2; m >= 0; --m) {{
            val = 2.0*T*val + exp_neg_T;
            val /= 2.0*m+1;
            F[m] = val;
        }}
    }}
    else {{
        double val = sqrt(CUDART_PI)*erf(sqrt(T));
        val /= 2.0*sqrt(T);
        F[0] = val;
        for (int m = 0; m < max_m-1; ++m) {{
            val = (2.0*m+1.0)*val - exp_neg_T;
            val /= 2.0*T;
            F[m+1] = val;
        }}
    }}
}}

__device__ void convolution(const double* E, int E_i, int E_j, int len_i, int len_j, double* C) {{

    int out_len = len_i + len_j - 1;
    for (int s = 0; s < out_len; ++s) {{
        C[s] = 0.0;
    }}

    for (int t = 0; t < len_i; ++t) {{
        for (int tau = 0; tau < len_j; ++tau) {{
            double sign = (tau & 1) ? -1.0 : 1.0;
            C[t+tau] += E[E_i + t]*E[E_j + tau]*sign;
        }}
    }}
}}

__device__ void calc_E(double alpha, double beta, double A, double B, int la, int lb, double* E) {{
    double prev[max_e];
    double next[max_e];

    for (int t = 0; t < max_e; ++t) {{
        prev[t] = 0.0;
        next[t] = 0.0;
        E[t] = 0.0;
    }}

    double p = alpha + beta;
    double q = alpha*beta/p;
    double Q = A - B;

    prev[0] = exp(-q*Q*Q);
    int order = 0;
    for (int step = 0; step < la; ++step) {{
        int new_order = order + 1;
        for (int t = 0; t < new_order; ++t) {{
            double value = 0.0;

            if (t <= order) {{
                value -= (beta*Q/p)*prev[t];
            }}

            if (t > 0) {{
                value += prev[t-1]/(2.0*p);
            }}

            if (t + 1 <= order) {{
                value += (t+1)*prev[t+1];
            }}
            next[t] = value;
        }}
        order = new_order;
        for (int t = 0; t <= order; ++t) {{
            prev[t] = next[t];
        }}
    }}

    for (int step = 0; step < lb; ++step) {{
        int new_order = order + 1;
        for (int t = 0; t < new_order; ++t) {{
            double value = 0.0;

            if (t <= order) {{
                value += (alpha*Q/p)*prev[t];
            }}

            if (t > 0) {{
                value += prev[t-1]/(2.0*p);
            }}

            if (t + 1 <= order) {{
                value += (t+1)*prev[t+1];
            }}
            next[t] = value;
        }}
        order = new_order;
        for (int t = 0; t <= order; ++t) {{
            prev[t] = next[t];
        }}
    }}
    for (int t = 0; t <= order; ++t) {{
        E[t] = prev[t];
    }}
}}

__device__ void calc_R(int x_len, int y_len, int z_len, int n_len, double dx, double dy, double dz, double param, double* R) {{

    double T = param*(dx*dx + dy*dy + dz*dz);
    double F[max_boys];
    boys(n_len, T, F);
    double scale = 1.0;

    for (int n = 0; n < n_len; ++n) {{
        R[RIDX(0, 0, 0, n)] = scale*F[n];
        scale *= -2.0*param;
    }}

    for (int x = 0; x < x_len; ++x) {{
        for (int y = 0; y < y_len; ++y) {{
            for (int z = 0; z < z_len; ++z) {{
                if (x == 0 && y == 0 && z == 0) {{
                    continue;
                }}
                int valid_n = n_len - (x + y + z);
                if (x != 0) {{
                    for (int n = 0; n < valid_n; ++n) {{
                        R[RIDX(x, y, z, n)] = dx * R[RIDX((x-1), y, z, (n+1))];
                    }}
                    if (x > 1) {{
                        for (int n = 0; n < valid_n; ++n) {{
                            R[RIDX(x, y, z, n)] += (x-1)*R[RIDX((x-2), y, z, (n+1))];
                        }}
                    }}
                }}
                else if (y != 0) {{
                    for (int n = 0; n < valid_n; ++n) {{
                        R[RIDX(x, y, z, n)] = dy * R[RIDX(x, (y-1), z, (n+1))];
                    }}
                    if (y > 1) {{
                        for (int n = 0; n < valid_n; ++n) {{
                            R[RIDX(x, y, z, n)] += (y-1)*R[RIDX(x, (y-2), z, (n+1))];
                        }}
                    }}
                }}
                else {{
                    for (int n = 0; n < valid_n; ++n) {{
                        R[RIDX(x, y, z, n)] = dz * R[RIDX(x, y, (z-1), (n+1))];
                    }}
                    if (z > 1) {{
                        for (int n = 0; n < valid_n; ++n) {{
                            R[RIDX(x, y, z, n)] += (z-1)*R[RIDX(x, y, (z-2), (n+1))];
                        }}
                    }}
                }}
            }}
        }}
    }}
}}

extern "C" __global__ void nuclear_kernel(
    const double* exp, 
    const double* cen, 
    const int* pow, 
    const int* atoms, 
    const double* atom_centers,
    const int num_atoms, 
    const int num_funcs, 
    double* V_output,
    double* Ex_output,
    double* Ey_output,
    double* Ez_output,
    double* p_output,
    double* P_output,
    int* nx_output,
    int* ny_output,
    int* nz_output
    ) {{

    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    long long num_pairs = (long long)num_funcs * (long long)num_funcs;

    if (q >= num_pairs) {{
        return;
    }}

    int a = q / num_funcs;
    int b = q % num_funcs;

    double Ex[max_e];
    double Ey[max_e];
    double Ez[max_e];

    double e1 = exp[a];
    double e2 = exp[b];
    double c1x = cen[3*a];
    double c2x = cen[3*b];
    double c1y = cen[3*a+1];
    double c2y = cen[3*b+1];
    double c1z = cen[3*a+2];
    double c2z = cen[3*b+2];
    int p1x = pow[3*a];
    int p2x = pow[3*b];
    int p1y = pow[3*a+1];
    int p2y = pow[3*b+1];
    int p1z = pow[3*a+2];
    int p2z = pow[3*b+2];

    calc_E(e1, e2, c1x, c2x, p1x, p2x, Ex);
    calc_E(e1, e2, c1y, c2y, p1y, p2y, Ey);
    calc_E(e1, e2, c1z, c2z, p1z, p2z, Ez);

    double p = e1 + e2;
    double Px = (e1*c1x + e2*c2x)/p;
    double Py = (e1*c1y + e2*c2y)/p;
    double Pz = (e1*c1z + e2*c2z)/p;

    int x_len = p1x + p2x + 1;
    int y_len = p1y + p2y + 1;
    int z_len = p1z + p2z + 1;
    int n_len = x_len + y_len + z_len - 2;

    double R[max_r];
    double value = 0.0;

    for (int atom = 0; atom < num_atoms; ++atom) {{
        double dx = Px - atom_centers[3*atom];
        double dy = Py - atom_centers[3*atom+1];
        double dz = Pz - atom_centers[3*atom+2];
        calc_R(x_len, y_len, z_len, n_len, dx, dy, dz, p, R);
        double contraction = 0.0;
        for (int x = 0; x < x_len; ++x) {{
            for (int y = 0; y < y_len; ++y) {{
                double exy = Ex[x] * Ey[y];
                for (int z = 0; z < z_len; ++z) {{
                    contraction += exy*Ez[z]*R[RIDX(x, y, z, 0)];
                }}
            }}
        }}
        value += atoms[atom]*contraction;
    }}
    value *= -2.0 * CUDART_PI / p;
    V_output[q] = value;
    for (int i = 0; i < max_e; ++i) {{
        Ex_output[q*max_e+i] = Ex[i];
        Ey_output[q*max_e+i] = Ey[i];
        Ez_output[q*max_e+i] = Ez[i];
    }}
    p_output[q] = p;
    P_output[3*q] = Px;
    P_output[3*q+1] = Py;
    P_output[3*q+2] = Pz;
    nx_output[q] = x_len;
    ny_output[q] = y_len;
    nz_output[q] = z_len;
}}

extern "C" __global__
void eri_kernel(
    const int row_start,
    const int K,
    const long long num_current,
    const double* p_k,
    const double* P_k,
    const double* Ex,
    const double* Ey,
    const double* Ez,
    const int* nx,
    const int* ny,
    const int* nz,
    const int sx,
    const int sy,
    const int sz,
    double* eri_output
) {{

    long long q =
    (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_current) {{
        return;
    }}

    int i = row_start + (int)(q / K);
    int j = (int)(q % K);

    double p_i = p_k[i];
    double p_j = p_k[j];

    int nxi = nx[i];
    int nxj = nx[j];
    int nyi = ny[i];
    int nyj = ny[j];
    int nzi = nz[i];
    int nzj = nz[j];

    int x_len = nxi + nxj - 1;
    int y_len = nyi + nyj - 1;
    int z_len = nzi + nzj - 1;
    int n_len = x_len + y_len + z_len - 2;

    double rho = (p_i*p_j)/(p_i+p_j);
    double r_ij0 = P_k[3*i] - P_k[3*j];
    double r_ij1 = P_k[3*i+1] - P_k[3*j+1];
    double r_ij2 = P_k[3*i+2] - P_k[3*j+2];
    double R[max_r];

    calc_R(x_len, y_len, z_len, n_len, r_ij0, r_ij1, r_ij2, rho, R);

    double Cx[max_conv];
    double Cy[max_conv];
    double Cz[max_conv];

    convolution(Ex, i*sx, j*sx, nxi, nxj, Cx);
    convolution(Ey, i*sy, j*sy, nyi, nyj, Cy);
    convolution(Ez, i*sz, j*sz, nzi, nzj, Cz);

    double integral = 0.0;
    for (int x = 0; x < x_len; ++x) {{
        for (int y = 0; y < y_len; ++y) {{
            double cxy = Cx[x]*Cy[y];
            for (int z = 0; z < z_len; ++z) {{
                integral += cxy*Cz[z]*R[RIDX(x, y, z, 0)];
            }}
        }}
    }}

    double term = 2.0 * CUDART_PI * CUDART_PI * sqrt(CUDART_PI);
    integral *= term / (p_i * p_j * sqrt(p_i + p_j));
    eri_output[q] = integral;
}}
"""


def pack_E(E):
    N = len(E)
    rows = [E[a][b] for a in range(N) for b in range(N)]
    lengths = np.asarray([row.size for row in rows], dtype=np.int32)
    stride = int(lengths.max())
    packed = xp.zeros((N * N, stride), dtype=xp.float64)
    for pair, row in enumerate(rows):
        packed[pair, :row.size] = row
    return packed.ravel(), xp.asarray(lengths, dtype=xp.int32), stride


def total_spin(unpaired_elec):
    N = int(xp.size(unpaired_elec))
    pairs = xp.stack([unpaired_elec, -unpaired_elec], axis=-1)
    total_sum = 0
    for i in range(N):
        shape = [1] * N
        shape[i] = 2
        reshaped_pair = pairs[i].reshape(shape)
        total_sum = total_sum + reshaped_pair

    return xp.min(xp.abs(total_sum)) / 2


def UHF_density(C_a, C_b, N_a, N_b):
    arr_a = xp.arange(0, N_a)
    arr_b = xp.arange(0, N_b)

    C_a_temp = C_a[:, arr_a]
    P_a = xp.sum(C_a_temp[:, None, :] * C_a_temp[None, :, :], axis=-1)

    C_b_temp = C_b[:, arr_b]
    P_b = xp.sum(C_b_temp[:, None, :] * C_b_temp[None, :, :], axis=-1)

    return P_a, P_b


def contract_2d(matrix, contracted_position, max_contr):
    cols = int(matrix.shape[1])
    temp = xp.zeros((max_contr, cols))
    xp.add.at(temp, contracted_position, matrix)
    temp = temp.T
    cols = int(temp.shape[1])
    new_matrix = xp.zeros((max_contr, cols))
    xp.add.at(new_matrix, contracted_position, temp)
    return new_matrix.T


def contract_4d(matrix, contracted_position, max_contr):
    N = int(xp.sqrt(matrix.shape[0]))
    matrix = matrix.reshape(N, N, N, N)

    temp = xp.zeros((max_contr, N, N, N))
    xp.add.at(temp, contracted_position, matrix)

    temp1 = xp.zeros((max_contr, max_contr, N, N))
    temp = xp.moveaxis(temp, 1, 0)
    xp.add.at(temp1, contracted_position, temp)
    xp.moveaxis(temp1, 0, 1)

    temp2 = xp.zeros((max_contr, max_contr, max_contr, N))
    temp1 = xp.moveaxis(temp1, 2, 0)
    xp.add.at(temp2, contracted_position, temp1)
    temp2 = xp.moveaxis(temp2, 0, 2)

    temp3 = xp.zeros((max_contr, max_contr, max_contr, max_contr))
    temp2 = xp.moveaxis(temp2, 3, 0)
    xp.add.at(temp3, contracted_position, temp2)
    temp3 = xp.moveaxis(temp3, 0, 3)

    return temp3


def contract_1d(matrix, contracted_position, max_contr):
    cols = int(matrix.shape[1])
    temp = xp.zeros((max_contr, cols), dtype=xp.float32)
    xp.add.at(temp, contracted_position, matrix)
    return temp

total_ao = 0
for shell in shells:
    shell["components"] = xp.array(combinations(shell["angular_momentum"]))
    shell["n_components"] = len(shell["components"])
    shell["n_contractions"] = shell["coefficients"].shape[0]
    shell["n_ao"] = shell["n_components"] * shell["n_contractions"]
    shell["ao_start"] = total_ao
    shell["ao_stop"] = total_ao + shell["n_ao"]
    shell["normalization"] = normal(shell["exponents"][:, None], shell["components"])
    current_ao = shell["ao_stop"]

overlaps = xp.empty((total_ao, total_ao))

for i, shell_a in enumerate(shells):
    for j in range(i+1):
        print(j)
        shell_b = shells[j]
        overlapx = overlap(shell_a["exponents"], shell_b["exponents"], shell_a["atom_pos"][0], shell_b["atom_pos"][0], shell_a["components"][:, 0], shell_b["components"][:, 0])

