import gc
import numpy as np
import cupy as cp
import cupyx.scipy.special as mspecial
#import basis_set_exchange as bse
import json
#import ragged
#import awkward as ak
import time
import math

xp = cp

data = json.load(open("data1.json"))

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
            momenta = [l[0]] * len(coefficients)
        elif len(l) == len(coefficients):
            momenta = l
        else:
            raise ValueError("Something went wrong")

        for l, coefficient in zip(momenta, coefficients):
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


cen1 = []
pow1 = []
contracted_position = []
for i, e in enumerate(exp):
    tempcen = []
    temppow = []
    tempcontr = []
    for _ in range(len(e)):
        tempcen.append(cen[i])
        temppow.append(pow[i])
        tempcontr.append(i)
    cen1.append(tempcen)
    pow1.append(temppow)
    contracted_position.append(tempcontr)

exp = [j for i in exp for j in i]
coeffs = [j for i in coeffs for j in i]
cen = [j for i in cen1 for j in i]
pow = [j for i in pow1 for j in i]
contracted_position = [j for i in contracted_position for j in i]

exp = xp.array(exp)
coeffs = xp.array(coeffs)
cen = xp.array(cen)
pow = xp.array(pow, dtype=xp.int32)
contracted_position = xp.array(contracted_position)

mask = (coeffs != 0.0)
exp = exp[mask]
coeffs = coeffs[mask]
cen = cen[mask]
pow = pow[mask]
contracted_position = contracted_position[mask]

max_contr = int(xp.max(contracted_position) + 1)
centers = xp.array(centers)
Z = xp.empty(len(atoms), xp.int32)
for i in range(len(atoms)):
    Z[i] = int(atoms[i])

def sym(matrix):
    return xp.isclose(matrix, matrix.T).all()

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
    print(idxs_a.shape)
    print(p1.shape)
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
    print(int_matrix.shape, "\n")
    return int_matrix*outer_coeff

def T_raw(exp, cen, pow, prev_overlap):
    result = -2*exp[None, :]*(2*pow[None, :] + 1)*prev_overlap
    result += 4*xp.square(exp[None, :])*overlap(exp, exp, cen, cen, pow, pow+2)
    if xp.any(pow >= 2):
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
    packed = xp.zeros((N*N, stride), dtype=xp.float64)
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

    return xp.min(xp.abs(total_sum))/2

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

print("Max exponent: ", xp.max(exp))
print("Min exponent: ", xp.min(exp))
print("Max position: ", xp.max(cen))
print("Min position: ", xp.min(cen))
print("Max power: ", xp.max(pow))
print("Min power: ", xp.min(pow), "\n")

print("Num Funcs: ", xp.size(exp), "\n")

print("Coeffs...")
normals_1d = normal(exp, pow)
normals = xp.outer(normals_1d, normals_1d)
mult_coeffs = xp.outer(coeffs, coeffs)

print("Overlap...")
overlapx = overlap(exp, exp, cen[:, 0], cen[:, 0], pow[:, 0], pow[:, 0])
overlapy = overlap(exp, exp, cen[:, 1], cen[:, 1], pow[:, 1], pow[:, 1])
overlapz = overlap(exp, exp, cen[:, 2], cen[:, 2], pow[:, 2], pow[:, 2])

print("T Matrix...")
T_x = T_raw(exp, cen[:, 0], pow[:, 0], overlapx)
T_y = T_raw(exp, cen[:, 1], pow[:, 1], overlapy)
T_z = T_raw(exp, cen[:, 2], pow[:, 2], overlapz)

print("Processing Overlap...")
uncontracted_overlaps = normals*mult_coeffs*overlapx*overlapy*overlapz
overlaps = contract_2d(uncontracted_overlaps, contracted_position, max_contr)

print("Processing T Matrix...")
uncontracted_T_primitive = T_x*overlapy*overlapz + T_y*overlapx*overlapz + T_z*overlapx*overlapy
uncontracted_T_matrix = -0.5*normals*mult_coeffs*uncontracted_T_primitive
T_matrix = contract_2d(uncontracted_T_matrix, contracted_position, max_contr)

print("Setting Up Kernel...")
N = int(xp.size(exp))
K = N**2
module = cp.RawModule(
    code=source,
    name_expressions=(
        "nuclear_kernel",
        "eri_kernel",
    ),
)
nuclear_kernel = module.get_function("nuclear_kernel")
eri_kernel = module.get_function("eri_kernel")
cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitStackSize,32768)

max_e = 7
Ex = xp.empty((K*max_e))
Ey = xp.empty((K*max_e))
Ez = xp.empty((K*max_e))
nx = xp.empty(K, xp.int32)
ny = xp.empty(K, xp.int32)
nz = xp.empty(K, xp.int32)
p_k = xp.empty(K)
P_k = xp.empty(K*3)
uncontracted_V_matrix = xp.empty(K)

print("Nuclear Kernel...")
threads = 128
blocks = (K + threads - 1) // threads
nuclear_kernel((blocks,), (threads,), (
    exp,
    cen,
    pow,
    Z,
    centers,
    xp.int32(xp.size(Z)),
    xp.int32(N),
    uncontracted_V_matrix,
    Ex,
    Ey,
    Ez,
    p_k,
    P_k,
    nx,
    ny,
    nz
))
cp.cuda.runtime.deviceSynchronize()
uncontracted_V_matrix = uncontracted_V_matrix.reshape((N, N))
V_matrix = normals*mult_coeffs*uncontracted_V_matrix
V_matrix = contract_2d(V_matrix, contracted_position, max_contr)

eri_values = xp.empty((K, K), dtype=xp.float64)
chunk_size = 256
write = 0
normals_K = normals.ravel()
print("ERI Kernel...")
while write != K:
    write_to = min(write + chunk_size, K)
    blocks = ((write_to-write) * K + threads - 1) // threads
    eri_output_chunk = eri_values[write:write_to, :].ravel()
    eri_kernel((blocks,), (threads,), (
        write,
        K,
        (write_to-write)*K,
        p_k,
        P_k,
        Ex,
        Ey,
        Ez,
        nx,
        ny,
        nz,
        max_e,
        max_e,
        max_e,
        eri_output_chunk
    ))
    eri_output_chunk = eri_output_chunk.reshape(write_to-write, K)
    eri_output_chunk *= normals_K[write:write_to][:, None]*normals_K[None, :]
    write = write_to

cp.cuda.runtime.deviceSynchronize()
print("Processing ERI...")
ERI = contract_4d(eri_values, contracted_position, max_contr)

H_matrix = T_matrix + V_matrix

E_NN = nuclear_repulsion(Z, centers)
elec_count = xp.sum(Z) - molecular_charge

print("Setting Up SCF...")
eig_vals, U = xp.linalg.eigh(overlaps)
s = xp.power(eig_vals, -0.5)
s = xp.diag(s)
U_t = U.T
X = U @ s @ U_t
X_t = X.T

F = T_matrix + V_matrix
F_prime = X_t @ F @ X
orb_energy, C_prime = xp.linalg.eigh(F_prime)
C = X @ C_prime
C_a, C_b = C, C

total_spin = total_spin(unpaired_elec)
mult = 2*total_spin + 1
print(mult)
N_a = (elec_count + mult - 1)/2
N_b = (elec_count - mult + 1)/2
N_e = N_a + N_b

print("SCF...")
Fock_a = 0
Fock_b = 0
P = 0
J_matrix = 0
K_a = 0
K_b = 0
orb_energy_a = 0
orb_energy_b = 0
P_a, P_b = UHF_density(C_a, C_b, N_a, N_b)
E_total = -1000
count = 0
while True:
    P = P_a + P_b
    J_matrix = xp.sum(P[None, None, :, :] * ERI, axis=(-1, -2))
    K_a = xp.sum(P_a[None, :, None, :] * ERI, axis=(1, 3))
    K_b = xp.sum(P_b[None, :, None, :] * ERI, axis=(1, 3))
    Fock_a = H_matrix + J_matrix - K_a
    Fock_b = H_matrix + J_matrix - K_b

    E_elec = 0.5*xp.sum(P*H_matrix + P_a*Fock_a + P_b*Fock_b)
    E_total_new = E_elec + E_NN
    delta_E = xp.abs(E_total_new - E_total)
    Fock_a_prime = X_t @ Fock_a @ X
    Fock_b_prime = X_t @ Fock_b @ X
    orb_energy_a, C_a_prime = xp.linalg.eigh(Fock_a_prime)
    orb_energy_b, C_b_prime = xp.linalg.eigh(Fock_b_prime)

    C_a_new = X @ C_a_prime
    C_b_new = X @ C_b_prime
    P_a_new, P_b_new = UHF_density(C_a_new, C_b_new, N_a, N_b)
    delta_P = xp.max(xp.maximum(xp.abs(P_a_new - P_a), xp.abs(P_b_new - P_b)))

    if (delta_E < 1e-8 and delta_P < 1e-6) or (count > 100):
        P = P_a + P_b
        J_matrix = xp.sum(P[None, None, :, :] * ERI, axis=(-1, -2))
        K_a = xp.sum(P_a[None, :, None, :] * ERI, axis=(1, 3))
        K_b = xp.sum(P_b[None, :, None, :] * ERI, axis=(1, 3))
        Fock_a = H_matrix + J_matrix - K_a
        Fock_b = H_matrix + J_matrix - K_b

        E_elec = 0.5 * xp.sum(P * H_matrix + P_a * Fock_a + P_b * Fock_b)
        E_total_new = E_elec + E_NN
        delta_E = xp.abs(E_total_new - E_total)
        Fock_a_prime = X_t @ Fock_a @ X
        Fock_b_prime = X_t @ Fock_b @ X
        orb_energy_a, C_a_prime = xp.linalg.eigh(Fock_a_prime)
        orb_energy_b, C_b_prime = xp.linalg.eigh(Fock_b_prime)
        C_a = C_a_new
        C_b = C_b_new
        break

    E_total = E_total_new
    P_a = P_a_new
    P_b = P_b_new
    count += 1



print("Initializing Grid...")
padding = 3
grid_spacing = 0.05
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

print("Clearing VRAM...")
keep = ["grid", "grid_len", "coeffs", "normals_1d", "cen", "pow", "exp", "P_a", "P_b", "C_a", "xp", "np", "contract_1d", "contracted_position", "max_contr"]
for name in list(globals().keys()):
    if not name.startswith('_') and name not in keep and name != 'cp' and name != 'gc':
        del globals()[name]
gc.collect()
cp.get_default_memory_pool().free_all_blocks()

print("Evaluating Grid...")
chunk_size = int(5e5)
write = 0
coeffs = coeffs.astype(xp.float32)
normals_1d = normals_1d.astype(xp.float32)
cen = cen.astype(xp.float32)
pow = pow.astype(xp.float32)
C_a = C_a.astype(xp.float32)


xp.save("preprocess/grid.npy", grid)
total_file = np.lib.format.open_memmap(
    "preprocess/total_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(grid_len,)
)

spin_file = np.lib.format.open_memmap(
    "preprocess/spin_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(grid_len,)
)

number_of_orbitals = C_a.shape[1]
print("Number of Orbitals: ", number_of_orbitals)

orbital_file = np.lib.format.open_memmap(
    "preprocess/orbital_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(number_of_orbitals, grid_len)
)
C_a = C_a.T
while write != grid_len:
    write_to = min(write+chunk_size, grid_len)
    uncontracted_chunk = (coeffs[:, None]
                      *normals_1d[:, None]
                      *xp.power(grid[write:write_to, 0][None, :] - cen[:, 0][:, None], pow[:, 0][:, None])
                      *xp.power(grid[write:write_to, 1][None, :] - cen[:, 1][:, None], pow[:, 1][:, None])
                      *xp.power(grid[write:write_to, 2][None, :] - cen[:, 2][:, None], pow[:, 2][:, None])
                      *xp.exp(-exp[:, None] * xp.sum(xp.square(grid[None, write:write_to, :] - cen[:, None, :]), axis=-1))
    )
    chunk = contract_1d(uncontracted_chunk, contracted_position, max_contr)
    alpha_chunk = xp.sum(P_a[:, :, None]*chunk[:, None, :]*chunk[None, :, :], axis=(0, 1))
    beta_chunk = xp.sum(P_b[:, :, None] * chunk[:, None, :] * chunk[None, :, :], axis=(0, 1))
    psi_chunk = C_a @ chunk
    orbital_chunk = xp.square(psi_chunk)
    total_chunk = alpha_chunk + beta_chunk
    spin_chunk = alpha_chunk - beta_chunk
    total_file[write:write_to] = xp.asnumpy(total_chunk)
    spin_file[write:write_to] = xp.asnumpy(spin_chunk)
    orbital_file[:, write:write_to] = xp.asnumpy(orbital_chunk)
    write = write_to




'''
grid /= 5
density_max = xp.max(total_density)
tau = 1e-6
density_min = tau * density_max
density_mask = (total_density > density_min)
total_density = total_density[density_mask]
q = xp.log10(total_density)
q_min = xp.log10(density_min)
q_max = xp.log10(density_max)
density_norm = xp.clip((q - q_min) / (q_max - q_min), 0, 1)
#density_norm = xp.asnumpy(density_norm)

#density_rgba = colormaps["Blues"](density_norm)
#density_rgb = density_rgba[:, :3]
density_rgb = xp.round(density_norm * 255)
density_rgb = xp.repeat(density_rgb[:, None], 3, axis=1)
density_rgb = density_rgb.astype(xp.uint8)
density_data = xp.column_stack((grid[density_mask], density_rgb))

spin_max = xp.max(xp.abs(spin_density))
spin_tau = 1e-4
spin_min = spin_tau * spin_max
spin_mask = (xp.abs(spin_density) > spin_min)
spin_density = spin_density[spin_mask]
spin_norm = 0.5*(spin_density/spin_max + 1)
#spin_norm = xp.asnumpy(spin_norm)
#spin_cmap = colors.LinearSegmentedColormap.from_list("spin_density",["blue", "white", "red"])
#spin_rgb = spin_cmap(spin_norm)
#spin_rgb = spin_rgb[:, :3]
spin_rgb = xp.round(spin_norm * 255)
spin_rgb = xp.repeat(spin_rgb[:, None], 3, axis=1)
spin_rgb = spin_rgb.astype(np.uint8)
spin_data = np.column_stack((grid[spin_mask], spin_rgb))

np.savetxt("data/density_data.xyz", density_data, fmt=["%.7f", "%.7f", "%.7f", "%d", "%d", "%d"], delimiter=" ")
np.savetxt("data/spin_data.xyz", spin_data, fmt=["%.7f", "%.7f", "%.7f", "%d", "%d", "%d"], delimiter=" ")
'''
'''
#Complete checks:
print("\n")
print("Diagonalized Overlap: ", xp.isclose(xp.diag(overlaps), 1).all())
print("Overlap Symmetry: ", sym(overlaps))
print("T Matrix Symmetry: ", sym(T_matrix))
print("T Matrix Positive Semidefinite: ", (xp.linalg.eigh(T_matrix)[0] > -1e-10).all())
print("V Matrix Symmetry: ", sym(V_matrix))
print("V Matrix Negative Semidefinite: ", (xp.linalg.eigh(V_matrix)[0] < 1e-10).all())
print("H Matrix Symmetry: ", sym(H_matrix))
print("Nuclear Repulsion Nonnegative: ", E_NN >= 0)
print("ERI Symmetry 1: ", xp.isclose(ERI, xp.transpose(ERI, axes=(1, 0, 2, 3))).all())
print("ERI Symmetry 2: ", xp.isclose(ERI, xp.transpose(ERI, axes=(0, 1, 3, 2))).all())
print("ERI Symmetry 3: ", xp.isclose(ERI, xp.transpose(ERI, axes=(2, 3, 0, 1))).all())
eri_size = ERI.shape[0]
idx = xp.arange(eri_size)
mu = xp.broadcast_to(idx[:, None], (eri_size, eri_size))
nu = xp.broadcast_to(idx[None, :], (eri_size, eri_size))
self_coulomb = ERI[mu, nu, mu, nu]
print("Self Coulomb Nonnegative :", (self_coulomb >= 0).all())
print("Schwartz Inequality: ", (xp.square(ERI) <= self_coulomb[:, :, None, None]*self_coulomb[None, None, :, :]).all())
print("s Positive: ", (xp.diagonal(s) > 0).all())
print("Orthogonalizer Symmetry 1: ", xp.isclose(X_t@overlaps@X, xp.identity(overlaps.shape[0])).all())
print("Orthogonalizer Symmetry 2: ", sym(X))
print("F_prime: ", sym(F_prime))
print("C: ", xp.isclose(C.T@overlaps@C, xp.identity(overlaps.shape[0])).all())
print("C_prime Orthogonal: ", xp.isclose(C_prime @ C_prime.T, xp.identity(C_prime.shape[0])).all())
print("Eigenvalue Equation: ", xp.isclose(F @ C, overlaps @ C @ xp.diag(orb_energy)).all())
print("Electron Count: ", N_e == elec_count)
print("Spin Population 1: ", N_e == N_a + N_b)
print("Spin Population 2: ", mult - 1 == N_a - N_b)
print("N_a: ", N_a >= 0)
print("N_b: ", N_b >= 0)
print("Trace Electron Count A: ", xp.isclose(xp.trace(P_a @ overlaps), N_a))
print("Trace Electron Count B: ", xp.isclose(xp.trace(P_b @ overlaps), N_b))
print("Trace Electron Count Total: ", xp.isclose(xp.trace((P_a + P_b) @ overlaps), N_e))
print("UHF Density Matrix A Symmetry: ", sym(P_a))
print("UHF Density Matrix B Symmetry: ", sym(P_b))
print("UHF Density Matrix A Idempotent: ", xp.isclose(P_a@overlaps@P_a, P_a).all())
print("UHF Density Matrix B Idempotent: ", xp.isclose(P_b@overlaps@P_b, P_b).all())
print("UHF Density Matrix A Rank: ", xp.linalg.matrix_rank(P_a) == N_a)
print("UHF Density Matrix B Rank: ", xp.linalg.matrix_rank(P_b) == N_b)
print("Coulomb Matrix Symmetry: ", sym(J_matrix))
print("Exchange Matrix A Symmetry: ", sym(K_a))
print("Exchange Matrix B Symmetry: ", sym(K_b))
print("Fock Matrix A Symmetry: ", sym(Fock_a))
print("Fock Matrix B Symmetry: ", sym(Fock_b))
print("C_a Overlap Normal: ", xp.isclose(C_a.T @ overlaps @ C_a, xp.identity(C_a.shape[0])).all())
print("C_b Overlap Normal: ", xp.isclose(C_b.T @ overlaps @ C_b, xp.identity(C_b.shape[0])).all())
print("Eigenvalue Fock A: ", xp.isclose(Fock_a @ C_a, overlaps @ C_a @ xp.diag(orb_energy_a)).all())
print("Eigenvalue Fock B: ", xp.isclose(Fock_b @ C_b, overlaps @ C_b @ xp.diag(orb_energy_b)).all())
print("UHF Density Matrix A Overlap Rank: ", xp.linalg.matrix_rank(P_a @ overlaps) == N_a)
print("UHF Density Matrix B Overlap Rank: ", xp.linalg.matrix_rank(P_b @ overlaps) == N_b)
print("UHF Density Matrix A Residual: ", xp.isclose(Fock_a @ P_a @ overlaps, overlaps @ P_a @ Fock_a).all())
print("UHF Density Matrix B Residual: ", xp.isclose(Fock_b @ P_b @ overlaps, overlaps @ P_b @ Fock_b).all())
print("Numerical Integral A: ", xp.isclose(N_a, xp.sum(alpha_density)*grid_spacing**3, rtol=1e-2, atol=1e-2))
print("Numerical Integral B: ", xp.isclose(N_b, xp.sum(beta_density)*grid_spacing**3, rtol=1e-2, atol=1e-2))
print("Alpha Density Nonnegative: ", (alpha_density >= 0).all())
print("Beta Density Nonnegative: ", (beta_density >= 0).all())
'''