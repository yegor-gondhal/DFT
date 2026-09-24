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
atoms = ["3"]
centers = [[0, 0, 0]]
unpaired_elec = [1]
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
                "atom_pos": xp.array(centers[i], dtype=xp.float64),
                "angular_momentum": l[0],
                "exponents": xp.array(exponents),
                "coefficients": xp.array(coefficients),
            }
            shells.append(shell)
        elif len(l) == len(coefficients):
            for momentum, coef_row in zip(l, coefficients):
                shell = {
                    "atom_idx": i,
                    "atom_pos": xp.array(centers[i], dtype=xp.float64),
                    "angular_momentum": momentum,
                    "exponents": xp.array(exponents),
                    "coefficients": xp.array([coef_row]),
                }
                shells.append(shell)
        else:
            raise ValueError("Something went wrong")

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


def T_raw(exp_a, exp_b, cen_a, cen_b, pow_a, pow_b, prev_overlap):
    result = -2 * exp_b[None, :] * (2 * pow_b[None, :] + 1) * prev_overlap
    result += 4 * xp.square(exp_b[None, :]) * overlap(exp_a, exp_b, cen_a, cen_b, pow_a, pow_b + 2)
    if xp.any(pow_b >= 2):
        idxs = xp.where(pow_b >= 2)[0]
        result[:, idxs] += pow_b[idxs][None, :] * (pow_b[idxs][None, :] - 1) * overlap(exp_a, exp_b[idxs], cen_a, cen_b, pow_a, pow_b[idxs] - 2)
    return result

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

__device__ void convolution(const double* E_ab, const double* E_cd, int E_i, int E_j, int len_i, int len_j, double* C) {{

    int out_len = len_i + len_j - 1;
    for (int s = 0; s < out_len; ++s) {{
        C[s] = 0.0;
    }}

    for (int t = 0; t < len_i; ++t) {{
        for (int tau = 0; tau < len_j; ++tau) {{
            double sign = (tau & 1) ? -1.0 : 1.0;
            C[t+tau] += E_ab[E_i + t]*E_cd[E_j + tau]*sign;
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
    const double* exp_a, 
    const double* exp_b, 
    const double* cen_a, 
    const double* cen_b, 
    const int* pow_a, 
    const int* pow_b,
    const int* atoms, 
    const double* atom_centers,
    const int num_atoms, 
    const int num_funcs_a, 
    const int num_funcs_b,
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

    long long num_pairs = (long long)num_funcs_a * (long long)num_funcs_b;

    if (q >= num_pairs) {{
        return;
    }}

    int a = q / num_funcs_b;
    int b = q % num_funcs_b;

    double Ex[max_e];
    double Ey[max_e];
    double Ez[max_e];

    double e1 = exp_a[a];
    double e2 = exp_b[b];
    double c1x = cen_a[0];
    double c2x = cen_b[0];
    double c1y = cen_a[1];
    double c2y = cen_b[1];
    double c1z = cen_a[2];
    double c2z = cen_b[2];
    int p1x = pow_a[3*a];
    int p2x = pow_b[3*b];
    int p1y = pow_a[3*a+1];
    int p2y = pow_b[3*b+1];
    int p1z = pow_a[3*a+2];
    int p2z = pow_b[3*b+2];

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
    const int K_ab,
    const double* p_ab,
    const double* P_ab,
    const double* Exab,
    const double* Eyab,
    const double* Ezab,
    const int* nxab,
    const int* nyab,
    const int* nzab,
    const int K_cd,
    const double* p_cd,
    const double* P_cd,
    const double* Excd,
    const double* Eycd,
    const double* Ezcd,
    const int* nxcd,
    const int* nycd,
    const int* nzcd,
    double* eri_output
) {{

    long long num_current =
    (long long)K_ab * (long long)K_cd;
    
    long long q =
    (long long)blockIdx.x * blockDim.x + threadIdx.x;

    if (q >= num_current) {{
        return;
    }}

    int i = q / K_cd;
    int j = q % K_cd;

    double p_i = p_ab[i];
    double p_j = p_cd[j];

    int nxi = nxab[i];
    int nxj = nxcd[j];
    int nyi = nyab[i];
    int nyj = nycd[j];
    int nzi = nzab[i];
    int nzj = nzcd[j];

    int x_len = nxi + nxj - 1;
    int y_len = nyi + nyj - 1;
    int z_len = nzi + nzj - 1;
    int n_len = x_len + y_len + z_len - 2;

    double rho = (p_i*p_j)/(p_i+p_j);
    double r_ij0 = P_ab[3*i] - P_cd[3*j];
    double r_ij1 = P_ab[3*i+1] - P_cd[3*j+1];
    double r_ij2 = P_ab[3*i+2] - P_cd[3*j+2];
    double R[max_r];

    calc_R(x_len, y_len, z_len, n_len, r_ij0, r_ij1, r_ij2, rho, R);

    double Cx[max_conv];
    double Cy[max_conv];
    double Cz[max_conv];

    convolution(Exab, Excd, i*max_e, j*max_e, nxi, nxj, Cx);
    convolution(Eyab, Eycd, i*max_e, j*max_e, nyi, nyj, Cy);
    convolution(Ezab, Ezcd, i*max_e, j*max_e, nzi, nzj, Cz);

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

def eri_loop(pair_list, pair_cache, shells, P, J_matrix, K_a, K_b):
    threads = 128

    for ab_idx, (a, b) in enumerate(pair_list):
        pair_ab = pair_cache[(a, b)]

        for cd_idx in range(ab_idx + 1):
            c, d = pair_list[cd_idx]
            pair_cd = pair_cache[(c, d)]
            eri_output = xp.zeros((pair_ab["K"] * pair_cd["K"]))
            blocks = (pair_ab["K"] * pair_cd["K"] + threads - 1) // threads
            eri_kernel((blocks,), (threads,), (
                pair_ab["K"],
                pair_ab["p"],
                pair_ab["P"],
                pair_ab["Ex"],
                pair_ab["Ey"],
                pair_ab["Ez"],
                pair_ab["nx"],
                pair_ab["ny"],
                pair_ab["nz"],
                pair_cd["K"],
                pair_cd["p"],
                pair_cd["P"],
                pair_cd["Ex"],
                pair_cd["Ey"],
                pair_cd["Ez"],
                pair_cd["nx"],
                pair_cd["ny"],
                pair_cd["nz"],
                eri_output
            ))
            size_a = shells[a]["n_exponents"] * shells[a]["n_components"]
            size_b = shells[b]["n_exponents"] * shells[b]["n_components"]
            size_c = shells[c]["n_exponents"] * shells[c]["n_components"]
            size_d = shells[d]["n_exponents"] * shells[d]["n_components"]
            eri_output = eri_output.reshape(size_a, size_b, size_c, size_d)
            eri_output *= shells[a]["normalization"][:, None, None, None]
            eri_output *= shells[b]["normalization"][None, :, None, None]
            eri_output *= shells[c]["normalization"][None, None, :, None]
            eri_output *= shells[d]["normalization"][None, None, None, :]
            eri_output = eri_output.reshape(
                shells[a]["n_components"],
                shells[a]["n_exponents"],
                shells[b]["n_components"],
                shells[b]["n_exponents"],
                shells[c]["n_components"],
                shells[c]["n_exponents"],
                shells[d]["n_components"],
                shells[d]["n_exponents"],
            )
            eri_output = xp.einsum(
                "apbqcrds,ip,jq,kr,ls->iajbkcld",
                eri_output,
                shells[a]["coefficients"],
                shells[b]["coefficients"],
                shells[c]["coefficients"],
                shells[d]["coefficients"],
            ).reshape(
                shells[a]["n_ao"],
                shells[b]["n_ao"],
                shells[c]["n_ao"],
                shells[d]["n_ao"],
            )
            sa = slice(shells[a]["ao_start"], shells[a]["ao_stop"])
            sb = slice(shells[b]["ao_start"], shells[b]["ao_stop"])
            sc = slice(shells[c]["ao_start"], shells[c]["ao_stop"])
            sd = slice(shells[d]["ao_start"], shells[d]["ao_stop"])
            s = [sa, sb, sc, sd]
            orientations = [
                ((0, 1, 2, 3), eri_output),
                ((1, 0, 2, 3), eri_output.transpose(1, 0, 2, 3)),
                ((0, 1, 3, 2), eri_output.transpose(0, 1, 3, 2)),
                ((1, 0, 3, 2), eri_output.transpose(1, 0, 3, 2)),
                ((2, 3, 0, 1), eri_output.transpose(2, 3, 0, 1)),
                ((3, 2, 0, 1), eri_output.transpose(3, 2, 0, 1)),
                ((2, 3, 1, 0), eri_output.transpose(2, 3, 1, 0)),
                ((3, 2, 1, 0), eri_output.transpose(3, 2, 1, 0)),
            ]
            base_shell_indices = (a, b, c, d)
            seen = set()

            for quad, eri in orientations:
                actual_quad = tuple(
                    base_shell_indices[index]
                    for index in quad
                )

                if actual_quad in seen:
                    continue

                seen.add(actual_quad)
                J_matrix[s[quad[0]], s[quad[1]]] += xp.einsum(
                    "abcd,cd->ab",
                    eri,
                    P[s[quad[2]], s[quad[3]]],
                )

                K_a[s[quad[0]], s[quad[2]]] += xp.einsum(
                    "abcd,bd->ac",
                    eri,
                    P_a[s[quad[1]], s[quad[3]]],
                )

                K_b[s[quad[0]], s[quad[2]]] += xp.einsum(
                    "abcd,bd->ac",
                    eri,
                    P_b[s[quad[1]], s[quad[3]]],
                )


print("Setting Kernel...")
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

print("Defining Shell...")
total_ao = 0
for shell in shells:
    shell["components"] = xp.array(combinations(shell["angular_momentum"]), dtype=xp.int32)
    shell["n_components"] = len(shell["components"])
    shell["n_contractions"] = shell["coefficients"].shape[0]
    shell["n_ao"] = shell["n_components"] * shell["n_contractions"]
    shell["n_exponents"] = xp.size(shell["exponents"])
    shell["ao_start"] = total_ao
    shell["ao_stop"] = total_ao + shell["n_ao"]
    exp = xp.tile(shell["exponents"], shell["n_components"])
    pow = xp.repeat(shell["components"], shell["n_exponents"], axis=0).reshape(-1, 3)
    shell["normalization"] = normal(exp, pow)
    total_ao = shell["ao_stop"]

overlaps = xp.empty((total_ao, total_ao))
T_matrix = xp.empty((total_ao, total_ao))
V_matrix = xp.empty((total_ao, total_ao))

pair_cache = {}
pair_list = []

print("Overlaps/T_matrix...")
for i, shell_a in enumerate(shells):
    for j in range(i+1):
        shell_b = shells[j]
        exp_a = xp.tile(shell_a["exponents"], shell_a["n_components"])
        exp_b = xp.tile(shell_b["exponents"], shell_b["n_components"])
        pow_a = xp.repeat(shell_a["components"], shell_a["n_exponents"], axis=0).reshape(-1, 3)
        pow_b = xp.repeat(shell_b["components"], shell_b["n_exponents"], axis=0).reshape(-1, 3)

        overlap_shell = xp.empty((3, xp.size(exp_a), xp.size(exp_b)))
        T_prim = xp.empty((3, xp.size(exp_a), xp.size(exp_b)))
        for dim in range(3):
            overlap_shell[dim] = overlap(exp_a, exp_b, shell_a["atom_pos"][dim], shell_b["atom_pos"][dim], pow_a[:, dim], pow_b[:, dim])
            T_prim[dim] = T_raw(exp_a, exp_b, shell_a["atom_pos"][dim], shell_b["atom_pos"][dim], pow_a[:, dim], pow_b[:, dim], overlap_shell[dim])

        overlap_prim = xp.prod(overlap_shell, axis=0)
        overlap_prim = overlap_prim * (shell_a["normalization"][:, None] * shell_b["normalization"][None, :])
        overlap_prim = overlap_prim.reshape(shell_a["n_components"], shell_a["n_exponents"], shell_b["n_components"], shell_b["n_exponents"])
        overlap_block = xp.einsum(
            "apbq,kp,lq->kalb",
            overlap_prim,
            shell_a["coefficients"],
            shell_b["coefficients"]
        ).reshape(shell_a["n_ao"], shell_b["n_ao"])

        idx1 = xp.array([0, 1, 2])
        idx2 = xp.array([2, 0, 1])
        idx3 = xp.array([1, 2, 0])
        T_block = xp.sum(T_prim[idx1] * overlap_shell[idx2] * overlap_shell[idx3], axis=0)
        T_block = T_block * (shell_a["normalization"][:, None] * shell_b["normalization"][None, :])
        T_block = T_block.reshape(shell_a["n_components"], shell_a["n_exponents"], shell_b["n_components"], shell_b["n_exponents"])
        T_block = xp.einsum(
            "apbq,kp,lq->kalb",
            T_block,
            shell_a["coefficients"],
            shell_b["coefficients"]
        ).reshape(shell_a["n_ao"], shell_b["n_ao"])
        T_block /= -2

        num_funcs_a = int(exp_a.size)
        num_funcs_b = int(exp_b.size)
        K = num_funcs_a * num_funcs_b
        max_e = 7
        Ex = xp.empty((K * max_e))
        Ey = xp.empty((K * max_e))
        Ez = xp.empty((K * max_e))
        nx = xp.empty(K, xp.int32)
        ny = xp.empty(K, xp.int32)
        nz = xp.empty(K, xp.int32)
        p_k = xp.empty(K)
        P_k = xp.empty(K * 3)
        uncontracted_V_matrix = xp.empty(K)
        threads = 128
        blocks = (K + threads - 1) // threads
        nuclear_kernel((blocks,), (threads,), (
            exp_a,
            exp_b,
            shell_a["atom_pos"],
            shell_b["atom_pos"],
            pow_a,
            pow_b,
            Z,
            centers,
            xp.int32(xp.size(Z)),
            num_funcs_a,
            num_funcs_b,
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
        uncontracted_V_matrix = uncontracted_V_matrix.reshape(num_funcs_a, num_funcs_b)
        V_block = uncontracted_V_matrix * (shell_a["normalization"][:, None] * shell_b["normalization"][None, :])
        V_block = V_block.reshape(shell_a["n_components"], shell_a["n_exponents"], shell_b["n_components"], shell_b["n_exponents"])
        V_block = xp.einsum(
            "apbq,kp,lq->kalb",
            V_block,
            shell_a["coefficients"],
            shell_b["coefficients"]
        ).reshape(shell_a["n_ao"], shell_b["n_ao"])

        key = (i, j)
        pair_cache[key] = {
            "K": K,
            "p": p_k,
            "P": P_k,
            "Ex": Ex,
            "Ey": Ey,
            "Ez": Ez,
            "nx": nx,
            "ny": ny,
            "nz": nz,
        }
        pair_list.append(key)

        slicea = slice(shell_a["ao_start"], shell_a["ao_stop"])
        sliceb = slice(shell_b["ao_start"], shell_b["ao_stop"])
        overlaps[slicea, sliceb] = overlap_block
        T_matrix[slicea, sliceb] = T_block
        V_matrix[slicea, sliceb] = V_block
        if i != j:
            overlaps[sliceb, slicea] = overlap_block.T
            T_matrix[sliceb, slicea] = T_block.T
            V_matrix[sliceb, slicea] = V_block.T


print("Setting Up SCF...")
E_NN = nuclear_repulsion(Z, centers)
elec_count = xp.sum(Z) - molecular_charge
H_matrix = T_matrix + V_matrix
eig_vals, U = xp.linalg.eigh(overlaps)
s = xp.power(eig_vals, -0.5)
s = xp.diag(s)
U_t = U.T
X = U @ s @ U_t
X_t = X.T

F_prime = X_t @ H_matrix @ X
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
    print(count)
    P = P_a + P_b

    J_matrix = xp.zeros_like(H_matrix)
    K_a = xp.zeros_like(H_matrix)
    K_b = xp.zeros_like(H_matrix)
    eri_loop(pair_list, pair_cache, shells, P, J_matrix, K_a, K_b)

    Fock_a = H_matrix + J_matrix - K_a
    Fock_b = H_matrix + J_matrix - K_b

    E_elec = 0.5 * xp.sum(P * H_matrix + P_a * Fock_a + P_b * Fock_b)
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
        J_matrix = xp.zeros_like(H_matrix)
        K_a = xp.zeros_like(H_matrix)
        K_b = xp.zeros_like(H_matrix)
        eri_loop(pair_list, pair_cache, shells, P, J_matrix, K_a, K_b)
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
chunk_size = int(1e6)
C_a = C_a.astype(xp.float32)


xp.save("preprocess/grid.npy", grid)
number_of_orbitals = C_a.shape[1]
total_file = np.lib.format.open_memmap(
    "preprocess/total_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(number_of_orbitals, grid_len)
)

spin_file = np.lib.format.open_memmap(
    "preprocess/spin_density.npy",
    mode="w+",
    dtype=np.float32,
    shape=(number_of_orbitals, grid_len)
)

C_a = C_a.T
C_b = C_b.T
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
        contracted_values = xp.einsum("kp,apg->kag",shell["coefficients"], vals)
        vals = vals.reshape(shell["n_ao"], -1)
        ao_vals[shell["ao_start"]:shell["ao_stop"], :] = vals

    psi_a = C_a @ ao_vals
    psi_b = C_b @ ao_vals
    mo_density_a = xp.real(psi_a.conj() * psi_a)
    mo_density_b = xp.real(psi_b.conj() * psi_b)
    occ_a = xp.zeros(total_ao)
    occ_b = xp.zeros(total_ao)
    occ_a[:int(N_a)] = 1.0
    occ_b[:int(N_b)] = 1.0
    occupied_density_a = occ_a[:, None] * mo_density_a
    occupied_density_b = occ_b[:, None] * mo_density_b
    orbital_total_density = occupied_density_a + occupied_density_b
    orbital_spin_density = occupied_density_a - occupied_density_b

    total_file[:, write:write_to] = orbital_total_density
    spin_file[:, write:write_to] = orbital_spin_density
