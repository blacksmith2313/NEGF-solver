# import os
from matplotlib.colors import LogNorm
import numpy as np
import device_profile as prf
import matplotlib.pyplot as plt
# from concurrent.futures import ProcessPoolExecutor
from numba import njit
from numba import prange
import cmath
# import math
#

hbar = 1.054e-34
q = 1.6022e-19
m0 = 9.1095e-31
kB = 1.38e-23
T = 300

meff = prf.meff
a = prf.a
l = np.sum(prf.L)
mode = prf.mode

m = m0
meff_0 = float(meff[0])
t0 = hbar**2 / (2 * m0 * a**2)  # in the device. Effective mass in taken into account in the Hamiltonian creation
tl = hbar**2 / (2 * meff[0] * m0 * a**2)  # in the left lead
tdl = tl  # in the device-left lead
tld = tl  # in the left lead-device
tr = hbar**2 / (2 * meff[-1] * m0 * a**2)  # in the right lead
tdr = tr  # in the device-right lead
trd = tr  # in the right lead-device

Na = prf.Na
Nd = prf.Nd
epsilons = prf.epsilons
EffMass = prf.EffMass * m0
N = prf.N
init_band = prf.init_band
vs_profile = prf.vs_profile
mass_density_profile = prf.mass_density_profile
optical_deformation_profile = prf.optical_deformation_profile
acoustic_deformation_profile = prf.acoustic_deformation_profile
E_phonon_profile = prf.E_phonon_profile
D_op = prf.D_op
D_ap = prf.D_ap
n_b = prf.n_b


@njit(cache=True, fastmath=True)
def f(E_total, lead, EFl, EFr):
    if lead == "left":
        return 1 / (1 + np.exp((E_total - EFl) / (kB * T)))
    elif lead == "right":
        return 1 / (1 + np.exp((E_total - EFr) / (kB * T)))
    return 0.0


@njit(cache=True, fastmath=True)
def k(El, Er):
    kl = (1 / a) * cmath.acos(-El / (2 * tl))
    kr = (1 / a) * cmath.acos(-Er / (2 * tr))

    if (kl.real < 0) or (kl.imag < 0 and np.abs(kl.imag) > 1e-6):
        kl = -kl
    if (kr.real < 0) or (kr.imag < 0 and np.abs(kr.imag) > 1e-6):
        kr = -kr

    return kl, kr


@njit(cache=True, fastmath=True)
def Sigma_Retarded(kl, kr, Sigma_R_phonon):
    Sigma_R_leads = np.zeros(N, dtype=np.complex128)
    Sigma_R_leads[0] = -tdl * np.exp(1j * kl * a) * (1 / tl) * tld
    Sigma_R_leads[N - 1] = -tdr * np.exp(1j * kr * a) * (1 / tr) * trd

    Sigma_R_total = Sigma_R_leads + Sigma_R_phonon
    return Sigma_R_total


@njit(cache=True, fastmath=True)
def Sigma_Lesser(Sigma_L_phonon, E_longitudinal, E_transverse, kl, kr, EFl, EFr):
    Sigma_R_leads_left = -tdl * np.exp(1j * kl * a) * (1 / tl) * tld
    Sigma_R_leads_right = -tdr * np.exp(1j * kr * a) * (1 / tr) * trd

    Sigma_L_leads_left = -2 * Sigma_R_leads_left.imag * f(E_longitudinal + E_transverse, "left", EFl, EFr)
    Sigma_L_leads_right = -2 * Sigma_R_leads_right.imag * f(E_longitudinal + E_transverse, "right", EFl, EFr)

    Sigma_L_total = np.zeros(N, dtype=np.complex128)
    Sigma_L_total[0] += Sigma_L_leads_left
    Sigma_L_total[N - 1] += Sigma_L_leads_right

    Sigma_L_total += Sigma_L_phonon
    return Sigma_L_total


@njit(cache=True, fastmath=True)
def Sigma_Greater(Sigma_G_phonon, E_longitudinal, E_transverse, kl, kr, EFl, EFr):
    Sigma_R_leads_left = -tdl * np.exp(1j * kl * a) * (1 / tl) * tld
    Sigma_R_leads_right = -tdr * np.exp(1j * kr * a) * (1 / tr) * trd

    Sigma_G_leads_left = -2 * Sigma_R_leads_left.imag * (1 - f(E_longitudinal + E_transverse, "left", EFl, EFr))
    Sigma_G_leads_right = -2 * Sigma_R_leads_right.imag * (1 - f(E_longitudinal + E_transverse, "right", EFl, EFr))

    Sigma_G_total = np.zeros(N, dtype=np.complex128)
    Sigma_G_total[0] += Sigma_G_leads_left
    Sigma_G_total[N - 1] += Sigma_G_leads_right

    Sigma_G_total += Sigma_G_phonon
    return Sigma_G_total


@njit(cache=True, fastmath=True)
def H_mat(Potential_energy_device):
    H = np.zeros((2, N))
    for i in range(N - 1):
        H[0, i] = -(hbar**2 / a**2) / (EffMass[i] + EffMass[i + 1])

    for i in range(N):
        if i == 0:
            t_right = -H[0, 0]
            H[1, i] = 2 * t_right + Potential_energy_device[i]
        elif i == N - 1:
            t_left = -H[0, N - 2]
            H[1, i] = 2 * t_left + Potential_energy_device[i]
        else:
            t_left = -H[0, i - 1]
            t_right = -H[0, i]
            H[1, i] = t_left + t_right + Potential_energy_device[i]

    return H


@njit(cache=True, fastmath=True)
def custom_I(factor):
    I = np.zeros((2, N), dtype=np.complex128)
    for i in range(N):
        I[1, i] = factor
    return I

CI = custom_I(1)

@njit(cache=True, fastmath=True)
def A_mat(E_longitudinal, Vx, Sigma_R_phonon):
    eta = 1j * q * 1e-7
    H = H_mat(Vx)

    A = (E_longitudinal * CI - H).astype(np.complex128)

    kl, kr = k(A[1, 0], A[1, N - 1])
    Sigma_R_total = Sigma_Retarded(kl, kr, Sigma_R_phonon)

    A = A + eta * CI
    A[1, :] = A[1, :] - Sigma_R_total

    return A, Sigma_R_total, kl, kr


@njit(cache=True, fastmath=True)
def RGF_Retarded(A):
    G_R = np.zeros(N, dtype=np.complex128)
    g_R_left = np.zeros(N, dtype=np.complex128)

    g_R_left[0] = 1 / A[1, 0]

    for i in range(1, N):
        g_R_left[i] = 1 / (A[1, i] - g_R_left[i - 1] * A[0, i - 1] ** 2)

    G_R[N - 1] = g_R_left[N - 1]

    for i in range(N - 2, -1, -1):
        G_R[i] = g_R_left[i] + (g_R_left[i] ** 2) * (A[0, i] ** 2) * G_R[i + 1]

    return G_R, g_R_left


@njit(cache=True, fastmath=True)
def RGF_Lesser_Greater(A, G_R, g_R_left, Sigma_L_total, Sigma_G_total):
    G_L = np.zeros(N, dtype=np.complex128)
    G_G = np.zeros(N, dtype=np.complex128)
    g_L_left = np.zeros(N, dtype=np.complex128)
    g_G_left = np.zeros(N, dtype=np.complex128)

    G_L_upper = np.zeros(N - 1, dtype=np.complex128)
    G_G_upper = np.zeros(N - 1, dtype=np.complex128)

    # Lower off-diagonal elements (i+1, i)
    G_L_lower = np.zeros(N - 1, dtype=np.complex128)
    G_G_lower = np.zeros(N - 1, dtype=np.complex128)

    g_L_left[0] = (np.abs(g_R_left[0]) ** 2) * Sigma_L_total[0]
    g_G_left[0] = (np.abs(g_R_left[0]) ** 2) * Sigma_G_total[0]

    for i in range(1, N):
        g_L_left[i] = (np.abs(g_R_left[i]) ** 2) * (Sigma_L_total[i] + (A[0, i - 1] ** 2) * g_L_left[i - 1])
        g_G_left[i] = (np.abs(g_R_left[i]) ** 2) * (Sigma_G_total[i] + (A[0, i - 1] ** 2) * g_G_left[i - 1])

    G_L[N - 1] = g_L_left[N - 1]
    G_G[N - 1] = g_G_left[N - 1]

    for i in range(N - 2, -1, -1):
        G_R_off = -g_R_left[i] * A[0, i] * G_R[i + 1]

        G_L_upper[i] = -g_R_left[i] * A[0, i] * G_L[i + 1] - g_L_left[i] * A[0, i] * np.conj(G_R[i + 1])
        G_G_upper[i] = -g_R_left[i] * A[0, i] * G_G[i + 1] - g_G_left[i] * A[0, i] * np.conj(G_R[i + 1])

        # Calculate lower off-diagonal G^{<,>}_{i+1, i} via anti-Hermiticity
        G_L_lower[i] = -np.conj(G_L_upper[i])
        G_G_lower[i] = -np.conj(G_G_upper[i])

        G_L[i] = g_L_left[i] - 2 * np.real(G_R_off) * g_L_left[i] * A[0, i] + (np.abs(g_R_left[i]) ** 2) * (A[0, i] ** 2) * G_L[i + 1]
        G_G[i] = g_G_left[i] - 2 * np.real(G_R_off) * g_G_left[i] * A[0, i] + (np.abs(g_R_left[i]) ** 2) * (A[0, i] ** 2) * G_G[i + 1]

    return G_L, G_G, G_L_upper, G_G_upper, G_L_lower, G_G_lower 

@njit(cache=True)
def acoustic_self_energies(N_E_longitudinal, G_R_El_grid, G_L_El_grid, G_G_El_grid ):
    Sigma_L_ap = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_G_ap = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_R_ap = np.zeros((N_E_longitudinal, N), dtype=np.complex128)

    for i in range(N_E_longitudinal):
        for x in range(N):
            prefactor = D_ap[x] * (EffMass[x] / (2 * np.pi * hbar**2))

            Sigma_L_ap[i,x] = prefactor * G_L_El_grid[i,x]  
            Sigma_G_ap[i,x] = prefactor * G_G_El_grid[i,x]  
            Sigma_R_ap[i,x] = (-0.5j)*(Sigma_L_ap[i,x] + Sigma_G_ap[i,x])

    return Sigma_L_ap, Sigma_G_ap, Sigma_R_ap

@njit(cache=True, fastmath=True)
def optical_self_energies(N_E_longitudinal, E_longitudinal_array, G_L_El_grid, G_G_El_grid, dE_longitudinal):
    Sigma_L_op = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_G_op = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_R_op = np.zeros((N_E_longitudinal, N), dtype=np.complex128)


    for i in range(N_E_longitudinal):
        for x in range(N):

            dE_actual = E_longitudinal_array[1] - E_longitudinal_array[0]
            val_minus = (E_longitudinal_array[i] - E_phonon_profile[x] - E_longitudinal_array[0]) / dE_actual
            val_plus = (E_longitudinal_array[i] + E_phonon_profile[x] - E_longitudinal_array[0]) / dE_actual 

            i_minus = int(round(val_minus))
            i_plus = int(round(val_plus))

            if 0 <= i_minus < N_E_longitudinal:
                G_L_m = G_L_El_grid[i_minus, x]
                G_G_m = G_G_El_grid[i_minus, x]
            else:
                G_L_m = 0.0j
                G_G_m = 0.0j

            if 0 <= i_plus < N_E_longitudinal:
                G_L_p = G_L_El_grid[i_plus, x]
                G_G_p = G_G_El_grid[i_plus, x]
            else:
                G_L_p = 0.0j
                G_G_p = 0.0j

            prefactor = D_op[x] * (EffMass[x] / (2 * np.pi * hbar**2))

            Sigma_L_op[i, x] = prefactor * ((n_b[x] + 1) * G_L_p + n_b[x] * G_L_m)
            Sigma_G_op[i, x] = prefactor * ((n_b[x] + 1) * G_G_m + n_b[x] * G_G_p)
            Sigma_R_op[i, x] = (-0.5j) * (Sigma_L_op[i, x] + Sigma_G_op[i, x])

    return Sigma_L_op, Sigma_G_op, Sigma_R_op


@njit(cache=True, fastmath=True)
def carrier_concentration(G_L_transverse_integrated_grid, G_G_transverse_integrated_grid, dE_longitudinal):
    prefactor = EffMass / (2 * (np.pi**2) * (hbar**2) * a)
    n = prefactor * (np.abs(np.sum(G_L_transverse_integrated_grid, axis=0))) * dE_longitudinal
    p = prefactor * (np.abs(np.sum(G_G_transverse_integrated_grid, axis=0))) * dE_longitudinal
    return n, p, prefactor * G_L_transverse_integrated_grid


@njit(parallel=True, cache=True, fastmath=True)
def Greens_and_Sigmas(Sigma_R_phonon, Sigma_L_phonon, Sigma_G_phonon, dE_transverse, N_E_longitudinal, N_E_transverse, E_longitudinal_array, E_transverse_array, Vx, EFl, EFr):
    Sigma_R_total = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_L_total = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_G_total = np.zeros((N_E_longitudinal, N), dtype=np.complex128)

    G_L = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    G_G = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    G_R = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    T_El_left = np.zeros(N_E_longitudinal, dtype=np.complex128)

    G_L_lower = np.zeros((N_E_longitudinal, N-1), dtype=np.complex128)
    G_L_upper = np.zeros((N_E_longitudinal, N-1), dtype=np.complex128)

    for i in prange(N_E_longitudinal):
        A, Sigma_R_total[i, :], kl, kr = A_mat(E_longitudinal_array[i], Vx, Sigma_R_phonon[i, :])
        G_R[i, :], g_R_left = RGF_Retarded(A)

        Sigma_L = np.zeros((N_E_transverse, N), dtype=np.complex128)
        Sigma_G = np.zeros((N_E_transverse, N), dtype=np.complex128)
        g_L = np.zeros((N_E_transverse, N), dtype=np.complex128)
        g_G = np.zeros((N_E_transverse, N), dtype=np.complex128)

        g_L_lower = np.zeros((N_E_transverse, N-1), dtype=np.complex128)
        g_L_upper = np.zeros((N_E_transverse, N-1), dtype=np.complex128)

        for j in range(N_E_transverse):
            Sigma_L[j, :] = Sigma_Lesser(Sigma_L_phonon[i, :], E_longitudinal_array[i], E_transverse_array[j], kl, kr, EFl, EFr)
            Sigma_G[j, :] = Sigma_Greater(Sigma_G_phonon[i, :], E_longitudinal_array[i], E_transverse_array[j], kl, kr, EFl, EFr)

            g_L[j, :], g_G[j, :], g_L_upper[j, :], _ , g_L_lower[j, :], _  = RGF_Lesser_Greater(A, G_R[i, :], g_R_left, Sigma_L[j, :], Sigma_G[j, :])

        Sigma_L_total[i, :] = np.sum(Sigma_L, axis=0) * dE_transverse
        Sigma_G_total[i, :] = np.sum(Sigma_G, axis=0) * dE_transverse
        G_L[i, :] = np.sum(g_L, axis=0) * dE_transverse
        G_G[i, :] = np.sum(g_G, axis=0) * dE_transverse

        G_L_lower[i, :] = np.sum(g_L_lower, axis=0) * dE_transverse
        G_L_upper[i, :] = np.sum(g_L_upper, axis=0) * dE_transverse

    return G_R, G_L, G_G, G_L_lower, G_L_upper, Sigma_L_total, Sigma_G_total, Sigma_R_total


@njit(cache=True, fastmath=True)
def current(N_E_longitudinal, dE_longitudinal, transverse_integrated_G_L_lower, transverse_integrated_G_L_upper):
    # I = (2 * q * m0 * meff_0 / (np.pi**2 * hbar**3 * 4)) * np.sum(T_El_left) * dE_longitudinal 
    factor = 2 *  q / (8 * np.pi**2 * a**2 * hbar)


    transmission = np.zeros((N_E_longitudinal,N) , dtype=np.complex128)
    for i in range(N_E_longitudinal):
        for j in range(N):
            transmission[i][j] = -factor * (transverse_integrated_G_L_upper[i][j].imag)

    # transmission = -factor * (transverse_integrated_G_L_upper.imag)

    # J = -factor * np.sum(transverse_integrated_G_L_lower - transverse_integrated_G_L_upper, axis=0) * dE_longitudinal 
    J = -factor * np.sum( transverse_integrated_G_L_upper.imag, axis=0) * dE_longitudinal 

    return J, transmission


@njit(cache=True, fastmath=True)
def SCBA(G_R_El, G_L_El, G_G_El, E_longitudinal, E_transverse, dE_longitudinal, N_E_longitudinal, N_E_transverse, dE_transverse, Bias, Vx, EFl, EFr):
    G_R_new = G_R_El
    G_L_new = G_L_El
    G_G_new = G_G_El
    
    Sigma_L_op_phonon_old, Sigma_G_op_phonon_old, Sigma_R_op_phonon_old = optical_self_energies(N_E_longitudinal, E_longitudinal, G_L_El, G_G_El, dE_longitudinal)
    Sigma_L_ac_phonon_old, Sigma_G_ac_phonon_old, Sigma_R_ac_phonon_old = acoustic_self_energies(N_E_longitudinal,G_R_El, G_L_El, G_G_El)

    Sigma_L_phonon_old = Sigma_L_op_phonon_old + Sigma_L_ac_phonon_old
    Sigma_G_phonon_old = Sigma_G_op_phonon_old + Sigma_G_ac_phonon_old
    Sigma_R_phonon_old = Sigma_R_op_phonon_old + Sigma_R_ac_phonon_old


    for iteration in range(1500):
        G_R_new, G_L_new, G_G_new, G_L_lower, G_L_upper, Sigma_L_total, Sigma_G_total, Sigma_R_total = Greens_and_Sigmas(Sigma_R_phonon_old, Sigma_L_phonon_old, Sigma_G_phonon_old, dE_transverse, N_E_longitudinal, N_E_transverse, E_longitudinal, E_transverse, Vx, EFl, EFr)

        Sigma_L_op_phonon_new, Sigma_G_op_phonon_new, Sigma_R_op_phonon_new = optical_self_energies(N_E_longitudinal, E_longitudinal, G_L_new, G_G_new, dE_longitudinal)
        Sigma_L_ac_phonon_new, Sigma_G_ac_phonon_new, Sigma_R_ac_phonon_new = acoustic_self_energies(N_E_longitudinal,G_R_new, G_L_new, G_G_new)

        Sigma_L_phonon_new = Sigma_L_op_phonon_new + Sigma_L_ac_phonon_new
        Sigma_G_phonon_new = Sigma_G_op_phonon_new + Sigma_G_ac_phonon_new
        Sigma_R_phonon_new = Sigma_R_op_phonon_new + Sigma_R_ac_phonon_new

        # Sigma_L_phonon_new, Sigma_G_phonon_new, Sigma_R_phonon_new = optical_self_energies(N_E_longitudinal, E_longitudinal, G_L_new, G_G_new, dE_longitudinal)

        rel_error = (np.max(np.abs(Sigma_L_phonon_new - Sigma_L_phonon_old))/ np.max(np.abs(Sigma_L_phonon_old) + 1e-30) ) * 100

        print("SCBA iteration",  iteration, " | max deviation", float(rel_error))

        # alpha = 0.2
        #

        if (iteration < 20):
            alpha = 0.15
        elif (iteration%10 == 0 or iteration%10 == 1):
            alpha = 1.0 
        else:
            alpha = 0.1

        Sigma_R_phonon_old = alpha * Sigma_R_phonon_new + (1 - alpha) * Sigma_R_phonon_old
        Sigma_L_phonon_old = alpha * Sigma_L_phonon_new + (1 - alpha) * Sigma_L_phonon_old
        Sigma_G_phonon_old = alpha * Sigma_G_phonon_new + (1 - alpha) * Sigma_G_phonon_old
        
        print("Sigma_R_phonon_old", np.max(np.abs(Sigma_R_phonon_old))/q)
        print("Sigma_L_phonon_old", np.max(np.abs(Sigma_L_phonon_old))/q)
        print("Sigma_G_phonon_old", np.max(np.abs(Sigma_G_phonon_old))/q)

        print("------")

        if rel_error < 0.01 and iteration < 1499:
            break
            
        if iteration == 499:
            print("|-----|")
            print("| xxx |")
            print("|-----|")
 
    return G_R_new, G_L_new, G_G_new, G_L_lower, G_L_upper, Sigma_L_phonon_old, Sigma_G_phonon_old, Sigma_R_phonon_old 


@njit(cache=True, fastmath=True)
def single_bias_execute(Bias, Vx, mode):
    EFl = 0.0
    EFr = EFl - (Bias * q)

    v_min = np.min(Vx)
    v_max = np.max(Vx)

    E_longitudinal_min = (v_min if v_min < EFr else EFr) - (0.1 * q)
    E_longitudinal_max = (v_max if v_max > EFl else EFl) + (0.3 * q)

    # E_longitudinal_min = min(np.min(Vx), EFr) - (0.1 * q)
    # E_longitudinal_max = max(np.max(Vx), EFl) + (0.3 * q)

    E_transverse_min = 0.0
    E_transverse_max = 0.25 * q

    dE_longitudinal = q * 1e-3
    dE_transverse = q * 1e-2

    N_E_longitudinal = int(np.abs((E_longitudinal_max - E_longitudinal_min) / dE_longitudinal))
    N_E_transverse = int(np.abs((E_transverse_max - E_transverse_min) / dE_transverse))

    E_longitudinal = np.linspace(E_longitudinal_min, E_longitudinal_max, N_E_longitudinal)
    E_transverse = np.linspace(E_transverse_min, E_transverse_max, N_E_transverse)

    I = np.zeros(N - 1, dtype=np.float64) 
    transmission = np.zeros((N_E_longitudinal, N), dtype=np.complex128)

    Sigma_R_phonon = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_L_phonon = np.zeros((N_E_longitudinal, N), dtype=np.complex128)
    Sigma_G_phonon = np.zeros((N_E_longitudinal, N), dtype=np.complex128)

    G_R, G_L, G_G, G_L_lower, G_L_upper, Sigma_L_total, Sigma_G_total, Sigma_R_total = Greens_and_Sigmas(Sigma_R_phonon, Sigma_L_phonon, Sigma_G_phonon, dE_transverse, N_E_longitudinal, N_E_transverse, E_longitudinal, E_transverse, Vx, EFl, EFr)

    n = np.zeros(N)
    p = np.zeros(N)
    if mode == "ballistic":
        n, p, spectrum = carrier_concentration(G_L, G_G, dE_longitudinal)
        I, transmission = current(N_E_longitudinal, dE_longitudinal, G_L_lower, G_L_upper)
    if mode == "phonon":
        G_R, G_L, G_G, G_L_lower, G_L_upper, Sigma_L_total, Sigma_G_total, Sigma_R_total = SCBA(G_R, G_L, G_G, E_longitudinal, E_transverse, dE_longitudinal, N_E_longitudinal, N_E_transverse, dE_transverse, Bias, Vx, EFl, EFr)
        n, p, spectrum = carrier_concentration(G_L, G_G, dE_longitudinal)
        I, transmission = current(N_E_longitudinal, dE_longitudinal, G_L_lower, G_L_upper)

    return n, p, I, spectrum, transmission, E_longitudinal


# if __name__ == "__main__":
#     x = np.linspace(0,l,N)
#     Vx, Ec = prf.potential_profile(0.3)
#     n, _ , I, spectrum, transmission, E_longitudinal = single_bias_execute(0.3, Vx, mode)
#     transmission = np.abs(transmission)
#     plt.contourf(x/1E-9, E_longitudinal/q, transmission,levels=1000, cmap='plasma') 
#     plt.plot(x/1E-9, Vx/q, color='red')
#     plt.xlabel("position (nm)")
#     plt.ylabel("Energy (ev)")
#     # print(I)
#     # plt.plot(n)
#     # # plt.plot(p)
#     plt.grid(True)
#     plt.show()
