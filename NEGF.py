import os
import numpy as np


# L = np.array([30E-9, 10E-9, 2.0E-9, 4.2E-9, 2.0E-9, 4.2E-9, 2.0E-9, 4.2E-9, 2.0E-9, 10E-9, 30E-9]) # length profile in m
# L = np.array([30E-9, 10E-9, 2.0E-9, 5.0E-9, 2.0E-9, 10E-9, 30E-9]) # length profile in m
# L = np.array([30E-9, 10E-9, 3.0E-9, 5.0E-9, 5.0E-9, 10E-9, 30E-9]) # length profile in m

import device_profile as prf
import matplotlib.pyplot as plt 
from concurrent.futures import ProcessPoolExecutor
from numba import njit
from numba import prange

hbar = 1.054E-34
q = 1.6022E-19
m0 = 9.1095E-31
kB = 1.38E-23
T = 300 

meff = prf.meff
a = prf.a
l = np.sum(prf.L)
meff = np.array(prf.meff, dtype=np.float64)

m   = m0
t0  = hbar**2/(2*m0*a**2) # in the device. Effective mass in taken into account in the Hamiltonian creation
tl  = t0/meff[0]          # in the left lead
tdl = t0/meff[0]          # in the device-left lead
tld = t0/meff[0]          # in the left lead-device
tr  = t0/meff[-1]         # in the right lead
tdr = t0/meff[-1]         # in the device-right lead
trd = t0/meff[-1]         # in the right lead-device



Na, Nd, epsilons, EffMass, grid_profile, N, init_band, vs_profile, mass_density_profile, optical_deformation_profile, acoustic_deformation_profile, E_phonon_profile = prf.device_profiles()
EffMass = np.array(EffMass, dtype=np.float64)

# grid_profile = np.round(L/a).astype(int)
# N = sum(grid_profile)

@njit(cache=True)
def f(Energy, Electrode, EFl, EFr):
    kBT = kB*T

    if (Electrode == "left"):
        F=m0*meff[0]*kBT/(2*np.pi*hbar**2)*np.log(1+np.exp(-(Energy-EFl)/(kBT)))
    elif (Electrode == "right"):
        F=m0*meff[-1]*kBT/(2*np.pi*hbar**2)*np.log(1+np.exp(-(Energy-EFr)/(kBT)))

    return(F)

@njit(cache=True)
def k(El, Er):
    kl = 0
    kr = 0

    kl = (1/a)*np.arccos(complex(-El/(2*tl)))
    kr = (1/a)*np.arccos(complex(-Er/(2*tr)))

    if ((kl.real < 0) or (kl.imag < 0  and np.abs(kl.imag) > 1E-6)):
        kl = -kl
    if ((kr.real < 0) or (kr.imag < 0  and np.abs(kr.imag) > 1E-6)):
        kr = -kr

    return kl, kr

@njit(cache=True)
def sigmas(kl, kr):

    SigmaL = -tdl * np.exp(1j * kl * a) * (1/tl) * tld
    SigmaR = -tdr * np.exp(1j * kr * a) * (1/tr) * trd

    return(SigmaL, SigmaR)

@njit(cache=True)
def H_mat(V_device):
    H = np.zeros((2, N))

    for i in range(N - 1):
        H[0, i] = -t0 * (2 / (EffMass[i] + EffMass[i+1]))

    for i in range(N):
        if i == 0:
            t_right = -H[0, 0]
            H[1, i] = 2 * t_right + V_device[i]
        elif i == N - 1:
            t_left = -H[0, N-2]
            H[1, i] = 2 * t_left + V_device[i]
        else:
            t_left = -H[0, i-1]
            t_right = -H[0, i]
            H[1, i] = t_left + t_right + V_device[i]

    return H

@njit(cache=True)
def custom_I(factor):
    I = np.zeros((2,N), dtype=np.complex128)
    for i in range(N):
        I[1,i] = factor
    
    return(I)

@njit(cache=True)
def A_mat(E,V_device):
    eta = 1j* q*1E-9

    H = H_mat(V_device)
    CI = custom_I(1)

    EmH = E*CI-H

    kl, kr = k(EmH[1,0],EmH[1,N-1])
    SigmaL, SigmaR = sigmas(kl, kr)

    
    # Adding the contact self-energies to the hamiltonian
    A = EmH.copy()
    A = A + eta*CI
    A[1,0] = A[1,0] - SigmaL
    A[1,N-1] = A[1,N-1] - SigmaR

    return(A, kl, kr,  SigmaL, SigmaR)


@njit(cache=True)
def G_mat(A):

    G = np.zeros((N,2), dtype=np.complex128) # the first column is for Gx1 second column is GxN-1 
    A_back = np.copy(A) # for bach substitution, this is for the first column of G 
    A_forw = np.copy(A) # for forward substitution this is for the second column of G 

    # I have to figure out a way to parallise
    # and modify the two A matrices to give 
    # me the solution for the first and the second columns of 
    # the G matrix

    # this loop is for first modifing the matrix before back substitution
    for i in range(1, N):
        A_back[1, N-1-i] =  A_back[1, N-1-i] - ( (A_back[0, N-1-i]*A_back[0, N-1-i]) / A_back[1,N-i])
        A_forw[1,i] = A_forw[1,i] - ((A_forw[0,i-1]**2)/A_forw[1,i-1])

    # let me first put the initial elements of the two columns of G 
    G[0,0] = 1/A_back[1,0]
    G[N-1,1] = 1/A_forw[1,N-1]


    # mow this loop is for doing the back substitution
    for i in range(1,N):
        G[i,0] = -1*A_back[0,i-1]*G[i-1,0]/A_back[1,i]
        G[N-1-i,1] = -1*A_forw[0,N-1-i]*G[N-i,1]/A_forw[1,N-1-i]

    # this should be it. I will have to build another function to verify this inveser calculator.

    return(G)

@njit(cache=True)
def n_spec(SigmaL,SigmaR,G, Energy, EFl, EFr):

    FermiDirac = f(Energy, 'left', EFl, EFr)
    SigmaInL = -2*SigmaL.imag*FermiDirac

    FermiDirac = f(Energy, 'right', EFl, EFr)
    SigmaInR = -2*SigmaR.imag*FermiDirac

    Gn = np.zeros(N)

    # for i in range(N):
    #     Gn[i] = (np.abs(G[i,0])**2)*SigmaInL + (np.abs(G[i,1])**2)*SigmaInR

    Gn = (np.abs(G[:, 0])**2) * SigmaInL + (np.abs(G[:, 1])**2) * SigmaInR

    n = (2/(2*np.pi*a))*(Gn)
    
    return(n)


@njit(cache=True)
def n_conc(eDensity_E_x, dE):
    ElectronConc = np.sum(eDensity_E_x, axis=0)
    ElectronConc = ElectronConc*dE

    return(ElectronConc)

# @njit(cache=True)
# def I_spec(SigmaL, SigmaR, G, Energy, EFl, EFr):
#
#     FermiDirac = f(Energy, 'left', EFl, EFr)
#     SigmaInL = -2*SigmaL.imag*FermiDirac
#
#     FermiDirac = f(Energy, 'right', EFl, EFr)
#     SigmaInR = -2*SigmaR.imag*FermiDirac
#
#     Gg = np.zeros(N-1, dtype=np.complex128)
#
#     for i in range(N-1):
#         Gg[i] = (G[i,0]*SigmaInL*np.conjugate(G[i+1,0])) - (G[i+1,0]*SigmaInL*np.conjugate(G[i,0])) + (G[i,1]*SigmaInR*np.conjugate(G[i+1,1])) - (G[i+1,1]*SigmaInR*np.conjugate(G[i,1]))
#
#     return Gg

@njit(cache=True)
def Transmission(SigmaL, SigmaR, G):

    Gamma_L = -2*SigmaL.imag
    Gamma_R = -2*SigmaR.imag

    T_E = Gamma_L*Gamma_R*(np.abs(G[0,1])**2)

    return T_E

@njit(cache=True)
def J(T_E, dE, NE, Energy, EFl, EFr):

    sum = 0.0
    for i in range(NE):
        sum = sum + T_E[i]*(f(Energy[i], 'left', EFl, EFr) - f(Energy[i], 'right', EFl, EFr))

    J = 2*q/(hbar * 2 * np.pi ) * sum * dE

    return J




# @njit(cache=True)
# def I_vs_x(J_density_E_x, dE):
#     J_dens = np.sum(J_density_E_x, axis=0)
#     J_dens = J_dens*dE 
#
#     t_local = t0 * (2 / (EffMass[:-1] + EffMass[1:]))
#
#     return J_dens * 1j * q * t_local / (hbar * np.pi)
#
#     A, kl, kr, SigmaL, SigmaR = A_mat(E, Vx)
#     G = G_mat(A)
#
#     e_row = n_spec(SigmaL, SigmaR, G, E, EFl, EFr)
#     j_row = I_spec(SigmaL, SigmaR, G, E, EFl, EFr)
#
#     return e_row, j_row


@njit(parallel=True, cache=True)
def single_bias_execute(Bias, Vx):

    EFl = 0.0
    EFr = EFl - (q * Bias)

    E_min = min(np.min(Vx), EFr) - (0.1 * q)
    E_max = max(np.max(Vx), EFl) + (0.3 * q)

    # NE = 6000 
    # Energy = np.linspace(E_min, E_max, NE)
    # dE = np.abs((E_max - E_min)/NE)

    dE = q*1e-4
    NE = int(np.round((np.abs(E_max - E_min)/dE))) 
    Energy = np.linspace(E_min, E_max, NE)

    # print(f"solving for bias {Bias}")

    # eDensity_E_x = np.zeros((NE, N))
    # J_density_E_x = np.zeros((NE, N-1), dtype='complex')

    SigmaL = np.zeros(NE, dtype=np.complex128)
    SigmaR = np.zeros(NE, dtype=np.complex128)
    n_row = np.zeros((NE, N), dtype=np.float64)
    # j_row = np.zeros((NE, N-1), dtype=np.complex128)
    T_E = np.zeros(NE, dtype=np.complex128)

    # chunk_size = (1 or (NE //(num_core*4)))
    #
    # Vx_per_inst = [Vx]*NE
    # EFl_per_inst = [EFl]*NE
    # EFr_per_inst = [EFr]*NE
    #
    # with ProcessPoolExecutor() as executor:
    #     results = executor.map(worker_single_energy, Energy, Vx_per_inst, EFl_per_inst, EFr_per_inst)
    #
    #     e_density, j_density = zip(*results)
    for i in prange(NE):
        A, kl, kr, SigmaL[i], SigmaR[i] = A_mat(Energy[i], Vx)
        G = G_mat(A)

        n_row[i,:] = n_spec(SigmaL[i], SigmaR[i], G, Energy[i], EFl, EFr)
        # j_row[i,:] = I_spec(SigmaL[i], SigmaR[i], G, Energy[i], EFl, EFr)
        T_E[i] = Transmission(SigmaL[i], SigmaR[i], G)


    # eDensity_E_x = np.array(e_density)
    # J_density_E_x = np.array(j_density)

    electron_density = n_conc(n_row, dE)
    # current_density = I_vs_x(j_row, dE)
    J_density = J(T_E, dE, NE, Energy, EFl, EFr)

    return electron_density, J_density

#sdf
# x = np.linspace(0,l,N)
#
# def main_execute(Biases):
#     Bias_no = np.size(Biases)
#
#     final_current = np.zeros(Bias_no)
#
#     for V_index in range(Bias_no):
#
#         Vx = prf.potential_profile(Biases[V_index])
#
#         print(f"solving for {Biases[V_index]}")
#         print("-----")
#         electron_density, current_density = single_bias_execute(Biases[V_index], Vx)
#         # plt.plot(x/1E-9, np.real(current_density), color='red')
#         # plt.title('Raw Current Density')
#         # plt.ylabel('Current Density')
#         # plt.xlabel('position in nm')
#         # plt.ylim(0 * -1E10,4E10)
#         # plt.grid(True)
#         # plt.show()
#
#         final_current[V_index] = np.real(current_density)
#
#     return final_current
# #
# # if __name__ == "__main__":
# #
# # Vx = prf.potential_profile(0.0)
# # electron_density, current_density = single_bias_execute(0.0, Vx)
# Biases = np.linspace(0,0.5,21)
# final_current = main_execute(Biases)
# # # plt.plot(electron_density)
# # plt.plot(final_current)
# # plt.show()
#     # electron_density, current_density = single_bias_execute(0.0, Vx)
#     # plt.plot(x/1E-9,electron_density, color='orange')
# plt.plot(final_current)
# plt.grid(True)
# plt.show()


























