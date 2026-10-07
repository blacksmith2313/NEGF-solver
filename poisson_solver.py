import math 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm
import NEGF as NEGF 
# import NEGF_phonon as NEGF_ph 
import device_profile as prf
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from functools import partial
from pathlib import Path
import QTNGEF as QTNEGF
from numba import njit
from numba import prange

folder = Path("results")
folder.mkdir(exist_ok=True)

# here i have to include all the functions
# so basically, have a potential distribution from 
# device_profile for a particular bias, input that 
# in the single_bias_execute from NEGF get the electron density
# from the function input that in a function made here
# get the new potential distribution, convergence check
# and then loop if failed else post proceesing succeeded

mode = prf.mode

@njit(cache=True)
def custom_I():
    I = np.zeros((2,N), dtype=np.complex128)
    for i in range(N):
        I[1,i] = 1 

    return(I)

Na, Nd, epsilons, EffMass, grid_profile, N, init_band, vs_profile, mass_density_profile, optical_deformation_profile, acoustic_deformation_profile, E_phonon_profile = prf.device_profiles()

l = np.sum(prf.L)
L = prf.L

confinement_characteristics = L[2:np.size(L)-2]

meff = prf.meff
a = prf.a
L = prf.L 

grid_profile = np.round(L/a).astype(int)
N = sum(grid_profile)

m0 = prf.m0
hbar = 1.054E-34
q = 1.6022E-19
kB = 1.38E-23
T = 300
Nc_prf = prf.Nc_prf

# Nc_3d = 2*((meff[0]*m0*kB*T/(2*3.14*(hbar**2)))**(1.5))
def solve_poisson(n_3D, V_old, Ec, bias, index):
    rho = q * (Nd - Na - n_3D)
    

    phi_old = -(V_old - Ec) / q
    # plt.plot(phi_old)
    # plt.show()

    gummel_factor = (q**2 * n_3D) / (kB * T)

    constant_matrix = np.zeros((N, N))
    V_new = np.zeros(N)

    for i in range(1, N - 1):
        V_new[i] = -rho[i] * (a**2) - gummel_factor[i] * phi_old[i] * (a**2)

    for i in range(1, N - 1):
        eps_left = (epsilons[i] + epsilons[i-1]) / 2
        eps_right = (epsilons[i] + epsilons[i+1]) / 2

        constant_matrix[i, i-1] = eps_left
        constant_matrix[i, i+1] = eps_right
        constant_matrix[i, i]   = -(eps_left + eps_right) - gummel_factor[i] * (a**2)

    constant_matrix[0, 0] = -1.0
    constant_matrix[0, 1] = 1.0
    V_new[0] = 0 #-phi_L

    constant_matrix[N-1, N-1] = -1.0
    constant_matrix[N-1, N-2] = 1.0
    V_new[N-1] = 0 #-bias * q - ( V_lead_eq/q)#phi_R

    # plt.plot(V_new)
    # plt.show()

    constant_matrix_sparse = sp.csr_matrix(constant_matrix)
    phi_new = spla.spsolve(constant_matrix_sparse, V_new)
 
    # plt.plot(phi_new)
    # plt.show()

    V_electrostatic = -q * phi_new
    V_total = V_electrostatic + init_band 

    if (index < 20):
        alpha = 0.15
    elif (index%10 == 0 or index%10 == 1):
        alpha = 1.0 
    else:
        alpha = 0.1

    V_new_f = (alpha * V_total) + ((1 - alpha) * V_old)

    # plt.plot(V_new_f)
    # plt.show()

    param = 5E-4 * q #-1.6E-10 * math.cbrt(Nd[0]) * q 

    return V_new_f #+ param 

x = np.linspace(0,l,N)

def self_consistency_solve(bias):

    V_old, Ec = prf.potential_profile(bias)


    for i in range(501):

        if (mode == 'ballistic'):
            electron_density, _ , current_density_new, spectrum, transmission, E_longitudinal = QTNEGF.single_bias_execute(bias, V_old, mode)
            # current_density = np.sum(current_density)/np.size(current_density)
            
        elif (mode == 'phonon'):
            electron_density, _ , current_density_new, spectrum, transmission, E_longitudinal = QTNEGF.single_bias_execute(bias, V_old, mode)

        current_density = np.mean(np.abs(current_density_new))
        transmission = np.abs(transmission) 
        spectrum = np.abs(spectrum)

        V_new = solve_poisson(electron_density, V_old, Ec, bias,i)

        diff = np.max(np.abs(V_new - V_old))
        print("--------")
        print(f"bias {bias:.2f} | current iteration {i}")
        print(f"Max deviation {diff/q}")
        print("--------")

        if (1E-4*q > diff):

            print("Convergence reached")
            print("||||||||")
                
            # file = folder / f"potentials_{bias:.2f}.dat"
            # np.savetxt(file, V_new)
            #
            # file = folder / f"electron_concentration_{bias:.2f}.dat"
            # np.savetxt(file, electron_density)
            #
            print(current_density)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            ax1.plot(x/1E-9,V_new/q, color='red', label='Potential Energy')
            ax1.contourf(x/1E-9, E_longitudinal/q, transmission, levels=1000, cmap='plasma') 
            ax1.set_title('Self-Consistent Profiles')
            ax1.set_ylabel('Potential Energy in ev')
            ax1.set_xlabel('position in nm')
            ax1.grid(True)

            # ax2.plot(x/1E-9,V_new/q, color='red', label='Potential Energy')
            # ax2.contourf(x/1E-9, E_longitudinal/q, spectrum, level=1000, cmap='plasma') 
            # ax2.set_title('Self-Consistent Profiles')
            # ax2.set_ylabel('Potential Energy in ev')
            # ax2.set_xlabel('position in nm')
            # ax2.grid(True)
            ax2.plot(x/1E-9,electron_density, color='orange', label='Electron Density')
            ax2.set_xlabel('position in nm')
            ax2.set_ylabel('Electron Density ($m^{-3}$)')
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
            return V_new, electron_density, current_density
        # elif (i%1 == 0 ): # and i != 0):
        #     fig, (ax1, ax2 ) = plt.subplots(2, 1, figsize=(8, 8))
        #
        #     ax1.plot(x/1E-9,V_new/q, color='red', label='Potential Energy')
        #     ax1.contourf(x/1E-9, E_longitudinal/q, transmission, level=1000, cmap='plasma') 
        #     ax1.set_title('Self-Consistent Profiles')
        #     ax1.set_ylabel('Potential Energy in ev')
        #     ax1.set_xlabel('position in nm')
        #     ax1.grid(True)
        #
        #     ax2.plot(x/1E-9,electron_density, color='orange', label='Electron Density')
        #     ax2.set_xlabel('position in nm')
        #     ax2.set_ylabel('Electron Density ($m^{-3}$)')
        #     ax2.grid(True)
        #
        #     # ax3.plot(c/1E-9, np.real(current_density), color='red')
        #     # ax3.set_ylabel('Current Density')
        #     # ax3.set_xlabel('position in nm')
        #     # ax3.set_ylim(0 * -1E10,4E10)
        #     # ax3.grid(True)
        #
        #     # plt.plot(Vx_poisson)
            # plt.tight_layout()
            # plt.show()
        elif (i == 500 and i != 0):
            print("convergence not reached")
            return V_new, electron_density, current_density



        V_old = V_new


Biases = np.linspace(0.0,0.5,11)
c = np.linspace(0,l,N-1)
def main_run(Biases):
    Bias_no = np.size(Biases)

    final_current = np.zeros(Bias_no)
    for i in range(len(Biases)):
        Vx_poisson, electron_density, current_density = self_consistency_solve(Biases[i])
    
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        #
        # ax1.plot(x/1E-9,Vx_poisson/q, color='blue', label='Potential Energy')
        # ax1.set_title('Self-Consistent Profiles')
        # ax1.set_ylabel('Potential Energy in ev')
        # ax1.set_xlabel('position in nm')
        # ax1.grid(True)
        #
        # ax2.plot(x/1E-9,electron_density, color='orange', label='Electron Density')
        # ax2.set_xlabel('position in nm')
        # ax2.set_ylabel('Electron Density ($m^{-3}$)')
        # ax2.grid(True)

        # ax3.plot(c/1E-9, np.real(current_density), color='red')
        # ax3.set_ylabel('Current Density')
        # ax3.set_xlabel('position in nm')
        # ax3.set_ylim(0 * -1E10,4E10)
        # ax3.grid(True)

        # if (mode == 'ballistic'):
        #     final_current[i] = np.real(np.sum(current_density)/np.size(current_density))
        final_current[i] = np.real(current_density)

        # plt.plot(Vx_poisson)
        # plt.tight_layout()
        # plt.show()
    return final_current

if __name__ == "__main__":

    cc_suffix = "".join(f"_{param}" for param in confinement_characteristics)

    if (mode == 'ballistic'):
        cc_suffix = cc_suffix + 'ballistic'
    elif (mode == 'phonon'):
        cc_suffix = cc_suffix + 'phonon'
    final_current = main_run(Biases)
    data = np.column_stack((Biases, final_current))
    file = folder / f"I-V_characteristics_{Biases[0]:.2f}_{Biases[np.size(Biases)-1]:.2f}_{np.size(Biases)}{cc_suffix}.dat"
    np.savetxt(file, data)
    # plt.plot(Biases, final_current, label="transfer characteristics", color="red")
    # plt.xlabel("Biases (V)")
    # plt.ylabel("Current Density (Am-2)")
    # plt.grid(True)
    # plt.show()
