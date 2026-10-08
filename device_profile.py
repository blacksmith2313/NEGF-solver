import numpy as np
import matplotlib.pyplot as plt
import math
from numba import njit
from numba import prange
# import NEGF as NEGF
# the device profiles are going to be defined here.
# arrays defining the material
# every thing will be in SI units

# constants in use


@njit(cache=True)
def custom_I():
    I = np.zeros((2,N), dtype=np.complex128)
    for i in range(N):
        I[1,i] = 1 

    return(I)

q = 1.6022E-19
m0 = 9.1093E-31
e = 8.85E-12
a = 2.0E-10 # grid spacing if you change this you will have to recache the code
hbar = 1.054E-34
q = 1.6022E-19
kB = 1.380649E-23
T = 300

mode = "phonon"

# L = np.array([30E-9, 10E-9, 2.2E-9, 6.4e-09 , 2.2E-9, 10E-9, 30E-9]) # length profile in m
# L = NEGF.L

# L = np.array([30E-9, 10E-9, 40.0E-9, 10E-9, 30E-9]) # length profile in m
# Na_regions = [0, 0, 0, 0, 0] # doping profile
# Nd_regions = [2E24, 1E21, 1E21, 1E21, 2E24] # doping profile
# meff    = [0.067, 0.067, 0.067, 0.067, 0.067]
# DeltaEc = [0.0*q, 0*q, 0*q, 0*q, 0.0*q]
# permittivity = [13.18*e, 13.18*e, 13.18*e, 13.18*e, 13.18*e]
# sound_velocity = [5E3, 5E3, 5E3, 5E3, 5E3] # in m/s
# mass_densities = [5320, 5320, 5320, 5320, 5320] # in kg/m3
# deformation_optical = [6E11*q, 6E11*q, 6E11*q, 6E11*q, 6E11*q] # in 
# deformation_acoustic = [7*q, 7*q, 7*q, 7*q, 7*q] # in 
# E_phonon = [0.0350*q, 0.0350*q, 0.0350*q, 0.0350*q, 0.0350*q] # in 

L = np.array([30E-9, 10E-9, 2.0E-9, 5.0E-9, 2.0E-9, 10E-9, 30E-9]) # length profile in m
Na_regions = [0, 0, 0, 0, 0, 0, 0] # doping profile
Nd_regions = [2E24, 1E21, 1E21, 1E21, 1E21, 1E21, 2E24] # doping profile
meff    = [0.067, 0.067, 0.0919, 0.067, 0.0919, 0.067, 0.067]
DeltaEc = [0.0*q, 0*q, 0.274*q, 0*q, 0.274*q, 0*q, 0.0*q]
permittivity = [09.18*e, 09.18*e, 11.01*e, 09.18*e, 11.01*e, 09.18*e, 09.18*e]
sound_velocity = [5E3, 5E3, 5.3E3, 5E3, 5.3E3, 5E3, 5E3] # in m/s
mass_densities = [5320, 5320, 4843, 5320, 4843, 5320, 5320] # in kg/m3
deformation_optical = [5E11*q, 5E11*q, 4.5E11*q, 5E11*q, 4.5E11*q, 5E11*q, 5E11*q] # in 
deformation_acoustic = [7*q, 7*q, 6.5*q, 7*q, 6.5*q, 7*q, 7*q] # in 
E_phonon = [0.0350*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0350*q] # in 

# Na_regions = [0, 0, 0, 0, 0, 0, 0, 0, 0] # doping profile
# Nd_regions = [2E24, 1E21, 1E21, 1E21, 1E21, 1E21, 1E21, 1E21, 2E24] # doping profile
# meff    = [0.067, 0.067, 0.0919, 0.067, 0.0919, 0.067, 0.0919, 0.067, 0.067]
# DeltaEc = [0.0*q, 0*q, 0.278*q, 0*q, 0.278*q, 0*q, 0.278*q, 0*q, 0.0*q]
# permittivity = [13.18*e, 13.18*e, 12.01*e, 13.18*e, 12.01*e, 13.18*e, 12.01*e, 13.18*e, 13.18*e]
# sound_velocity = [5E3, 5E3, 5.3E3, 5E3, 5.3E3, 5E3, 5.3E3, 5E3, 5E3] # in m/s
# mass_densities = [5320, 5320, 4843, 5320, 4843, 5320, 4843, 5320, 5320] # in kg/m3
# deformation_optical = [5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 5E10*q] # in 
# deformation_acoustic = [7*q, 7*q, 6.5*q, 7*q, 6.5*q, 7*q, 6.5*q, 7*q, 7*q] # in 
# E_phonon = [0.0350*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0350*q] # in 

# Na_regions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # doping profile
# Nd_regions = [2E24, 1E21, 1E21, 1E21, 1E21, 1E21, 1E21, 1E21, 1E21, 1E21, 2E24] # doping profile
# meff    = [0.067, 0.067, 0.0919, 0.067, 0.0919, 0.067, 0.0919, 0.067, 0.0919, 0.067, 0.067]
# DeltaEc = [0.0*q, 0*q, 0.278*q, 0*q, 0.278*q, 0*q, 0.278*q, 0*q, 0.278*q, 0*q, 0.0*q]
# permittivity = [13.18*e, 13.18*e, 12.01*e, 13.18*e, 12.01*e, 13.18*e, 12.01*e, 13.18*e, 12.01*e, 13.18*e, 13.18*e]
# sound_velocity = [5E3, 5E3, 5.3E3, 5E3, 5.3E3, 5E3, 5.3E3, 5E3, 5.3E3, 5E3, 5E3] # in m/s
# mass_densities = [5320, 5320, 4843, 5320, 4843, 5320, 4843, 5320, 4843, 5320, 5320] # in kg/m3
# deformation_optical = [5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 4.5E10*q, 5E10*q, 5E10*q] # in 
# deformation_acoustic = [7*q, 7*q, 6.5*q, 7*q, 6.5*q, 7*q, 6.5*q, 7*q, 6.5*q, 7*q, 7*q] # in 
# E_phonon = [0.0350*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0365*q, 0.0350*q, 0.0350*q] # in 

# L = np.array([40E-9, 120E-9, 40E-9]) # length profile in m
# Na_regions = [0, 0, 0] # doping profile
# Nd_regions = [2E24, 1E23, 2E24] # doping profile
# meff    = [0.065, 0.065, 0.065]
# DeltaEc = [0.0*q, 0*q, 0.0*q]
# permittivity = [11.7*e, 11.7*e, 11.7*e]


# L = np.array([40E-9, 60E-9, 40E-9, 60E-9, 40E-9]) # length profile in m
# Na_regions = [0, 0, 0, 0, 0] # doping profile
# Nd_regions = [2E24, 0, 1E23, 0, 2E24] # doping profile
# meff    = [0.065, 0.065, 0.065, 0.065, 0.065]
# DeltaEc = [0.0*q, 0*q, 0*q, 0*q, 0.0*q]
# permittivity = [11.7*e, 11.7*e, 11.7*e, 11.7*e, 11.7*e]

Nc_prf = 2*((meff[0]*m0*kB*T/(2*3.14*(hbar**2)))**(1.5))#9.8E23


grid_profile = np.round(L/a).astype(int)
N = sum(grid_profile)

def device_profiles():
    Na = np.zeros(N)
    Nd = np.zeros(N)
    epsilons = np.zeros(N)
    m_eff = np.zeros(N)
    init_band = np.zeros(N)
    vs_profile = np.zeros(N)
    mass_density_profile = np.zeros(N)
    optical_deformation_profile = np.zeros(N)
    acoustic_deformation_profile = np.zeros(N)
    E_phonon_profile = np.zeros(N)


    start = 0

    for i in range(np.size(grid_profile)):

        end = grid_profile[i] + start

        Na[start:end] = Na_regions[i]
        Nd[start:end] = Nd_regions[i]
        epsilons[start:end] = permittivity[i]
        m_eff[start:end] = meff[i]
        init_band[start:end] = DeltaEc[i]
        vs_profile[start:end] = sound_velocity[i]
        mass_density_profile[start:end] = mass_densities[i]
        optical_deformation_profile[start:end]  = deformation_optical[i]
        acoustic_deformation_profile[start:end] = deformation_acoustic[i]
        E_phonon_profile[start:end] = E_phonon[i]

        start = end

    return Na, Nd, epsilons, m_eff, grid_profile, N, init_band, vs_profile, mass_density_profile, optical_deformation_profile, acoustic_deformation_profile, E_phonon_profile

Na, Nd, epsilons, EffMass, grid_profile, N, init_band, vs_profile, mass_density_profile, optical_deformation_profile, acoustic_deformation_profile, E_phonon_profile = device_profiles()

# grid_profile = np.round(L/a).astype(int)
# plt.plot(grid_profile)
# plt.show()

@njit(cache=True)
def self_energy_constants():

    D_AP = np.zeros(N)
    D_OP = np.zeros(N)
    N_b = np.zeros(N)


    for i in range(N):
        if (i == N-1):
            D_OP[i] = (((optical_deformation_profile[i-1]+optical_deformation_profile[i])/2)**2) * hbar * hbar /(2 * ((mass_density_profile[i-1] + mass_density_profile[i])/2)*((E_phonon_profile[i-1]+E_phonon_profile[i])/2)*(a))
            D_AP[i] = (((acoustic_deformation_profile[i-1]+acoustic_deformation_profile[i])/2)**2) * kB * T /( ((mass_density_profile[i-1] + mass_density_profile[i])/2)*(((vs_profile[i-1]+vs_profile[i])/2)**2)*(a ))
            N_b[i] = 1 / (math.exp(((E_phonon_profile[i-1]+E_phonon_profile[i])/2)/(kB*T)) - 1)
        else:
            D_OP[i] = (((optical_deformation_profile[i]+optical_deformation_profile[i+1])/2)**2)  * hbar * hbar /(2 * ((mass_density_profile[i] + mass_density_profile[i+1])/2)*((E_phonon_profile[i]+E_phonon_profile[i+1])/2)*(a))
            D_AP[i] = (((acoustic_deformation_profile[i] +acoustic_deformation_profile[i+1])/2)**2) * kB * T /( ((mass_density_profile[i] + mass_density_profile[i+1])/2)*(((vs_profile[i]+vs_profile[i+1])/2)**2)*(a ))
            N_b[i] = 1 / (math.exp(((E_phonon_profile[i]+E_phonon_profile[i+1])/2)/(kB*T)) - 1)

    return D_OP, D_AP, N_b

D_op, D_ap, n_b = self_energy_constants()


def potential_profile(bias):

    Nc = Nc_prf
    r = Nd_regions[0] / Nc

    eta = np.log(r) + r/np.sqrt(8) + r**2/(48*np.sqrt(2)) + r**3/768 + r**4/(12288*np.sqrt(2))
    # eta = 0

    param = 0#.09E-20
    Ec_lead_eq =  -(kB * T * eta + param)

    # E_c = DeltaEc - q * V
    Ec = np.zeros(N)
    start = 0
    for i in range(np.size(grid_profile)):
        end = grid_profile[i] + start
        Ec[start:end] = DeltaEc[i]
        start = end

    V_L = np.ones(grid_profile[0]) * (  Ec_lead_eq)

    V_R = np.ones(grid_profile[-1]) * (-q * bias + ( Ec_lead_eq))

    active_nodes = N - grid_profile[0] - grid_profile[-1]
    V_active = np.linspace(0.0 , -q * bias, active_nodes)

    V_bias = np.concatenate((V_L, V_active, V_R))

    return Ec + V_bias, Ec

# l = np.sum(L)
# x = np.linspace(0,l,N)
# Vx = potential_profile(0.0)
# plt.plot(x/1e-9,Vx/q)
# plt.xlabel("postion in nm")
# plt.ylabel("Energy in ev")
# plt.title("intial conduction band")
# plt.grid(True)
# plt.show()
