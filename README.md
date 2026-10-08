# NEGF-solver
Simulate quantum transport in RTD using NEGF formalism for Ballistc and Dissipative scenarios. 

Self consistency is also implemented in this solver.

Here, all the device parameters are contained in the `device_profiles.py` file. Edit those parameters for different configurations. 

**Note: If the `L` is changed, make sure to re-cache `QTNEGF.py` or `NEGF.py` as numba doesn't cache these files, if there are no changes made to them**

To run this solver, run the command `python3 wrapper.py`. It should start the solver for the defined device, after all the needed libraries are installed. This runs both the Ballistic and Dissipative with self consistency. Instead of using the wrapper, It is possible to directly run `QTNEGF.py` or `NEGF.py` after un-commenting the supporting code at the end of the files. These are the obtained plots

![Linear 1 drop at 0.3V](./plots/phonon_relaxation_illus.png)

*linear potential drop with phonon relaxation. T(E) Vs x and Ec Vs x at 0.3V*

![Linear 2 drop at 0.3V](./plots/ballistic_iilus.png)

*linear potential drop without phonon relaxation. T(E) Vs x and Ec Vs x at 0.3V*

![RTD1](./plots/RTD_phonon_relaxation_illus.png)

*Conduction band and Electron concentration profile, for RTD, with phonons, with self consistency at 0.25V*

![RTD2](./plots/RTD_ballistic_illus.png)

*Conduction band and Electron concentration profile, for RTD, without phonons, with self consistency at 0.25V*

![RTD3](./plots/RTD_phonon_relaxation_2.png)

*Conduction band and Electron concentration profile, for RTD, with phonons, with self consistency at 0.5V*

![RTD4](./plots/RTD_ballistic_illus_2.png)

*Conduction band and Electron concentration profile, for RTD, without phonons, with self consistency at 0.5V*
