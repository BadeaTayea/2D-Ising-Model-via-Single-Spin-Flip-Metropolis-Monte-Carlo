import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global lattice size and coupling constant
L = 10  # linear size of the lattice
N = L * L  # total number of spins
J = 1  # coupling constant

def nn_sum(x, i, j):
    """
    Computes the sum of the nearest-neighbor spins for the spin at position (i,j)
    using periodic boundary conditions.
    """
    # Nearest neighbors with periodic boundaries are used.
    return x[(i+1) % L, j] + x[(i-1) % L, j] + x[i, (j+1) % L] + x[i, (j-1) % L]

def total_energy(x):
    """
    Computes the total energy of the spin configuration x.
    Energy is given by E = -J * sum_<i,j> sigma_i sigma_j, where each bond is counted once.
    """
    energy = 0
    # Bonds to the right and bottom are summed to avoid double counting.
    for i in range(L):
        for j in range(L):
            energy += -J * x[i, j] * (x[(i+1) % L, j] + x[i, (j+1) % L])
    return energy

def move(x, M, E, b):
    """
    Performs one Monte Carlo move using the Metropolis algorithm.
    
    Parameters:
        x: current spin configuration (LxL array)
        M: total magnetization of configuration x
        E: total energy of configuration x
        b: inverse temperature (beta = 1/T)
        
    Returns:
        Updated spin configuration, magnetization, and energy.
    """
    # A site is selected at random.
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    s = x[i, j]
    # Local field from nearest neighbors is computed.
    h = nn_sum(x, i, j)
    # Energy difference for flipping spin (i,j) is computed.
    deltaE = 2 * J * s * h
    # Acceptance probability is computed.
    if deltaE <= 0:
        R = 1.0
    else:
        R = np.exp(-b * deltaE)
    
    # Spin flip is accepted with probability R.
    if np.random.rand() < R:
        # Spin is flipped.
        x[i, j] *= -1
        # Magnetization is updated.
        M += -2 * s  # change equals new spin minus old spin
        # Energy is updated.
        E += deltaE
    return x, M, E

# Inverse temperature values to be simulated.
beta_vals = np.linspace(0.1, 0.8, 10)
Nthermalization = int(1e6)  # number of thermalization steps
Nsample = 5000            # number of measurements
Nsubsweep = 10 * N        # number of moves between measurements

# Arrays to store observables at each beta.
M_arr = np.zeros_like(beta_vals)  # average absolute magnetization (per spin)
E_arr = np.zeros_like(beta_vals)  # average energy (per spin)
M_err = np.zeros_like(beta_vals)  # standard deviation of magnetization
E_err = np.zeros_like(beta_vals)  # standard deviation of energy
chi_arr = np.zeros_like(beta_vals)  # magnetic susceptibility
cv_arr = np.zeros_like(beta_vals)   # heat capacity

# Loop over different inverse temperatures.
for t in range(beta_vals.size):
    b = beta_vals[t]
    print('Running for inverse temperature =', b)
    
    # A random initial configuration is generated.
    x = np.random.choice([-1, 1], size=(L, L))
    # Total magnetization and energy are computed.
    M = np.sum(x)
    E = total_energy(x)
    
    # Thermalization steps are performed.
    for _ in range(Nthermalization):
        x, M, E = move(x, M, E, b)
    
    # Arrays for measurements are initialized.
    M_data_signed = np.zeros(Nsample)  # signed magnetization per spin
    M_data_abs = np.zeros(Nsample)     # absolute magnetization per spin
    E_data = np.zeros(Nsample)         # energy per spin
    
    # Initial measurement is recorded.
    M_data_signed[0] = M / N
    M_data_abs[0] = np.abs(M) / N
    E_data[0] = E / N
    
    # Measurement loop over samples.
    for n in tqdm(range(1, Nsample)):
        # A number of subsweeps are performed to decorrelate the samples.
        for _ in range(Nsubsweep):
            x, M, E = move(x, M, E, b)
        # Measurements are recorded.
        M_data_signed[n] = M / N
        M_data_abs[n] = np.abs(M) / N
        E_data[n] = E / N
    
    # Averages and fluctuations are computed.
    M_arr[t] = np.mean(M_data_abs)   # average absolute magnetization per spin
    E_arr[t] = np.mean(E_data)         # average energy per spin
    M_err[t] = np.std(M_data_abs)
    E_err[t] = np.std(E_data)
    # Susceptibility is computed using fluctuation–dissipation theorem:
    # chi = beta * N * ( <m^2> - <m>^2 ) where m is magnetization per spin.
    chi_arr[t] = b * N * (np.mean(M_data_signed**2) - np.mean(M_data_signed)**2)
    # Heat capacity is computed as:
    # Cv = beta^2 * N * ( <E^2> - <E>^2 ).
    cv_arr[t] = b**2 * N * (np.mean(E_data**2) - np.mean(E_data)**2)

# Plotting the observables as functions of beta.

# Plot absolute magnetization.
plt.figure()
plt.errorbar(beta_vals, M_arr, yerr=M_err, fmt='o-', capsize=5)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle |M| \rangle$ (per spin)')
plt.title('Magnetization')
plt.savefig('M.pdf')

# Plot energy.
plt.figure()
plt.errorbar(beta_vals, E_arr, yerr=E_err, fmt='o-', capsize=5)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle E \rangle$ (per spin)')
plt.title('Energy')
plt.savefig('E.pdf')

# Plot magnetic susceptibility.
plt.figure()
plt.plot(beta_vals, chi_arr, 'o-')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\chi$')
plt.title('Magnetic Susceptibility')
plt.savefig('chi.pdf')

# Plot heat capacity.
plt.figure()
plt.plot(beta_vals, cv_arr, 'o-')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$C_V$')
plt.title('Heat Capacity')
plt.savefig('Cv.pdf')

plt.show()
