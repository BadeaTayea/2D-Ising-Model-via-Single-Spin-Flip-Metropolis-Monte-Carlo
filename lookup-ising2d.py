import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Create custom output directory
output_dir = "task-4"
os.makedirs(output_dir, exist_ok=True)

# Global parameters
L = 10                   # lattice linear size
N = L * L                # total number of spins
J = 1                    # coupling constant
Tc = 2.269               # theoretical critical temperature
beta_theo = 1 / Tc       # theoretical inverse critical temperature (~0.441)

def nn_sum(x, i, j, L=10):
    """
    Computes the sum of nearest-neighbor spins for site (i,j)
    using periodic boundary conditions.
    """
    return x[(i+1) % L, j] + x[(i-1) % L, j] + x[i, (j+1) % L] + x[i, (j-1) % L]

def total_energy(x, L=10):
    """
    Computes the total energy of the spin configuration x.
    Only bonds to the right and bottom are summed to avoid double counting.
    """
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += -J * x[i, j] * (x[(i+1) % L, j] + x[i, (j+1) % L])
    return energy

def move(x, M, E, b, L=10, lookup=None):
    """
    Performs one Monte Carlo move using the Metropolis algorithm.
    
    Parameters:
        x : 2D array (spin configuration)
        M : total magnetization
        E : total energy
        b : inverse temperature
        L : linear system size
        lookup : (optional) dictionary mapping possible ΔE values to the 
                 acceptance probability (to avoid repeated exponential evaluations)
        
    Returns:
        Updated x, M, and E.
    """
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    s = x[i, j]
    h = nn_sum(x, i, j, L)
    deltaE = 2 * J * s * h
    if lookup is not None:
        # Use the precomputed lookup table for acceptance probability.
        # (For the 2d Ising model with 4 neighbors, deltaE ∈ {-8, -4, 0, 4, 8}.)
        if deltaE in lookup:
            R = lookup[deltaE]
        else:
            R = 1.0 if deltaE <= 0 else np.exp(-b * deltaE)
    else:
        if deltaE <= 0:
            R = 1.0
        else:
            R = np.exp(-b * deltaE)
    if np.random.rand() < R:
        x[i, j] *= -1    # spin is flipped
        M += -2 * s      # magnetization is updated
        E += deltaE      # energy is updated
    return x, M, E

# Define beta values for the simulation.
beta_vals = np.linspace(0.1, 0.8, 10)
Nthermalization = int(1e6)  # number of thermalization steps
Nsample = 5000             # number of measurements
Nsubsweep = 10 * N         # number of moves between measurements

# Arrays for observables
M_arr = np.zeros_like(beta_vals)    # average absolute magnetization (per spin)
E_arr = np.zeros_like(beta_vals)    # average energy (per spin)
M_err = np.zeros_like(beta_vals)    # standard deviation magnetization
E_err = np.zeros_like(beta_vals)    # standard deviation energy
chi_arr = np.zeros_like(beta_vals)  # magnetic susceptibility
cv_arr = np.zeros_like(beta_vals)   # heat capacity

# Loop over each inverse temperature
for t, b in enumerate(beta_vals):
    print('Running for inverse temperature =', b)
    
    # Precompute acceptance probabilities for possible ΔE values.
    # For the 2d Ising model (4 neighbors), possible ΔE values are -8, -4, 0, 4, 8.
    lookup = {}
    for deltaE in [-8, -4, 0, 4, 8]:
        if deltaE <= 0:
            lookup[deltaE] = 1.0
        else:
            lookup[deltaE] = np.exp(-b * deltaE)
    
    # Generate a random initial configuration.
    x = np.random.choice([-1, 1], size=(L, L))
    M = np.sum(x)
    E = total_energy(x, L)
    
    # Thermalization steps.
    for _ in range(Nthermalization):
        x, M, E = move(x, M, E, b, L, lookup)
    
    # Arrays for recording measurements.
    M_data_signed = np.zeros(Nsample)
    M_data_abs = np.zeros(Nsample)
    E_data = np.zeros(Nsample)
    
    M_data_signed[0] = M / N
    M_data_abs[0] = np.abs(M) / N
    E_data[0] = E / N
    
    # Measurement loop.
    for n in tqdm(range(1, Nsample)):
        for _ in range(Nsubsweep):
            x, M, E = move(x, M, E, b, L, lookup)
        M_data_signed[n] = M / N
        M_data_abs[n] = np.abs(M) / N
        E_data[n] = E / N
    
    # Compute averages and fluctuations.
    M_arr[t] = np.mean(M_data_abs)
    E_arr[t] = np.mean(E_data)
    M_err[t] = np.std(M_data_abs)
    E_err[t] = np.std(E_data)
    chi_arr[t] = b * N * (np.mean(M_data_signed**2) - np.mean(M_data_signed)**2)
    cv_arr[t] = b**2 * N * (np.mean(E_data**2) - np.mean(E_data)**2)

# Inference of the effective beta (transition point) from each observable.
dM_dbeta = np.abs(np.gradient(M_arr, beta_vals))
index_M = np.argmax(dM_dbeta)
beta_inferred_M = beta_vals[index_M]

dE_dbeta = np.abs(np.gradient(E_arr, beta_vals))
index_E = np.argmax(dE_dbeta)
beta_inferred_E = beta_vals[index_E]

index_chi = np.argmax(chi_arr)
beta_inferred_chi = beta_vals[index_chi]

index_cv = np.argmax(cv_arr)
beta_inferred_cv = beta_vals[index_cv]

# Plotting the results with inferred beta markers.

# Magnetization plot.
plt.figure()
plt.errorbar(beta_vals, M_arr, yerr=M_err, fmt='o-', capsize=5, label='Data', color='tab:cyan')
plt.axvline(x=beta_inferred_M, color='tab:olive', linestyle='--', label=r'$\beta_{inf}$ = %.3f' % beta_inferred_M)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle |M| \rangle$ (per spin)')
plt.title(r'Magnetization' + "\n" + r'($\beta_{inf}$ = %.3f, $\beta_{theo}$ = %.3f)' % (beta_inferred_M, beta_theo))
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'M.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'M.png'), dpi=300)

# Energy plot.
plt.figure()
plt.errorbar(beta_vals, E_arr, yerr=E_err, fmt='o-', capsize=5, label='Data', color='tab:cyan')
plt.axvline(x=beta_inferred_E, color='tab:olive', linestyle='--', label=r'$\beta_{inf}$ = %.3f' % beta_inferred_E)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle E \rangle$ (per spin)')
plt.title(r'Energy' + "\n" + r'($\beta_{inf}$ = %.3f, $\beta_{theo}$ = %.3f)' % (beta_inferred_E, beta_theo))
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'E.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'E.png'), dpi=300)

# Magnetic susceptibility plot.
plt.figure()
plt.plot(beta_vals, chi_arr, 'o-', label='Data', color='tab:cyan')
plt.axvline(x=beta_inferred_chi, color='tab:olive', linestyle='--', label=r'$\beta_{inf}$ = %.3f' % beta_inferred_chi)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\chi$')
plt.title(r'Magnetic Susceptibility' + "\n" + r'($\beta_{inf}$ = %.3f, $\beta_{theo}$ = %.3f)' % (beta_inferred_chi, beta_theo))
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'chi.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'chi.png'), dpi=300)

# Heat capacity plot.
plt.figure()
plt.plot(beta_vals, cv_arr, 'o-', label='Data', color='tab:cyan')
plt.axvline(x=beta_inferred_cv, color='tab:olive', linestyle='--', label=r'$\beta_{inf}$ = %.3f' % beta_inferred_cv)
plt.xlabel(r'$\beta$')
plt.ylabel(r'$C_V$')
plt.title(r'Heat Capacity' + "\n" + r'($\beta_{inf}$ = %.3f, $\beta_{theo}$ = %.3f)' % (beta_inferred_cv, beta_theo))
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'Cv.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'Cv.png'), dpi=300)

plt.show()
