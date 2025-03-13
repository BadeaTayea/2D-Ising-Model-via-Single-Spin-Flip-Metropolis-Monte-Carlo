import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# Create custom output directory
output_dir = "task-4"
os.makedirs(output_dir, exist_ok=True)

# Global coupling constant
J = 1

def nn_sum(x, i, j, L):
    """
    Computes the sum of the nearest-neighbor spins for site (i, j)
    using periodic boundary conditions.
    
    Parameters:
        x : 2D numpy array (spin configuration)
        i, j : integer indices of the spin
        L : linear system size of the lattice
        
    Returns:
        Sum of the four nearest neighbors.
    """
    return x[(i+1) % L, j] + x[(i-1) % L, j] + x[i, (j+1) % L] + x[i, (j-1) % L]

def total_energy(x, L):
    """
    Computes the total energy of configuration x.
    Each bond is counted only once.
    
    Parameters:
        x : 2D numpy array (spin configuration)
        L : linear system size of the lattice
        
    Returns:
        Total energy.
    """
    energy = 0
    for i in range(L):
        for j in range(L):
            # Count bonds to right and bottom to avoid double counting.
            energy += -J * x[i, j] * (x[(i+1) % L, j] + x[i, (j+1) % L])
    return energy

def create_lookup(b, L):
    """
    Creates a lookup dictionary for the acceptance probabilities for possible ΔE values.
    For the 2D Ising model (4 neighbors), possible ΔE values are -8, -4, 0, 4, 8.
    """
    lookup = {}
    for deltaE in [-8, -4, 0, 4, 8]:
        if deltaE <= 0:
            lookup[deltaE] = 1.0
        else:
            lookup[deltaE] = np.exp(-b * deltaE)
    return lookup

def move(x, M, E, b, L, lookup=None):
    """
    Performs one Monte Carlo move using the Metropolis algorithm.
    
    Parameters:
        x : 2D numpy array (spin configuration)
        M : current total magnetization
        E : current total energy
        b : inverse temperature (beta)
        L : linear system size
        lookup : (optional) dictionary mapping ΔE values to acceptance probabilities
        
    Returns:
        Updated x, M, and E.
    """
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    s = x[i, j]
    h = nn_sum(x, i, j, L)
    deltaE = 2 * J * s * h
    if lookup is not None:
        # Use the precomputed acceptance probability if available.
        R = lookup.get(deltaE, 1.0 if deltaE <= 0 else np.exp(-b * deltaE))
    else:
        if deltaE <= 0:
            R = 1.0
        else:
            R = np.exp(-b * deltaE)
    if np.random.rand() < R:
        x[i, j] *= -1   # Spin is flipped.
        M += -2 * s     # Update magnetization.
        E += deltaE     # Update energy.
    return x, M, E

def run_simulation(L, beta_vals, Nthermalization, Nsample, Nsubsweep):
    """
    Runs the Monte Carlo simulation for a given system size L and an array of inverse temperatures.
    
    Parameters:
        L : linear system size.
        beta_vals : 1D numpy array of inverse temperatures.
        Nthermalization : number of thermalization steps.
        Nsample : number of measurement samples.
        Nsubsweep : number of moves between measurements.
        
    Returns:
        Arrays of observables: magnetization, energy, their errors, susceptibility, and heat capacity.
    """
    N = L * L
    M_arr   = np.zeros_like(beta_vals)
    E_arr   = np.zeros_like(beta_vals)
    M_err   = np.zeros_like(beta_vals)
    E_err   = np.zeros_like(beta_vals)
    chi_arr = np.zeros_like(beta_vals)
    cv_arr  = np.zeros_like(beta_vals)
    
    for t, b in enumerate(beta_vals):
        print(f'Running simulation for L={L}, beta={b:.3f}')
        # Create lookup table for current beta and system size.
        lookup = create_lookup(b, L)
        
        # Generate a random initial configuration.
        x = np.random.choice([-1, 1], size=(L, L))
        M = np.sum(x)
        E = total_energy(x, L)
        
        # Thermalization loop.
        for _ in range(Nthermalization):
            x, M, E = move(x, M, E, b, L, lookup)
        
        # Arrays for recording measurements.
        M_data_signed = np.zeros(Nsample)
        M_data_abs    = np.zeros(Nsample)
        E_data        = np.zeros(Nsample)
        
        M_data_signed[0] = M / N
        M_data_abs[0]    = np.abs(M) / N
        E_data[0]        = E / N
        
        # Measurement loop with subsweeps between samples.
        for n in tqdm(range(1, Nsample), desc=f"L={L}, beta={b:.3f}"):
            for _ in range(Nsubsweep):
                x, M, E = move(x, M, E, b, L, lookup)
            M_data_signed[n] = M / N
            M_data_abs[n]    = np.abs(M) / N
            E_data[n]        = E / N
        
        # Compute averages and fluctuations.
        M_arr[t] = np.mean(M_data_abs)
        E_arr[t] = np.mean(E_data)
        M_err[t] = np.std(M_data_abs)
        E_err[t] = np.std(E_data)
        # Magnetic susceptibility via fluctuation-dissipation theorem.
        chi_arr[t] = b * N * (np.mean(M_data_signed**2) - np.mean(M_data_signed)**2)
        # Heat capacity from energy fluctuations.
        cv_arr[t] = b**2 * N * (np.mean(E_data**2) - np.mean(E_data)**2)
    
    return M_arr, E_arr, M_err, E_err, chi_arr, cv_arr

# ---------------------------
# Main simulation for multiple system sizes
# ---------------------------
# Choose system sizes.
L_values = [5, 10, 15, 20, 25]
beta_vals = np.linspace(0.1, 0.8, 10)  # inverse temperatures

# Adjust simulation parameters for quick runs on small systems.
Nthermalization = int(1e4)  # number of thermalization steps
Nsample         = 500      # number of measurement samples

data = {}  # Dictionary to store simulation data for each system size

# Lists to store peak data for susceptibility and heat capacity.
peak_beta_chi = []   # Inferred beta at peak of susceptibility.
peak_beta_cv  = []   # Inferred beta at peak of heat capacity.
peak_chi      = []   # Peak susceptibility values.
peak_cv       = []   # Peak heat capacity values.

# Run simulations for each system size.
for L in L_values:
    Nsubsweep = 10 * L * L  # Subsweeps scale with system size.
    M_arr, E_arr, M_err, E_err, chi_arr, cv_arr = run_simulation(L, beta_vals, Nthermalization, Nsample, Nsubsweep)
    data[L] = {'beta_vals': beta_vals, 'M_arr': M_arr, 'E_arr': E_arr,
               'M_err': M_err, 'E_err': E_err, 'chi_arr': chi_arr, 'cv_arr': cv_arr}
    
    # Infer the peak positions.
    index_chi = np.argmax(chi_arr)
    beta_peak_chi = beta_vals[index_chi]
    peak_beta_chi.append(beta_peak_chi)
    peak_chi.append(chi_arr[index_chi])
    
    index_cv = np.argmax(cv_arr)
    beta_peak_cv = beta_vals[index_cv]
    peak_beta_cv.append(beta_peak_cv)
    peak_cv.append(cv_arr[index_cv])

# ---------------------------
# Write peak data to an output CSV file.
# ---------------------------
csv_filename = os.path.join(output_dir, "peak_data.csv")
with open(csv_filename, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["L", "beta_peak_chi", "beta_peak_cv", "peak_chi", "peak_cv"])
    for i, L in enumerate(L_values):
        writer.writerow([L, peak_beta_chi[i], peak_beta_cv[i], peak_chi[i], peak_cv[i]])
print(f"Peak data has been written to '{csv_filename}'.")

# ---------------------------
# Plot Set 1: Observable vs. Inverse Temperature for different system sizes
# ---------------------------
# Magnetization vs beta.
plt.figure()
for L in L_values:
    plt.errorbar(data[L]['beta_vals'], data[L]['M_arr'], yerr=data[L]['M_err'],
                 fmt='o-', capsize=5, label=f'L = {L}')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle |M| \rangle$ (per spin)')
plt.title("Magnetization vs. Inverse Temperature")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'M_vs_beta.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'M_vs_beta.png'), dpi=300)

# Energy vs beta.
plt.figure()
for L in L_values:
    plt.errorbar(data[L]['beta_vals'], data[L]['E_arr'], yerr=data[L]['E_err'],
                 fmt='o-', capsize=5, label=f'L = {L}')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\langle E \rangle$ (per spin)')
plt.title("Energy vs. Inverse Temperature")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'E_vs_beta.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'E_vs_beta.png'), dpi=300)


# Magnetic susceptibility vs beta.
plt.figure()
for L in L_values:
    plt.plot(data[L]['beta_vals'], data[L]['chi_arr'], 'o-', label=f'L = {L}')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\chi$')
plt.title("Magnetic Susceptibility vs. Inverse Temperature")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'chi_vs_beta.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'chi_vs_beta.png'), dpi=300)

# Heat capacity vs beta.
plt.figure()
for L in L_values:
    plt.plot(data[L]['beta_vals'], data[L]['cv_arr'], 'o-', label=f'L = {L}')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$C_V$')
plt.title("Heat Capacity vs. Inverse Temperature")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'Cv_vs_beta.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'Cv_vs_beta.png'), dpi=300)

# ---------------------------
# Plot Set 2: Peak Analysis
# ---------------------------
# Plot inferred peak positions (beta_peak) as a function of system size.
plt.figure()
plt.plot(L_values, peak_beta_chi, 'o-', label='Susceptibility peak')
plt.plot(L_values, peak_beta_cv, 'o-', label='Heat capacity peak')
plt.xlabel('System size L')
plt.ylabel(r'Inferred $\beta_{\rm peak}$')
plt.title("Peak Positions vs. System Size")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, 'beta_peak_vs_L.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'beta_peak_vs_L.png'), dpi=300)


# Plot peak heights versus system size on a log-log scale.
plt.figure()
plt.loglog(L_values, peak_chi, 'o-', label='Peak susceptibility')
plt.loglog(L_values, peak_cv, 'o-', label='Peak heat capacity')
plt.xlabel('System size L')
plt.ylabel('Peak value')
plt.title("Peak Heights vs. System Size (log-log scale)")
plt.legend()
plt.grid(True, which="both", ls="--")


plt.savefig(os.path.join(output_dir, 'peak_heights_vs_L.pdf'), dpi=300)
plt.savefig(os.path.join(output_dir, 'peak_heights_vs_L.png'), dpi=300)

plt.show()
