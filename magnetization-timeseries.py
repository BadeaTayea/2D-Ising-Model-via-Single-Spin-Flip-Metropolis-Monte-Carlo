import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parse command-line arguments.
parser = argparse.ArgumentParser(description="Simulate 2D Ising model and plot M vs. time.")
parser.add_argument("--L", type=int, default=10, help="Lattice linear size (default: 10)")
args = parser.parse_args()
L = args.L

# Create output folder "task-5"
output_dir = "task-5"
os.makedirs(output_dir, exist_ok=True)

# Global simulation parameters
N = L * L                 # Total number of spins
T = 2.0                   # Temperature (T < T_c)
beta = 1.0 / T            # Inverse temperature
Nthermalization = 1000    # Number of sweeps for thermalization
Nmeasurement = 10000      # Number of sweeps for recording magnetization

def nn_sum(x, i, j, L):
    r"""Compute the sum of nearest-neighbor spins for the spin at \((i,j)\)
    using periodic boundary conditions.
    """
    return x[(i+1) % L, j] + x[(i-1) % L, j] + x[i, (j+1) % L] + x[i, (j-1) % L]

def total_energy(x, L):
    r"""Compute the total energy of the configuration \(x\).
    Only bonds to the right and bottom are counted.
    """
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += -x[i, j] * (x[(i+1) % L, j] + x[i, (j+1) % L])
    return energy

def move(x, M, E, b, L):
    r"""Perform a single Monte Carlo spin update using the Metropolis algorithm.
    
    Parameters:
        x : 2D numpy array (spin configuration)
        M : total magnetization
        E : total energy
        b : inverse temperature \(\beta\)
        L : lattice linear size
        
    Returns:
        Updated \(x\), \(M\), and \(E\).
    """
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    s = x[i, j]
    h = nn_sum(x, i, j, L)
    deltaE = 2 * s * h
    if deltaE <= 0:
        R = 1.0
    else:
        R = np.exp(-b * deltaE)
    if np.random.rand() < R:
        x[i, j] *= -1   # Flip spin
        M += -2 * s     # Update magnetization
        E += deltaE     # Update energy
    return x, M, E

# Initialize a random configuration.
x = np.random.choice([-1, 1], size=(L, L))
M = np.sum(x)
E = total_energy(x, L)

# Thermalize the system (each sweep = N spin updates).
for sweep in range(Nthermalization):
    for _ in range(N):
        x, M, E = move(x, M, E, beta, L)

# Record the total magnetization M at each sweep.
M_time = np.zeros(Nmeasurement)
for sweep in tqdm(range(Nmeasurement), desc="Measuring M vs. time"):
    for _ in range(N):
        x, M, E = move(x, M, E, beta, L)
    M_time[sweep] = M

# Plot the time series of magnetization.
plt.figure()
plt.plot(np.arange(Nmeasurement), M_time, lw=0.5, color='tab:cyan')
plt.xlabel("Simulation Time (Sweeps)")
plt.ylabel("Total Magnetization $M$")
plt.title("Time Series of $M$ at $T=2.0$ (< $T_c$), $L=%d$" % L)
plt.grid(True)

# Save the plot with L indicated in the file name.
filename = os.path.join(output_dir, "M_vs_time_L%d.png" % L)
plt.savefig(filename, dpi=300)
plt.show()
