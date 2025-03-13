import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Global parameters for the test
L = 10
J = 1
b = 0.5  # Choose an arbitrary beta value for testing

def nn_sum(x, i, j, L):
    return x[(i+1) % L, j] + x[(i-1) % L, j] + x[i, (j+1) % L] + x[i, (j-1) % L]

def move(x, M, E, b, L, lookup=None):
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    s = x[i, j]
    h = nn_sum(x, i, j, L)
    deltaE = 2 * J * s * h
    if lookup is not None:
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
        x[i, j] *= -1
        M += -2 * s
        E += deltaE
    return x, M, E

def create_lookup(b, L):
    lookup = {}
    # For 2D Ising (4 neighbors), possible deltaE values are typically -8, -4, 0, 4, 8.
    for deltaE in [-8, -4, 0, 4, 8]:
        if deltaE <= 0:
            lookup[deltaE] = 1.0
        else:
            lookup[deltaE] = np.exp(-b * deltaE)
    return lookup

def measure_move_time(use_lookup, iterations=100000):
    # Initialize a random configuration.
    x = np.random.choice([-1, 1], size=(L, L))
    M = np.sum(x)
    E = 0  # E is not important here; we only compare timing.
    lookup = create_lookup(b, L) if use_lookup else None
    start_time = time.time()
    for _ in range(iterations):
        x, M, E = move(x, M, E, b, L, lookup)
    return time.time() - start_time

# Number of iterations for timing.
iterations = 500000

time_with = measure_move_time(use_lookup=True, iterations=iterations)
time_without = measure_move_time(use_lookup=False, iterations=iterations)

# Prepare the output text.
output_text = (
    f"Time with lookup: {time_with:.4f} s over {iterations} iterations\n"
    f"Time without lookup: {time_without:.4f} s over {iterations} iterations\n"
)

print(output_text)

# Create output folder "task-4" and write output text file.
output_folder = "task-4"
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, "timing_output.txt")
with open(output_filename, "w") as f:
    f.write(output_text)

# Plot the comparison.
labels = ['With Lookup', 'Without Lookup']
times = [time_with, time_without]

plt.figure()
plt.bar(labels, times, color=['tab:cyan', 'tab:orange'])
plt.ylabel("Time (s)")
plt.title(f"Comparison of move() Execution Time (${iterations}$ iterations)")
plt.grid(True)

plt.savefig(os.path.join(output_folder, "comparison_time.pdf"), dpi=300)
plt.savefig(os.path.join(output_folder, "comparison_time.png"), dpi=300)

plt.show()
