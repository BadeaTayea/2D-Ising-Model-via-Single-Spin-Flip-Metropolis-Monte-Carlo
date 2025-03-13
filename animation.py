import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
L = 50                   # Lattice linear size
J = 1                    # Coupling constant
temperatures = [1.0, 2.269, 3.0]  # Three temperatures: below, at, and above T_c
betas = [1.0 / T for T in temperatures]

def nn_sum(x, i, j, L):
    """
    Compute the sum of nearest-neighbor spins for the spin at (i, j)
    using periodic boundary conditions.
    """
    return x[(i+1) % L, j] + x[(i-1) % L, j] + x[i, (j+1) % L] + x[i, (j-1) % L]

def move(x, b, L):
    """
    Perform one single-spin update using the Metropolis algorithm.
    
    Parameters:
        x : 2D numpy array (spin configuration)
        b : inverse temperature (\beta)
        L : lattice linear size
        
    Returns:
        Updated configuration x.
    """
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    s = x[i, j]
    h = nn_sum(x, i, j, L)
    deltaE = 2 * J * s * h
    if deltaE <= 0:
        R = 1.0
    else:
        R = np.exp(-b * deltaE)
    if np.random.rand() < R:
        x[i, j] *= -1
    return x

# Initialize a configuration for each temperature
configs = [np.random.choice([-1, 1], size=(L, L)) for _ in temperatures]

# Set up the figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
titles = [f"$T = {T}$" for T in temperatures]  # Use mathtext formatting
ims = []
for ax, config, title in zip(axs, configs, titles):
    im = ax.imshow(config, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(title)
    ax.axis('off')
    ims.append(im)
fig.suptitle(f"2D Ising Model Dynamics at Different Temperatures $|$ $L = {L}$", fontsize=16)

def update(frame):
    global configs
    # For each configuration, perform several updates per frame.
    for i, b in enumerate(betas):
        for _ in range(L * L // 10):
            configs[i] = move(configs[i], b, L)
        ims[i].set_data(configs[i])
    return ims

# Output folder
output_dir = "animations"
os.makedirs(output_dir, exist_ok=True)

# Create the animation.
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

anim.save("animations/ising_animation.mp4", writer="ffmpeg", dpi=300)
anim.save("animations/ising_animation.gif", writer="imagemagick", dpi=300)

plt.show()
