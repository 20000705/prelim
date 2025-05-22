import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Task 1
# Sample sizes and corresponding absolute errors (from your output)
N_values = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
abs_errors = np.array([11.588333, 3.588333, 0.668333, 0.191667, 0.112967, 0.004263])

# Log-log plot
plt.figure(figsize=(7, 5))
plt.loglog(N_values, abs_errors, marker='o', linestyle='-', color='navy', label='Absolute Error')

# Reference line for O(1/sqrt(N)) convergence
ref_line = abs_errors[0] * (N_values[0] / N_values) ** 0.5
plt.loglog(N_values, ref_line, linestyle='--', color='gray', label=r'Ref: $\mathcal{O}(N^{-1/2})$')

# Plot aesthetics
plt.xlabel("Number of Samples (N)")
plt.ylabel("Absolute Error")
plt.title("Convergence of Monte Carlo Estimate (Absolute Error)")
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save and show
plt.savefig("Q2_1.png", dpi=300)
plt.show()

# Task 2
# Data from MPI runs
threads = [1, 2, 4, 8]
times = [4.837739, 2.430522, 1.212996, 0.590663]

# Calculate speedup
baseline_time = times[0]
speedup = [baseline_time / t for t in times]

# Create and print a table with pandas
df = pd.DataFrame({
    "Threads": threads,
    "Execution Time (s)": times,
    "Speedup": speedup
})

print(df.to_string(index=False))

# Plot both execution time and speedup
fig, ax1 = plt.subplots(figsize=(8, 5))

# Execution Time (blue line)
ax1.plot(threads, times, marker='o', linestyle='-', color='b', label="Execution Time (s)")
ax1.set_xlabel("Number of Threads")
ax1.set_ylabel("Execution Time (s)", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Speedup (red dashed line)
ax2 = ax1.twinx()
ax2.plot(threads, speedup, marker='s', linestyle='--', color='r', label="Speedup")
ax2.set_ylabel("Speedup", color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Title and grid
plt.title("Parallel Performance: Execution Time and Speedup")
ax1.grid(True, linestyle="--", alpha=0.6)

# Save and show
plt.tight_layout()
plt.savefig("Q2_2.png", dpi=300)
plt.show()