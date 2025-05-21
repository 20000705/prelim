import numpy as np
import glob
import matplotlib.pyplot as plt

files = sorted(glob.glob("result_rank0.txt"))
x_all, u_all = [], []
for file in files:
    data = np.loadtxt(file)
    x_all.extend(data[:, 0])
    u_all.extend(data[:, 1])

x_all, u_all = np.array(x_all), np.array(u_all)
idx = np.argsort(x_all)
x_all, u_all = x_all[idx], u_all[idx]

plt.plot(x_all, u_all, label="u(x, t=20)")
plt.xlabel("x")
plt.ylabel("u")
plt.title("1D Heat Equation with α(x) on Non-uniform Grid")
plt.grid(True)
plt.legend()
plt.show()