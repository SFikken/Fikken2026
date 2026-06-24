import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplot2tikz import save

dx = np.logspace(-5, -1, 500)
dt = np.logspace(-5, -1, 500)
dx, dt = np.meshgrid(dx, dt)

alpha = 1
tau = 0.01

fourier1 = (alpha*dt/dx**2 <= 0.5)
cattaneo1 = (alpha*dt/dx**2 <= tau/dt + 0.5)
fourier4 = (alpha*dt/dx**2 <= 0.6963)

sqrt_disc = np.sqrt(1 + 4 * tau * alpha * -4 / dx**2 + 0j)     # force complex-safe sqrt

r_plus = (-1 + sqrt_disc) / (2 * tau)
r_minus = (-1 - sqrt_disc) / (2 * tau)

z_plus = dt * r_plus
z_minus = dt * r_minus

R = lambda z: 1 + z + 0.5*z**2 + (1/6)*z**3 + (1/24)*z**4

cattaneo4 = (
    (np.abs(R(dt*r_plus)) <= 1) &
    (np.abs(R(dt*r_minus)) <= 1)
)
# print((abs(R(z_plus)) <= 1), (abs(R(z_minus)) <= 1))

# plt.figure(figsize=(6,6))

# plt.contourf(dx, dt, fourier1, levels=[0.5, 1], alpha=0.25, colors=['red'])
# plt.contourf(dx, dt, fourier4, levels=[0.5, 1], alpha=0.25, colors=['blue'])

plt.contourf(dx, dt, cattaneo1, levels=[0.5, 1], alpha=0.25, colors=['red'])
plt.contourf(dx, dt, cattaneo4, levels=[0.5, 1], alpha=0.25, colors=['blue'])

legend_patches = [
    mpatches.Patch(color='red', alpha=0.25, label='First-Order Cattaneo'),
    mpatches.Patch(color='blue', alpha=0.25, label='Fourth-Order Cattaneo'),
]

plt.legend(handles=legend_patches, loc='upper right')

plt.xscale('log')
plt.yscale('log')
# plt.gca().set_aspect('equal')
plt.xlabel('dx')
plt.ylabel('dt')
plt.grid(True)

save('StabilityCattaneo.tex')
plt.show()