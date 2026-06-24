from Fourier1D import fourier_fw
from Cattaneo1D import betas
import numpy as np
from matplotlib import pyplot as plt
from matplot2tikz import save


def cattaneo_fw(L, tmax, dx, dt, ic1, ic2, composition=[(1, 1, 0)], tau=0.01, heat_loss=False, c=0, tinf=0):
    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)

    sol = np.zeros((n, m))
    k, R = betas(composition, n, gridx, dx)
    beta = dx / (0.5 * dx / k[:-1] + R + 0.5 * dx / k[1:])

    # initial conditions
    sol[:, 0] = ic1(gridx)
    sol[:, 1] = sol[:, 0] + dt * ic2(gridx)

    # boundary conditions
    if not heat_loss:
        for j in range(m):
            sol[0][j] = T
            sol[n - 1][j] = T

    for j in range(2, m):
        print('t = ', round(gridt[j], 6))
        if heat_loss:
            sol[0, j] = (sol[0, j - 1]
                         + dt / dx * (beta[0] * (sol[1, j - 1] - sol[0, j - 1]) / dx - c * (
                            sol[0, j - 1] - tinf)) - tau * (-2 * sol[0, j - 1] + sol[0, j - 2]) / dt) / (1 + tau / dt)
        laplacian = (beta[1:] * (sol[2:, j - 1] - sol[1:-1, j - 1]) - beta[:-1] * (
                sol[1:-1, j - 1] - sol[:-2, j - 1])) / dx ** 2
        sol[1:-1, j] = (sol[1:-1, j - 1]
                        + dt * laplacian - tau * (-2 * sol[1:-1, j - 1] + sol[1:-1, j - 2]) / dt) / (1 + tau / dt)
        if heat_loss:
            sol[n - 1, j] = (sol[n - 1, j - 1] + dt / dx * (
                        beta[-1] * (sol[n - 2, j - 1] - sol[n - 1, j - 1]) / dx - c * (
                            sol[n - 1, j - 1] - tinf)) - tau * (-2 * sol[n - 1, j - 1] + sol[n - 1, j - 2]) / dt) / (
                                        1 + tau / dt)
    return sol


def initial_condition(x):
    return 0 * np.ones_like(x)


def boundary_condition_left(t):
    return T * np.ones_like(t)


def boundary_condition_right(t):
    return T * np.ones_like(t)


def ic2(x):
    return np.zeros_like(x)


ic, bcl, bcr = initial_condition, boundary_condition_left, boundary_condition_right

# fig, (ax1, ax2) = plt.subplots(1, 2)
files70 = ['70Data.txt', '70Model.txt']
files90 = ['90Data.txt', '90Model.txt']
tmax = 1.2 #0.28
data = np.loadtxt('70Model.txt', skiprows=1, delimiter=',')
x = data[:, 0]
y = data[:, 1]
plt.plot(x * 1e3, y, label=f'Fourier Model 70°C')
data = np.loadtxt('70Data.txt', skiprows=1, delimiter=',')
x = data[:, 0]
y = data[:, 1]
plt.plot(x * 1e3, y, label=f'Measurements 70°C')
# data = np.loadtxt('90Data.txt', skiprows=1, delimiter=',')
# x = data[:, 0]
# y = data[:, 1]
# list90.append((x/15, (y - 18) / (90 - 18)))
# plt.plot(x/15, (y - 18) / (90 - 18), label=f'Data 90', marker='v', color='C1')
# data = np.loadtxt('90Model.txt', skiprows=1, delimiter=',')
# x = data[:, 0]
# y = data[:, 1]
# list90.append((x/15, (y - 18) / (90 - 18)))
# plt.plot(x/15, (y - 18) / (90 - 18), label=f'Model 90', linestyle=(0, (5, 5)), color='C1')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [°C]')
# ax1.set(xlabel='Time [s]', ylabel='Temperature [°C]')
save('Data70.tex')
plt.show()
plt.clf()


T = 1
L = 1
dx = 0.01
dt = 0.00001
alpha = 1
# noinspection LanguageDetectionInspection
composition = [(alpha, 1, 0.125) for i in range(21)] #0.125 #0.075

sol = fourier_fw(L, tmax, dx, dt, ic, bcl, bcr, composition, False)
dxcat = 0.0005 #0.0005
dtcat = dt * 1
tau = 0.025#0.018
composition = [(alpha, 1, 0.125) for i in range(21)]
sol2 = cattaneo_fw(L, tmax, dxcat, dtcat, ic, ic2, composition, tau, False)

# t = L / (alpha / tau)**0.5
gridx = np.arange(0, L + dx, dx)
n = len(gridx)
gridxcat = np.arange(0, L + dxcat, dxcat)
ncat = len(gridxcat)
gridt = np.arange(0, tmax + dt, dt)
gridtcat = np.arange(0, tmax + dtcat, dtcat)
shiftf = np.argmin(np.abs(sol[n//2, :] - 0.0001))
shiftc = np.argmin(np.abs(sol2[ncat//2, :] - 0.0001))

skip = 1000
plt.plot(gridt[:-shiftf][::skip], sol[n//2, shiftf:][::skip], label=f'Fourier Scheme')
plt.plot(gridtcat[:-shiftc][::skip], sol2[ncat//2, shiftc:][::skip], label=f'Cattaneo Scheme')
plt.legend()
plt.xlabel('tilde t')
plt.ylabel('tilde u')
plt.xlim(0, 1)
save('HydrogelPaperMySchemes.tex')
plt.show()
