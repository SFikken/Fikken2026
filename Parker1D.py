from Fourier1D import fourier_cn, fourier_fw, fourier_rk
from Cattaneo1D import cattaneo_cn, cattaneo_fw, cattaneo_rk
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from matplot2tikz import save


def initial_condition(x):
    n = len(x)
    k = int(n / 100)
    return np.concatenate([1/k * np.ones(k), np.zeros(n - k)])


def second_initial_condition(x):
    return np.zeros_like(x)


def boundary_condition_left(t):
    return np.zeros_like(t)


def boundary_condition_right(t):
    return np.zeros_like(t)


bcl = boundary_condition_left
bcr = boundary_condition_right


t_tildemax = 1
dx = 0.005
dt = 0.000005
ic = initial_condition
ic2 = second_initial_condition

gridx_tilde = np.arange(0, 1, dx)
gridt_tilde = np.arange(0, t_tildemax + dt, dt)
m = len(gridt_tilde)
gridomega = np.pi ** 2 * gridt_tilde

n = 50

a_trunc, b_trunc = 1, 4.9
loc = 1
scale = 1
a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
k = stats.truncnorm.rvs(a, b, loc, scale, n)

a_trunc, b_trunc = 0, 0.05
loc = 0.02
scale = 1
a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
r = stats.truncnorm.rvs(a, b, loc, scale, n)

# composition = [(k[i], 1, r[i]) for i in range(n)]
composition = [(1, 1, 0)]

# an_sol = np.ones_like(gridomega)
# for i in range(1, 10):
#     print(i)
#     for j in range(len(gridomega)):
#         an_sol[j] += 2 * (-1)**i * np.exp(- i**2 * gridomega[j])


sol = fourier_fw(1, t_tildemax, dx, dt, ic, bcl, bcr, composition, True, 0, 0)
tempmax = sol[-1, -1] #max(sol[-1, :])# sol[-1, -1]
t = np.abs(sol[-1, :] / tempmax - 0.5).argmin()
# plt.scatter(gridomega[t], sol[-1, t] / tempmax, color='red', label='Grid point closest to V = 0.5')

# print('max', np.max(np.abs(np.maximum(an_sol, np.zeros_like(an_sol)) - sol[-1, :] / tempmax)))

skip = 100
fourier_sol = (sol[-1, :] / tempmax)
# analytic_sol = (np.maximum(an_sol, np.zeros_like(an_sol)))
plt.plot(gridomega[::skip], fourier_sol[::skip], label=r'Fourier Scheme')
# plt.plot(gridomega[::skip], analytic_sol[::skip], label='Analytic solution', linestyle=(0, (5, 5)), linewidth=2)

tau_tilde_list = [0.00125, 0.0025, 0.005, 0.01]
for tau_tilde in tau_tilde_list:
    print(f'tau_tilde = {tau_tilde}')
    sol = cattaneo_fw(1, t_tildemax, dx, dt, ic, ic2, composition, tau_tilde, True, 0, 0)
    tempmax = sol[-1, -1]#max(sol[-1, :])# sol[-1, -1]
    t = np.abs(sol[-1, :] / tempmax - 0.5).argmin()
    # plt.scatter(gridomega[t], sol[-1, t] / tempmax, color='red', label='Grid point closest to V = 0.5')
    plt.plot(gridomega[::skip], sol[-1, :][::skip] / tempmax, label=r'Cattaneo Scheme $\tilde{\tau}$ = ' + f'{tau_tilde}')
# plt.plot(gridomega[::skip], 0.5 * np.ones(m)[::skip], linestyle=(0, (5, 5)), label='V = 0.5')
# plt.vlines(1.38, 0, 1, linestyle=(0, (5, 5)), color='green', label=r'$\omega$ = 1.38')
plt.xlabel(r'$\pi^2 \tilde t$')
plt.ylabel(r'$\tilde u$')
# plt.grid()
plt.title(r'Temperature $\tilde u$ at $\tilde x = 1$ over time')
plt.legend()
plt.xlim(0, 2)
# plt.savefig('Parker')
# save('Tikz Plots/ComparisonZoomed.tex')
plt.show()