import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import sparse, stats


def betas(composition, n, gridx, dx):
    ratios = np.array([c[1] for c in composition])
    true_divide = n * ratios / sum(ratios)
    rounded_down_divide = np.floor(true_divide)
    total_remainder = n - sum(rounded_down_divide)
    remainders = true_divide - rounded_down_divide
    sorted_indices = np.argsort(remainders)[::-1]
    for i in range(int(total_remainder)):
        rounded_down_divide[sorted_indices[i]] += 1

    betas = []
    for i in range(len(ratios)):
        alpha = composition[i][0]
        for j in range(int(rounded_down_divide[i])):
            betas.append(alpha)

    R = np.zeros(n - 1)

    index = 0
    for i in range(len(ratios) - 1):
        index += int(rounded_down_divide[i])
        R[index - 1] = composition[i][2]
        # x = gridx[index] - dx / 2
        # plt.vlines(x, 0, 1, linestyle=(0, (5, 5)), label=f'Interface x = {x:.3f}')
    return np.array(betas), R



def cattaneo_rk(L, tmax, dx, dt, ic1, ic2, alpha=1, tau=0.01, heat_loss=False, c=0, tinf=0):
    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)

    sol = np.zeros((n, m))
    # initial conditions
    sol[:, 0] = ic1(gridx)

    # boundary conditions
    if not heat_loss:
        sol[0, :] = np.zeros(m)
        sol[n - 1, :] = np.zeros(m)

    r = alpha / (dx ** 2)
    if not heat_loss:
        A = sparse.diags(
            [r * np.ones(n - 3),
             -2 * r * np.ones(n - 2),
             r * np.ones(n - 3)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        wn = ic2(gridx)[1:-1]
        for j in range(1, m):
            un = sol[1:n - 1, j - 1]
            b = np.zeros(n - 2)
            b[0] = r * sol[0][j]
            b[n - 3] = r * sol[n - 1][j]
            k1 = dt * wn
            l1 = dt * (A * un - wn + b) / tau
            k2 = dt * (wn + 0.5 * l1)
            l2 = dt * (A * (un + 0.5 * k1) - (wn + 0.5 * l1) + b) / tau
            k3 = dt * (wn + 0.5 * l2)
            l3 = dt * (A * (un + 0.5 * k2) - (wn + 0.5 * l2) + b) / tau
            k4 = dt * (wn + l3)
            l4 = dt * (A * (un + k3) - (wn + l3) + b) / tau
            sol[1:n - 1, j] = un + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4
            wn = wn + 1 / 6 * l1 + 1 / 3 * l2 + 1 / 3 * l3 + 1 / 6 * l4
    else:
        A = sparse.diags(
            [r * np.ones(n - 1),
             -2 * r * np.ones(n),
             r * np.ones(n - 1)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        A[0, 0] -= 2 * r * c * dx
        A[0, 1] += r
        A[n - 1, n - 2] += r
        A[n - 1, n - 1] -= 2 * r * c * dx
        wn = ic2(gridx)
        for j in range(1, m):
            un = sol[:, j - 1]
            b = np.zeros(n)
            b[0] += 2 * r * c * dx * tinf
            b[n - 1] += 2 * r * c * dx * tinf
            k1 = dt * wn
            l1 = dt * (A * un - wn + b) / tau
            k2 = dt * (wn + 0.5 * l1)
            l2 = dt * (A * (un + 0.5 * k1) - (wn + 0.5 * l1) + b) / tau
            k3 = dt * (wn + 0.5 * l2)
            l3 = dt * (A * (un + 0.5 * k2) - (wn + 0.5 * l2) + b) / tau
            k4 = dt * (wn + l3)
            l4 = dt * (A * (un + k3) - (wn + l3) + b) / tau
            sol[:, j] = un + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4
            wn = wn + 1 / 6 * l1 + 1 / 3 * l2 + 1 / 3 * l3 + 1 / 6 * l4
    return sol


def cattaneo_cn(L, tmax, dx, dt, ic1, ic2, alpha=1, tau=0.01, heat_loss=False, c=0, tinf=0):
    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)

    sol = np.zeros((n, m))
    # initial conditions
    sol[:, 0] = ic1(gridx)

    uxx = np.zeros(n)
    if not heat_loss:
        uxx[1:-1] = (sol[2:, 0] - 2 * sol[1:-1, 0] + sol[:-2, 0]) / dx ** 2
    else:
        uxx[1:-1] = (sol[2:, 0] - 2 * sol[1:-1, 0] + sol[:-2, 0]) / dx ** 2
        uxx[0] = 2 * (sol[1, 0] - sol[0, 0] - c * dx * (sol[0, 0] - tinf)) / dx ** 2
        uxx[-1] = 2 * (sol[n - 2, 0] - sol[n - 1, 0] - c * dx * (sol[n - 1, 0] - tinf)) / dx ** 2

    sol[:, 1] = sol[:, 0] + dt * ic2(gridx) + 0.5 * dt ** 2 * (alpha * uxx - ic2(gridx)) / tau

    # boundary conditions
    if not heat_loss:
        sol[0, :] = np.zeros(m)
        sol[n - 1, :] = np.zeros(m)

    r = dt * alpha / dx ** 2
    q = 2 * tau / dt
    if not heat_loss:
        A = sparse.diags(
            [-r * np.ones(n - 3),
             (1 + 2 * r + q) * np.ones(n - 2),
             -r * np.ones(n - 3)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        B1 = 2 * q * sparse.identity(n - 2, format='csr')
        B2 = sparse.diags(
            [r * np.ones(n - 3),
             (1 - 2 * r - q) * np.ones(n - 2),
             r * np.ones(n - 3)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        for j in range(2, m):
            # all x indexes one to the left (index 0 is x_1)
            b = B1 @ sol[1:n - 1, j - 1] + B2 @ sol[1:n - 1, j - 2]
            b[0] -= -r * sol[0, j]
            b[n - 3] -= -r * sol[n - 1, j]
            sol[1:n - 1, j] = sparse.linalg.spsolve(A, b)
    else:
        A = sparse.diags(
            [-r * np.ones(n - 1),
             (1 + 2 * r + q) * np.ones(n),
             -r * np.ones(n - 1)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        A[0, 0] += 2 * r * c * dx
        A[0, 1] -= r
        A[n - 1, n - 2] -= r
        A[n - 1, n - 1] += 2 * r * c * dx
        B1 = 2 * q * sparse.identity(n, format='csr')
        B2 = sparse.diags(
            [r * np.ones(n - 1),
             (1 - 2 * r - q) * np.ones(n),
             r * np.ones(n - 1)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        B2[0, 0] -= 2 * r * c * dx
        B2[0, 1] += r
        B2[n - 1, n - 2] += r
        B2[n - 1, n - 1] -= 2 * r * c * dx
        for j in range(2, m):
            # all x indexes one to the left (index 0 is x_1)
            b = B1 @ sol[:, j - 1] + B2 @ sol[:, j - 2]
            b[0] += 4 * r * dx * c * tinf
            b[n - 1] += 4 * r * dx * c * tinf
            sol[:, j] = sparse.linalg.spsolve(A, b)
    return sol


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
            sol[0][j] = 0
            sol[n - 1][j] = 0

    for j in range(2, m):
        # if __name__ == '__main__':
        #     print('t = ', round(gridt[j], 6))
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


def analytic_sol(x, t, tau):
    u = 0
    for n in range(1, 2):
        dn = 1 - 4 * tau * np.pi ** 2 * n ** 2
        if n == 1:
            fn = 1
        else:
            fn = 0
        if n % 2 == 0:
            gn = 0
        else:
            gn = 0  # 8 / (np.pi ** 3 * n ** 3)
        if dn > 0:
            An = fn * (dn ** 0.5 + 1) / 2 + gn * tau / (dn ** 0.5)
            Bn = fn * (dn ** 0.5 - 1) / 2 - gn * tau / (dn ** 0.5)
            Tn = An * np.exp(t * (dn ** 0.5 - 1) / (2 * tau)) + Bn * np.exp(-t * (dn ** 0.5 + 1) / (2 * tau))
        else:
            Cn = (2 * tau * gn + fn) / ((-dn) ** 0.5)
            Dn = fn
            Tn = np.exp(-t / (2 * tau)) * (
                    Cn * np.sin((t * (-dn) ** 0.5) / (2 * tau)) + Dn * np.cos((t * (-dn) ** 0.5) / (2 * tau)))
        u += Tn * np.sin(np.pi * n * x) / (1 - 4 * tau * np.pi ** 2)**0.5
    return u


if __name__ == '__main__':
    L = 1
    tmax = 0.05
    dx = 0.001
    dt = 0.01
    alpha = 10
    tau = 0.01
    heat_loss = True
    c = 0
    tinf = 0
    # conductivity, ratio, contact resistance on right
    # composition = [
    #     (1, 2, 0),
    #     (1, 1, 0),
    #     (1, 3, 0)
    # ]

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

    composition = [(k[i], 1, r[i]) for i in range(n)]
    composition = [(1, 1, 0)]


    def first_initial_condition(x):
        return np.sin(np.pi * x)


    def second_initial_condition(x):
        return np.zeros_like(x)


    ic1 = first_initial_condition
    ic2 = second_initial_condition

    n = int(L / dx) + 1
    m = int(tmax / dt) + 1
    gridx = np.linspace(0, L, n)
    gridt = np.linspace(0, tmax, m)

    # sol1 = cattaneo_fw(L, tmax, dx, dt, ic1, ic2, composition, tau, heat_loss, c)
    sol2 = cattaneo_cn(L, tmax, dx, dt, ic1, ic2, alpha, tau, heat_loss, c)
    # sol3 = cattaneo_rk(L, tmax, dx, 0.000001, ic1, ic2, alpha, tau, heat_loss, c)
    for k in range(6):
        j = int(k * (m - 1) / 5)
        plt.plot(gridx, sol2[:, j], label=f'cn t = {round(gridt[j], 3)}')
        # plt.plot(gridx, sol2[:, j], label=f'fw t = {round(gridt[j], 3)}', linestyle=(0, (5, 5)), linewidth=2)
        # plt.plot(gridx, sol3[:, j], label=f'fw t = {round(gridt[j], 3)}', linestyle=(0, (8, 8)), linewidth=3)
        # plt.plot(gridx, analytic_sol(gridx, gridt[j], tau), label=f'analytic t = {round(gridt[j], 3)}', linestyle=(0, (5, 5)), linewidth=2)
    plt.title(f'Cattaneo Heat Equation, tau = {tau}')
    plt.legend(loc='upper right')
    plt.xlabel('$x$')
    plt.ylabel('$T(x, t)$')
    tau_str = str(tau).replace('.', '_')
    # if neumann:
    #     plt.savefig(f'Cattaneo_Neumann{tau_str}')
    # else:
    #     plt.savefig(f'Cattaneo{tau_str}')
    plt.show()
