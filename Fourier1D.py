import numpy as np
from scipy import sparse, stats
from matplotlib import pyplot as plt
import math


def semi_discrete_exact(x, t, dx, alpha):
    lambda_h = -alpha * 4 * np.sin(np.pi * dx / 2) ** 2 / dx ** 2
    return np.sin(np.pi * x) * np.exp(lambda_h * t)


def analytic_sol(x, t):
    u = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t)
    return u


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


def fourier_rk(L, tmax, dx, dt, ic, bcl, bcr, alpha=1, heat_loss=False, c=0, tinf=0):
    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)
    sol = np.zeros((n, m))

    # initial condition
    sol[:, 0] = ic(gridx)

    # dirichlet boundary conditions
    if not heat_loss:
        sol[0, :] = bcl(gridt)
        sol[n - 1, :] = bcr(gridt)

    r = alpha / (dx ** 2)
    if not heat_loss:
        A = sparse.diags(
            [r * np.ones(n - 3),
             -2 * r * np.ones(n - 2),
             r * np.ones(n - 3)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        for j in range(1, m):
            t = gridt[j - 1]
            b = np.zeros(n - 2)
            un = sol[1:n - 1, j - 1]
            b[0], b[n - 3] = r * bcl(t), r * bcr(t)
            k1 = dt * (A * un + b)
            b[0], b[n - 3] = r * bcl(t + 0.5 * dt), r * bcr(t + 0.5 * dt)
            k2 = dt * (A * (un + 0.5 * k1) + b)
            k3 = dt * (A * (un + 0.5 * k2) + b)
            b[0], b[n - 3] = r * bcl(t + dt), r * bcr(t + dt)
            k4 = dt * (A * (un + k3) + b + c * k3)
            sol[1:n - 1, j] = un + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4
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
        for j in range(1, m):
            un = sol[:, j - 1]
            b = np.zeros(n)
            b[0] += 2 * r * c * dx * tinf
            b[n - 1] += 2 * r * c * dx * tinf
            k1 = dt * (A * un + b)
            k2 = dt * (A * (un + 0.5 * k1) + b)
            k3 = dt * (A * (un + 0.5 * k2) + b)
            k4 = dt * (A * (un + k3) + b)
            sol[:, j] = un + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4
    return sol


def fourier_cn(L, tmax, dx, dt, ic, bcl, bcr, alpha=1, heat_loss=False, c=0, tinf=0):
    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)
    sol = np.zeros((n, m))

    # initial condition
    sol[:, 0] = ic(gridx)

    # dirichlet boundary conditions
    if not heat_loss:
        sol[0, :] = bcl(gridt)
        sol[n - 1, :] = bcr(gridt)

    r = dt * alpha / (2 * dx ** 2)
    if not heat_loss:
        A = sparse.diags(
            [-r * np.ones(n - 3),
             (1 + 2 * r) * np.ones(n - 2),
             -r * np.ones(n - 3)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        B = sparse.diags(
            [r * np.ones(n - 3),
             (1 - 2 * r) * np.ones(n - 2),
             r * np.ones(n - 3)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        for j in range(1, m):
            # all x indexes one to the left (index 0 is x_1)
            b = B @ sol[1:n - 1, j - 1]
            b[0] += r * sol[0, j]
            b[n - 3] += r * sol[n - 1, j]
            sol[1:n - 1, j] = sparse.linalg.spsolve(A, b)
    else:
        A = sparse.diags(
            [-r * np.ones(n - 1),
             (1 + 2 * r) * np.ones(n),
             -r * np.ones(n - 1)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        A[0, 0] += 2 * r * c * dx
        A[0, 1] -= r
        A[n - 1, n - 2] -= r
        A[n - 1, n - 1] += 2 * r * c * dx  # should be minus
        B = sparse.diags(
            [r * np.ones(n - 1),
             (1 - 2 * r) * np.ones(n),
             r * np.ones(n - 1)],
            offsets=[-1, 0, 1],
            format='csr'
        )
        B[0, 0] -= 2 * r * c * dx
        B[0, 1] += r
        B[n - 1, n - 2] += r
        B[n - 1, n - 1] -= 2 * r * c * dx  # should be plus
        for j in range(1, m):
            b = B @ sol[:, j - 1]
            b[0] += 4 * r * dx * c * tinf
            b[n - 1] += 4 * r * dx * c * tinf  # should be minus
            sol[:, j] = sparse.linalg.spsolve(A, b)
    return sol


def fourier_fw(L, tmax, dx, dt, ic, bcl, bcr, composition=[(1, 1, 0)], heat_loss=False, c=0, tinf=0):
    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)

    sol = np.zeros((n, m))
    k, R = betas(composition, n, gridx, dx)

    beta = dx / (0.5 * dx / k[:-1] + R + 0.5 * dx / k[1:])
    # beta = beta / np.sum(beta * dx)
    # initial condition
    sol[:, 0] = ic(gridx)

    # dirichlet boundary conditions
    if not heat_loss:
        sol[0, :] = bcl(gridt)
        sol[n - 1, :] = bcr(gridt)

    for j in range(1, m):
        print(gridt[j])
        if heat_loss:
            sol[0, j] = sol[0, j - 1] + dt / dx * (
                        beta[0] * (sol[1, j - 1] - sol[0, j - 1]) / dx - c * (sol[0, j - 1] - tinf))

        laplacian = (beta[1:] * (sol[2:, j - 1] - sol[1:-1, j - 1]) - beta[:-1] * (
                    sol[1:-1, j - 1] - sol[:-2, j - 1])) / dx ** 2
        sol[1:-1, j] = sol[1:-1, j - 1] + dt * laplacian
        if heat_loss:
            sol[n - 1, j] = sol[n - 1, j - 1] + dt / dx * (
                        beta[-1] * (sol[n - 2, j - 1] - sol[n - 1, j - 1]) / dx - c * (sol[n - 1, j - 1] - tinf))
    return sol


if __name__ == '__main__':
    L = 1
    tmax = 0.05
    dx = 0.001
    dt = 0.001
    alpha = 10
    heat_loss = True
    c = 0
    tinf = 0
    # conductivity, ratio, contact resistance on right
    # composition = [
    #     (3, 2, 0.5),
    #     (1, 1, 0.2),
    #     (5, 3, 0)
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
    composition2 = [(1, 1, 0)]

    def initial_condition(x):
        return np.sin(np.pi * x)


    def boundary_condition_left(t):
        return np.zeros_like(t)


    def boundary_condition_right(t):
        return np.zeros_like(t)


    ic = initial_condition
    bcl = boundary_condition_left
    bcr = boundary_condition_right

    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    # print(betas(composition, n))
    sol1 = fourier_cn(L, tmax, dx, dt, ic, bcl, bcr, alpha, heat_loss, c, tinf)
    # sol3 = fourier_fw(L, tmax, dx, dt, ic, bcl, bcr, composition2, heat_loss, c, tinf)
    # sol3 = fourier_fw(L, tmax, dx, dt, ic, bcl, bcr, composition, heat_loss, c, tinf)
    for k in range(6):
        gridt = np.arange(0, tmax + dt, dt)
        m = len(gridt)
        j = int(k * (m - 1) / 5)
        plt.plot(gridx, sol1[:, j], label=f't = {round(gridt[j], 3)}')
        # plt.plot(gridx, sol1[:, j], label=f'cn t = {round(gridt[j], 3)}', linestyle=(0, (8, 8)), linewidth=3)
        # plt.plot(gridx, sol2[:, j], label=f'fw t = {round(gridt[j], 3)}', linestyle=(0, (5, 5)), linewidth=2)
        # gridt = np.arange(0, tmax + 0.001, 0.001)
        # m = len(gridt)
        # j = int(k * (m-1) / 5)
        # plt.plot(gridx, sol2[:, j], label=f'dt 0.001 t = {round(gridt[j], 3)}')
        # gridt = np.arange(0, tmax + 0.0001, 0.0001)
        # m = len(gridt)
        # j = int(k * (m-1) / 5)
        # plt.plot(gridx, sol3[:, j], label=f'dt 0.0001 t = {round(gridt[j], 3)}')
        # plt.plot(gridx, [analytic_sol(x, gridt[j]) for x in gridx], label=f'analytic t = {round(gridt[j], 3)}',
        #          linestyle=(0, (5, 5)), linewidth=2)
    plt.title('Fourier Heat Equation, with interface resistance')
    plt.legend(loc='upper right')
    plt.xlabel('$x$')
    plt.ylabel('$T(x, t)$')
    # plt.savefig('Fourier')
    plt.show()
