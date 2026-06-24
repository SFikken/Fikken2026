import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from matplot2tikz import save


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
        x = gridx[index] - dx / 2
        plt.vlines(x, 0, 4.2, linestyle=(0, (5, 5)), label=f'Interface x = {x:.5f}')
    return np.array(betas), R


def numerical_sol(L, tmax, dx, dt, ic1, ic2, composition=[(1, 1, 0)], tau=0.01):
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

    # solve
    for j in range(2, m):
        print('t = ', round(gridt[j], 6))
        t = gridt[j]
        h0 = 0
        hL = 0
        C1 = (h0 * (T0 - sol[0][j - 1]) - tau * h0 * (sol[0][j - 1] - sol[0][j - 2]) / dx + qtilde(t)) / k1
        C2 = -(hL * (sol[n - 1][j - 1] - T0) + tau * hL * (sol[n - 1][j - 1] - sol[n - 1][j - 2]) / dx) / k1
        sol[0, j] = (sol[0, j - 1]
                     + dt / dx**2 * (beta[0] * (sol[1, j - 1] - sol[0, j - 1])
                                     - k[0] * (sol[0, j - 1] - sol[1, j - 1] - 2 * dx * C1))
                     - tau * (-2 * sol[0, j - 1] + sol[0, j - 2]) / dt) / (1 + tau / dt)
        laplacian = (beta[1:] * (sol[2:, j - 1] - sol[1:-1, j - 1]) - beta[:-1] * (
                sol[1:-1, j - 1] - sol[:-2, j - 1])) / dx ** 2
        sol[1:-1, j] = (sol[1:-1, j - 1]
                        + dt * laplacian - tau * (-2 * sol[1:-1, j - 1] + sol[1:-1, j - 2]) / dt) / (1 + tau / dt)

        sol[n - 1, j] = (sol[n - 1, j - 1]
                         + dt / dx **2 * (beta[-1] * (sol[n - 2, j - 1] - sol[n - 1, j - 1])
                                      - k[-1] * (sol[n - 1, j - 1] - sol[n - 2, j - 1] - 2 * dx * C2))
                         - tau * (-2 * sol[n - 1, j - 1] + sol[n - 1, j - 2]) / dt) / (1 + tau / dt)
    return sol


def qtilde(t):
    qt = 7000 * t / (0.001**2) * np.exp(-t / 0.001)
    qprimet = 7000 / (0.001**2) * np.exp(-t / 0.001) * (1 - t / 0.001)
    return tau * qprimet + qt


if __name__ == '__main__':
    L = 0.002
    tmax = 0.1
    dx = 0.00001
    dt = 0.0000001
    alpha = 9.1766e-5
    # Fourier, tau = 0
    tau = 0.00 #0.001
    C = 0
    T0 = 0
    k1 = 222

    n = 1000
    np.random.seed(1)

    a_trunc, b_trunc = 2e-5, 1.447e-4
    loc = 8.267e-5
    scale = 0.01
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    k = stats.truncnorm.rvs(a, b, loc, scale, n)

    a_trunc, b_trunc = 0, 0.5
    loc = 0.25
    scale = 0.1
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    r = stats.truncnorm.rvs(a, b, loc, scale, n)

    # composition = [(k[i], 1, r[i]) for i in range(n)]
    # composition = [(9e-5, 1, 0)]
    composition = [
        (9.1766e-5, 1, 0),
        (13e-5, 2, 0),
        (5e-5, 3, 0)
    ]


    def first_initial_condition(x):
        return T0 * np.ones_like(x)


    def second_initial_condition(x):
        return np.zeros_like(x)


    ic1 = first_initial_condition
    ic2 = second_initial_condition

    gridx = np.arange(0, L + dx, dx)
    n = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)

    rear = False
    if rear:
        tmax = 0.1
        gridt = np.arange(0, tmax + dt, dt)
        tau_list = [0, 0.00025,
                    0.0005, 0.001, 0.002]
        skip = 2000
        for tau in tau_list:
            sol = numerical_sol(L, tmax, dx, dt, ic1, ic2, composition, tau)
            plt.plot(gridt[::skip], sol[-1, :][::skip] / sol[-1, -1], label=f'Tau = {tau}')

    else:
        composition = [
            (9.1766e-5, 1, 0),
            (13e-5, 2, 0),
            (5e-5, 3, 0)
        ]
        dx = 1e-6
        dt = 1e-7
        tau = 0.001
        sol = numerical_sol(L, tmax, dx, dt, ic1, ic2, composition, tau)
        # sol1 = analytical_sol(x, t)
        times = [0.0001, 0.0005, 0.0012, 0.003, 0.006, 0.01, 0.1]
        #times = [0.0004, 0.002, 0.0048, 0.012, 0.024, 0.04, 0.4]
        errors = []
        skip = 1
        gridx = np.arange(0, L + dx, dx)
        n = len(gridx)
        gridt = np.arange(0, tmax + dt, dt)
        m = len(gridt)
        for t in times:
            j = int(t / dt)
            plt.plot(gridx[::skip], sol[:, j][::skip]/1.4468, label=f't = {round(gridt[j], 4)}')
    # plt.title(f'Cattaneo Heat Equation, tau = {tau}')
    plt.title(f'Fourier Heat Equation')
    plt.legend(loc='upper right')
    plt.xlabel('$t$')
    # plt.xlim(0, 0.015)
    plt.ylabel('$T(x, t)$')
    tau_str = str(tau).replace('.', '_')
    # plt.savefig(f'Carr{tau_str}')
    save('Tikz Plots/InterfacesCattaneo.tex')
    plt.show()

