import numpy as np
from matplotlib import pyplot as plt
import math
from scipy.integrate import quad
from scipy.special import iv
from matplot2tikz import save


def numerical_sol(L, tmax, dx, dt, tau):
    n = int(L / dx) + 1
    m = int(tmax / dt) + 1
    gridx = np.linspace(0, L, n)
    gridt = np.linspace(0, tmax, m)

    sol = np.zeros((n, m))
    # initial conditions
    for i in range(n):
        x = gridx[i]
        sol[i][0] = T0
        sol[i][1] = T0

    # solve
    for j in range(2, m):
        print('t = ', round(gridt[j], 6))
        t = gridt[j]
        h0 = 0
        hL = 0
        C1 = -(h0 * (T0 - sol[0][j - 1]) - tau * h0 * (sol[0][j - 1] - sol[0][j - 2]) / dx + qtilde(t)) / k
        C2 = -(hL * (sol[n - 1][j - 1] - T0) + tau * hL * (sol[n - 1][j - 1] - sol[n - 1][j - 2]) / dx) / k
        sol[0][j] = (sol[0][j - 1]
                     + alpha * dt * (2 * sol[1][j - 1] - 2 * sol[0][j - 1] - 2 * dx * C1) / (dx ** 2)
                     - tau * (-2 * sol[0][j - 1] + sol[0][j - 2]) / dt) / (1 + tau / dt)
        for i in range(1, n - 1):
            sol[i][j] = (sol[i][j - 1]
                         + alpha * dt * (sol[i + 1][j - 1] - 2 * sol[i][j - 1] + sol[i - 1][j - 1]) / (dx ** 2)
                         - tau * (-2 * sol[i][j - 1] + sol[i][j - 2]) / dt) / (1 + tau / dt)
        sol[n - 1][j] = (sol[n - 1][j - 1]
                         + alpha * dt * (2 * sol[n - 2][j - 1] - 2 * sol[n - 1][j - 1] + 2 * dx * C2) / (dx ** 2)
                         - tau * (-2 * sol[n - 1][j - 1] + sol[n - 1][j - 2]) / dt) / (1 + tau / dt)
    return sol


def qtilde(t):
    qt = 7000 * t / (0.001**2) * np.exp(-t / 0.001)
    qprimet = 7000 / (0.001**2) * np.exp(-t / 0.001) * (1 - t / 0.001)
    return tau * qprimet + qt


def fn(u, x, t):
    if u**2 - tau / alpha * x**2 <= 0:
        return 0
    bessel = iv(0, (1 / (2 * tau)) * np.sqrt(u**2 - tau / alpha * x**2))
    return qtilde(t - u) * np.exp(-u / (2 * tau)) * bessel


def v(u, x, t):
    sum1, sum2 = 0, 0
    for i in range(math.floor(((alpha/tau)**0.5 * t - x)/(2 * L)) + 1):
        heavi = np.heaviside(u - (tau/alpha)**0.5 * (x + 2 * i * L), 0.5)
        a = u**2 - tau / alpha * (x + 2 * i * L)**2
        if a>=0:
            bessel = iv(0, 1 / (2 * tau) * np.sqrt(a))
        else:
            bessel = 0
        sum1 += heavi * bessel

    for i in range(math.floor(((alpha/tau)**0.5 * t + x)/(2 * L) - 1) + 1):
        heavi = np.heaviside(u - (tau/alpha)**0.5 * (2 * (i + 1) * L - x), 0.5)
        a = u**2 - tau / alpha * (2 * (i + 1) * L - x)**2
        if a >= 0:
            bessel = iv(0, 1 / (2 * tau) * np.sqrt(a))
        else:
            bessel = 0
        sum2 += heavi * bessel
    # print(np.exp(- u / (2 * tau)) * (sum1 + sum2))
    return np.exp(- u / (2 * tau)) * (sum1 + sum2)


def analytical_sol(x, t):
    if x >= (alpha / tau)**0.5 * t:
        return 0
    else:
        integral, _ = quad(lambda u: qtilde(t - u) * v(u, x, t),0, t,limit=400)
        # print(integral)
        return T0 + 1 / k * (alpha / tau)**0.5 * integral


if __name__ == '__main__':
    L = 0.002
    tmax = 0.1
    dx = 0.000001
    dt = 0.000001
    alpha = 9.1766e-5
    tau = 0.001
    neumann = True
    C = 0
    T0 = 0
    k = 222

    n = int(L / dx) + 1
    m = int(tmax / dt) + 1
    gridx = np.linspace(0, L, n)
    gridt = np.linspace(0, tmax, m)

    sol = numerical_sol(L, tmax, dx, dt, tau)
    # sol1 = analytical_sol(x, t)
    times = [0.0001, 0.0005, 0.0012, 0.003, 0.006, 0.01, 0.1]
    #times = [0.0004, 0.002, 0.0048, 0.012, 0.024, 0.04, 0.4]
    errors = []
    skip = 10
    for t in times:
        j = int(t / dt)
        line, = plt.plot(gridx[::skip], sol[:, j][::skip]/1.4468, label=f't = {round(gridt[j], 4)}')
        color = line.get_color()
        an_sol = [analytical_sol(x, gridt[j])/1.4468 for x in gridx]
        plt.plot(gridx[::skip], an_sol[::skip], label=f'analytic t = {round(gridt[j], 4)}', linestyle=(0, (5, 5)), linewidth=2, color=color)
        errors.append(np.max(np.abs(sol[:, j]/1.4468 - an_sol)))
    print(errors, np.max(errors))
    plt.title(f'Cattaneo Heat Equation, tau = {tau}')
    plt.legend(loc='upper right')
    plt.xlabel('$x$')
    plt.ylabel('$T(x, t)$')
    tau_str = str(tau).replace('.', '_')
    # plt.savefig(f'Carr{tau_str}')
    save('Tikz Plots/Carr.tex')
    plt.show()

