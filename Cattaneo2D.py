import numpy as np
from matplotlib import pyplot as plt
import math


def numerical_sol(Lx, Ly, tmax, dx, dy, dt, tau, alpha, neumann, C=0):
    gridx = np.arange(0, Lx + dx, dx)
    nx = len(gridx)
    gridy = np.arange(0, Ly + dy, dy)
    ny = len(gridx)
    gridt = np.arange(0, tmax + dt, dt)
    m = len(gridt)
    sol = np.zeros((nx, ny, m))

    # initial condition
    for ix in range(nx):
        for iy in range(ny):
            x = gridx[ix]
            y = gridy[iy]
            sol[ix][iy][0] = math.sin(math.pi * x) * math.sin(math.pi * y)
            sol[ix][iy][1] = math.sin(math.pi * x) * math.sin(math.pi * y)

    # boundary conditions
    if not neumann:
        for j in range(m):
            for ix in range(nx):
                sol[ix][0][j] = 0
                sol[ix][ny - 1][j] = 0
            for iy in range(ny):
                sol[0][iy][j] = 0
                sol[nx - 1][iy][j] = 0

    # solve
    for j in range(2, m):
        print(f't = {gridt[j]}')
        if neumann:
            C1x = [C * sol[0][iy][j - 1] - 0 for iy in range(ny)]
            C2x = [-C * sol[nx - 1][iy][j - 1] - 0 for iy in range(ny)]
            C1y = [C * sol[ix][0][j - 1] - 0 for ix in range(nx)]
            C2y = [-C * sol[ix][ny - 1][j - 1] - 0 for ix in range(nx)]
            for iy in range(ny):
                sol[0][iy][j] = (sol[0][iy][j - 1] + dt * alpha * (
                            2 * sol[1][iy][j - 1] - 2 * sol[0][iy][j - 1] - 2 * dx * C1x[iy]) / (
                                         dx ** 2) - tau * (-2 * sol[0][iy][j - 1] + sol[0][iy][j - 2]) / dt) / (
                                            1 + tau / dt)

        for ix in range(1, nx - 1):
            if neumann:
                sol[ix][0][j] = (sol[ix][0][j - 1] + dt * alpha * (
                            2 * sol[ix][1][j - 1] - 2 * sol[ix][0][j - 1] - 2 * dy * C1y[ix]) / (
                                         dy ** 2) - tau * (-2 * sol[ix][0][j - 1] + sol[ix][0][j - 2]) / dt) / (
                                            1 + tau / dt)
            for iy in range(1, ny - 1):
                sol[ix][iy][j] = (sol[ix][iy][j - 1]
                                  + dt * alpha * ((sol[ix + 1][iy][j - 1] - 2 * sol[ix][iy][j - 1] + sol[ix - 1][iy][
                            j - 1]) / (dx ** 2) + (sol[ix][iy + 1][j - 1] - 2 * sol[ix][iy][j - 1] + sol[ix][iy - 1][
                            j - 1]) / (dy ** 2)) - tau / dt * (-2 * sol[ix][iy][j - 1] + sol[ix][iy][j - 2])) / (
                                         1 + tau / dt)
            if neumann:
                sol[ix][ny - 1][j] = (sol[ix][ny - 1][j - 1] + dt * alpha * (
                            2 * sol[ix][ny - 2][j - 1] - 2 * sol[ix][ny - 1][j - 1] + 2 * dy * C2y[ix]) / (
                                              dy ** 2) - tau * (
                                                  -2 * sol[ix][ny - 1][j - 1] + sol[ix][ny - 1][j - 2]) / dt) / (
                                                 1 + tau / dt)
        if neumann:
            for iy in range(ny):
                sol[nx - 1][iy][j] = (sol[nx - 1][iy][j - 1] + dt * alpha * (
                            2 * sol[nx - 2][iy][j - 1] - 2 * sol[nx - 1][iy][j - 1] + 2 * dx * C2x[iy]) / (
                                              dx ** 2) - tau * (
                                                  -2 * sol[nx - 1][iy][j - 1] + sol[nx - 1][iy][j - 2]) / dt) / (
                                                 1 + tau / dt)

    return sol


def analytic_sol(x, t):
    u = math.sin(math.pi * x) * math.exp(-math.pi ** 2 * t)
    return u


if __name__ == '__main__':
    Lx = 1
    Ly = 1
    tmax = 2
    dx = 0.01
    dy = 0.01
    dt = 0.0001
    tau = 1
    alpha = 1
    neumann = True
    C = 0

    tau_str = str(tau).replace('.', '_')
    sol = numerical_sol(Lx, Ly, tmax, dx, dy, dt, tau, alpha, neumann, C)
    np.save(f'2DCattaneo_Neumann{tau_str}sol_sin', sol)
