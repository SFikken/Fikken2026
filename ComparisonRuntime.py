import numpy as np
from matplotlib import pyplot as plt
from Fourier1D import fourier_cn, fourier_fw, fourier_rk
from Cattaneo1D import cattaneo_cn, cattaneo_rk, cattaneo_fw
from matplot2tikz import save
import time


def plot_runtimes(schemes, dx_dt_lists, labels, markers, fourier):
    for scheme, dx_dt_list, label, marker in zip(schemes , dx_dt_lists, labels, markers):
        times = []
        dt_values = np.array([pair[1] for pair in dx_dt_list])
        for dx, dt in dx_dt_list:
            print(scheme, dx, dt)
            start = time.time()
            if fourier:
                sol = scheme(L, tmax, dx, dt, ic, bcl, bcr, heat_loss=heat_loss)  # , alpha, heat_loss, c)
            else:
                sol = scheme(L, tmax, dx, dt, ic1, ic2, heat_loss=heat_loss)  # , alpha, tau, heat_loss, c)
            end = time.time()
            times.append(end - start)
        plt.loglog(dt_values, times, label=label, marker=marker)
    plt.loglog(dt_values, times[0] * (dt_values / dt_values[0]) ** -1, '--',label=f'Line order {1}')
    plt.xlabel('dt')
    plt.ylabel('Runtime [s]')
    plt.xticks(dt_values)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.grid()
    save('Tikz Plots/Fourier_Runtime_dt.tex')
    plt.show()



if __name__ == '__main__':
    def initial_condition(x):
        return np.sin(np.pi * x)


    ic = initial_condition


    def boundary_condition_left(t):
        return np.zeros_like(t)


    def boundary_condition_right(t):
        return np.zeros_like(t)


    bcl = boundary_condition_left
    bcr = boundary_condition_right


    def first_initial_condition(x):
        return np.sin(np.pi * x)


    def second_initial_condition(x):
        return np.zeros_like(x)


    ic1 = first_initial_condition
    ic2 = second_initial_condition

    L = 1
    tmax = 2
    alpha = 1
    tau = 0.01
    heat_loss = False
    c = 0

    fourier = True
    if fourier:
        schemes = [fourier_fw, fourier_cn, fourier_rk]
    else:
        schemes = [cattaneo_fw, cattaneo_cn, cattaneo_rk]

    # dx_dt_lists = [
    #     [(0.01, 5e-5), (0.01, 2.5e-5), (0.01, 1.25e-5), (0.01, 6.25e-6), (0.01, 3.125e-6)],
    #     [(0.01, 5e-5), (0.01, 2.5e-5), (0.01, 1.25e-5), (0.01, 6.25e-6), (0.01, 3.125e-6)],
    #     [(0.01, 5e-5), (0.01, 2.5e-5), (0.01, 1.25e-5), (0.01, 6.25e-6), (0.01, 3.125e-6)]
    # ]
    dx_dt_lists = [
        [(0.05, 1e-3), (0.05, 5e-4), (0.05, 2.5e-4), (0.05, 1.25e-4), (0.05, 6.25e-5)],
        [(0.05, 1e-3), (0.05, 5e-4), (0.05, 2.5e-4), (0.05, 1.25e-4), (0.05, 6.25e-5)],
        [(0.05, 1e-3), (0.05, 5e-4), (0.05, 2.5e-4), (0.05, 1.25e-4), (0.05, 6.25e-5)],
    ]
    # dx_dt_lists = [
    #     ([(0.1, 1e-5), (0.05, 1e-5), (0.025, 1e-5), (0.0125, 1e-5), (0.00625, 1e-5)]),
    #     ([(0.1, 1e-5), (0.05, 1e-5), (0.025, 1e-5), (0.0125, 1e-5), (0.00625, 1e-5)]),
    #     ([(0.1, 1e-5), (0.05, 1e-5), (0.025, 1e-5), (0.0125, 1e-5), (0.00625, 1e-5)])
    # ]
    labels = ['Forward', 'Crank-Nicolson', 'Runge-Kutta']
    markers = ['s', 'o', 'v']

    plot_runtimes(schemes, dx_dt_lists, labels, markers, fourier)