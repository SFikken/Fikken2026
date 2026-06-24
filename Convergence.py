import numpy as np
from matplotlib import pyplot as plt
from Fourier1D import fourier_cn, fourier_fw, fourier_rk
from Fourier1D import analytic_sol as fourier_analytic
from Fourier1D import semi_discrete_exact
from Cattaneo1D import cattaneo_cn, cattaneo_rk, cattaneo_fw
from Cattaneo1D import analytic_sol as cattaneo_analytic
from matplot2tikz import save



def convergence_analytic(schemes, dx_dt_lists, labels, orders, markers, fourier=True, discrete=False):
    for scheme, dx_dt_list, label, order, marker in zip(schemes, dx_dt_lists, labels, orders, markers):
        print(scheme)
        errors = []
        for dx, dt in dx_dt_list:
            if fourier:
                sol = scheme(L, tmax, dx, dt, ic, bcl, bcr, heat_loss=heat_loss)#, alpha, heat_loss, c)
            else:
                sol = scheme(L, tmax, dx, dt, ic1, ic2, heat_loss=heat_loss)#, alpha, tau, heat_loss, c)
            gridt = np.arange(0, tmax + dt, dt)
            gridx = np.arange(0, L + dx, dx)
            if fourier:
                if discrete:
                    an_sol = semi_discrete_exact(gridx, gridt[-1], dx, alpha)
                else:
                    an_sol = fourier_analytic(gridx, gridt[-1])
            else:
                an_sol = cattaneo_analytic(gridx, gridt[-1], tau)
            # print(an_sol)
            error = np.max(np.abs(sol[1:-1, -1] - an_sol[1:-1]))
            # an_sol = np.zeros_like(gridt)
            # for i in range(len(gridt)):
            #     t = gridt[i]
            #     an_sol[i] = fourier_analytic(gridx[int((len(gridx) - 1) / 2)], t)#, tau)
            # error = np.max(np.abs(sol[int((len(gridx) - 1) / 2), :] - an_sol))
            print(error)
            errors.append(error)
        dt_values = np.array([pair[1] for pair in dx_dt_list])
        plt.loglog(dt_values, errors, marker=marker, label=label)
        print('Plotted')
    # plt.loglog(dt_values, errors[0] * (dt_values / dt_values[0]) ** order, linestyle=(0, (5, 5)), linewidth=2, label=f'Line order {order}')
    # if discrete:
    #     plt.title('Max error with discrete analytic solution at time t = 0.2 for various dt')
    # else:
    #     plt.title('Max error with analytic solution at time t = 0.2 for various dt')
    plt.xlabel('dt')
    plt.ylabel('Max error')
    # plt.ylim(errors[-1], errors[0])
    plt.legend()
    plt.xticks(dt_values)
    plt.gca().invert_xaxis()
    plt.grid()
    # save('Tikz Plots/Fourier_Convergence_dt.tex')
    plt.show()


def p_value(scheme1, dx_dt_list, fourier=True):
    print(scheme1)
    solutions = []
    for dx, dt in dx_dt_list:
        gridx = np.arange(0, L + dx, dx)
        if fourier:
            sol = scheme1(L, tmax, dx, dt, ic, bcl, bcr, heat_loss=heat_loss)#, alpha, heat_loss, c)
        else:
            sol = scheme1(L, tmax, dx, dt, ic1, ic2, heat_loss=heat_loss)#, alpha, tau, heat_loss, c)
        solutions.append(sol[:, -1])
        # solutions.append(sol[int((len(gridx) - 1) / 2), :])

    p_list = []
    for k in range(len(solutions) - 2):
        u_dt = solutions[k]
        u_dt2 = solutions[k + 1]
        u_dt4 = solutions[k + 2]
        diff1 = np.max(np.abs(u_dt - u_dt2))
        diff2 = np.max(np.abs(u_dt2 - u_dt4))
        p = round(float(np.log2(diff1 / diff2)), 8)
        print(diff1/diff2)
        p_list.append(p)
    return p_list, sum(p_list)/len(p_list)


def half_time_step(schemes, dx_dt_lists, labels, orders, markers, fourier=True):
    for scheme, dx_dt_list, label, order, marker in zip(schemes, dx_dt_lists, labels, orders, markers):
        solutions = []
        dt_values = np.array([pair[1] for pair in dx_dt_list])
        for dx, dt in dx_dt_list:
            if fourier:
                sol = scheme(L, tmax, dx, dt, ic, bcl, bcr, heat_loss=heat_loss)#, alpha, heat_loss, c)
            else:
                sol = scheme(L, tmax, dx, dt, ic1, ic2, heat_loss=heat_loss)#, alpha, tau, heat_loss, c)
            solutions.append(sol[:, -1])
            gridx = np.arange(0, L + dx, dx)
            # solutions.append(sol[int((len(gridx) - 1) / 2), :])

        errors = []
        for k in range(len(solutions) - 1):
            diff = np.max(np.abs(solutions[k] - solutions[k + 1]))
            errors.append(diff)
        plt.loglog(dt_values[:-1], errors, marker=marker, label=label)
        plt.loglog(dt_values[:-1], errors[0] * (dt_values[:-1] / dt_values[0]) ** order, '--',
                   label=f'Line order {order}')

    # plt.title('Max difference at t = 0.2 between time steps')
    plt.ylabel('Max error')
    plt.xlabel('dt')
    plt.legend()
    plt.xticks(dt_values[:-1])
    plt.gca().invert_xaxis()
    plt.grid()
    save('Tikz Plots/Fourier_Half_Convergence_dt.tex')
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
    tmax = 0.2
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
    orders = [1, 2, 4]
    # orders = [2, 2, 2]

    convergence_analytic(schemes, dx_dt_lists, labels, orders, markers, fourier, False)

    # for i in range(3):
    #     print(labels[i], p_value(schemes[i], dx_dt_lists[i], fourier))

    # half_time_step(schemes, dx_dt_lists, labels, orders, markers, fourier)
