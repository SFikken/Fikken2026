import numpy as np
from matplotlib import pyplot as plt
from Fourier1D import fourier_fw
from Cattaneo1D import cattaneo_fw
from matplot2tikz import save


times, volts = np.loadtxt(
    # "Sem - Copy/100/I-G_1__100 hydrogel_1_760.LF",
    "Sem - Copy/50/I-G_1__50 hydrogel_1_graphite_774.LF",
    usecols=(0, 1),
    unpack=True,
    skiprows=1  # skip header
)
cutoff = np.argmin(abs(times - 10000))
volts = volts[:cutoff]
times = times[:cutoff]
index = 0
for t in times:
    if t < 0:
        index += 1
    else:
        break

t0 = sum(volts[:index]/index)
tinf = sum(volts[-100:]/100)
print(t0)
window = 50
pad = window // 2
volts_padded = np.pad(
    volts,
    pad_width=(window//2, window//2),
    mode='constant',
    constant_values=(t0, tinf)
)
avg_volts = (np.convolve(volts_padded, np.ones(window)/window, mode='same')[window//2:-window//2]-t0)/(tinf - t0)
t = np.abs(avg_volts - 0.5).argmin()
print(times[t], avg_volts[t])


def initial_condition(x):
    n = len(x)
    k = int(n / 100)
    return np.concatenate([n/(k + 1) * np.ones(k + 1), np.zeros(n - k - 1)])


def boundary_condition_left(t):
    return np.zeros_like(t)


def boundary_condition_right(t):
    return np.zeros_like(t)


def ic2(x):
    return np.zeros_like(x)


ic, bcl, bcr = initial_condition, boundary_condition_left, boundary_condition_right
L = 0.005
#1.65e-6
alpha = 1.38 * L**2 / (np.pi**2 * times[t] * 1e-3)
composition = [(alpha, 1, 0)]
dt = (times[index+1] - times[index]) * 1e-6
dx = 0.00001
tau = 0.1
sol = fourier_fw(L, times[-1]*1e-3, dx, dt, ic, bcl, bcr, composition, True)
sol2 = cattaneo_fw(L, times[-1]*1e-3, dx, dt, ic, ic2, composition, tau, True)
print(dt, alpha, composition)
skip = 1000
plt.plot(times, (volts - t0)/(tinf - t0), label='Scaled Measurement Data')
plt.plot(times, avg_volts, label=f'Moving Average, window = {window}')
# plt.scatter(times[t], avg_volts[t], color='red', label='Grid point closest to Vtilde = 0.5', s=200)
# plt.plot(times, 0.5 * np.ones_like(times), linestyle=(0, (5, 5)), label='Vtilde = 0.5')
plt.plot(times[index:], sol[-1, ::skip], label='Fourier Model')
plt.plot(times[index:], sol2[-1, ::skip], label=f'Cattaneo Model, tau = {tau}')
# plt.plot(times[index:], (sol[-1, ::skip] + 0.07 * sol[-1, -1])/(1.07 * sol[-1, -1]), label='Fourier Model, shifted up')
# plt.xlim(0, 10000)
plt.ylabel('Vtilde = (V - V0)/(Vinf - V0)')
plt.xlabel('Time [ms]')
plt.legend()
save('Measurements.tex')
plt.show()
