import numpy as np
from scipy import integrate

def gen_tau(alpha, tau_0):
    return np.sqrt(alpha**2 / tau_0**2 + tau_0**2)

def gen_a_gauss(alpha, tau_0):
    return alpha / (alpha**2 + tau_0**4)

def f_gauss(t, A, tau_0, omega, alpha):
    return (A / np.sqrt(2 * np.pi * tau_0 * gen_tau(alpha, tau_0)
                      ) * np.exp(- t**2 / (2 * gen_tau(alpha, tau_0)**2)))**2

def phi_gauss(t, tau_0, alpha):
    return gen_a_gauss(alpha, tau_0) * t**2 / 2

def E_gauss(t, A, tau_0, omega, alpha):
     return f_gauss(t, A, tau_0, omega, alpha
                   )**(1 / 2) * np.exp(-1j * omega * t - 1j * phi_gauss(t, tau_0, alpha))

def delta_phi_gauss(t, tau, tau_0, alpha):
    return phi_gauss(t + tau, tau_0, alpha) - phi_gauss(t, tau_0, alpha)

def delta_phi_secant(t, tau, tau_0, alpha):
    return phi_secant(t + tau, tau_0, alpha) - phi_secant(t, tau_0, alpha)

def f_secant(t, A, tau_0, omega, alpha):
    return A * (2 / (np.exp(t / tau_0) + np.exp(- t / tau_0)))**2

def phi_secant(t, tau_0, alpha):
    return alpha * (t / tau_0)**2

def E_secant(t, A, tau_0, omega, alpha):
     return f_secant(t, A, tau_0, omega, alpha
                   )**(1 / 2) * np.exp(-1j * omega * t - 1j * phi_secant(t, tau_0, alpha))

# calculate A so that \int f(t) dt = 1

def normalize_f(f, tau_0, omega):
    A = 1 / integrate.quad(f, - np.inf, np.inf,
                                  args=(1, tau_0, omega, 0))[0]
    return A
