import math
import numpy as np
import torch
from scipy.interpolate import griddata


def interpolate_surface_linear_nearest(day_data, value_col, LM_mesh, T_mesh):
    points = np.column_stack([
        day_data['LogMoneyness'].values,
        day_data['Tau'].values
    ])
    values = day_data[value_col].values
    xi = np.column_stack([LM_mesh.ravel(), T_mesh.ravel()])

    interp_linear = griddata(points, values, xi, method='linear')
    interp_nearest = griddata(points, values, xi, method='nearest')
    interp = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)
    return interp.reshape(LM_mesh.shape)


def GenerateGaussLaguerre(n, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from math import factorial

    def nchoosek(n_, r_):
        return factorial(n_) / (factorial(r_) * factorial(n_ - r_))

    x, _ = np.polynomial.laguerre.laggauss(n)
    w = np.zeros(n)
    dL = np.zeros((n, len(x)))

    for j in range(len(x)):
        for k in range(n):
            dL[k, j] = (-1) ** (k + 1) / factorial(k) * nchoosek(n, k + 1) * x[j] ** k
        w[j] = 1 / x[j] / sum(dL[:, j]) ** 2
        w[j] = w[j] * np.exp(x[j])

    return (
        torch.tensor(x, dtype=torch.complex128, device=device).unsqueeze(0),
        torch.tensor(w, dtype=torch.complex128, device=device).unsqueeze(0)
    )


def JumpCF_PT(phi, lambdJ, muJ, sigmaJ, T):
    i = 1j
    term1 = -lambdJ * muJ * i * phi * T
    term2 = lambdJ * T * (
        (1 + muJ) ** (i * phi) * torch.exp(0.5 * sigmaJ**2 * i * phi * (i * phi - 1)) - 1
    )
    return torch.exp(term1 + term2)


def HestonCF_PT(phi, kappa, theta, sigma, v0, rho, T, S, r, q):
    i = 1j
    x = torch.log(S)
    a = kappa * theta
    u = -0.5
    b = kappa

    d = torch.sqrt((rho * sigma * i * phi - b) ** 2 - sigma**2 * (2 * u * i * phi - phi**2))
    c = (b - rho * sigma * i * phi - d) / (b - rho * sigma * i * phi + d)
    exp_dT = torch.exp(-d * T)

    D = ((b - rho * sigma * i * phi - d) / sigma**2) * ((1 - exp_dT) / (1 - c * exp_dT))
    G = (1 - c * exp_dT) / (1 - c)
    C = (r - q) * i * phi * T + a / sigma**2 * ((b - rho * sigma * i * phi - d) * T - 2 * torch.log(G))

    return torch.exp(C + D * v0 + i * phi * x)


def BatesCF_PT(phi, kappa, theta, sigma, v0, rho, lambdJ, muJ, sigmaJ, T, S, r, q):
    return HestonCF_PT(phi, kappa, theta, sigma, v0, rho, T, S, r, q) * JumpCF_PT(phi, lambdJ, muJ, sigmaJ, T)
