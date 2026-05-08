import math
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_float_scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().reshape(-1)[0].item())
    return float(x)


def _to_1d_strike_tensor(K_strikes, device):
    if torch.is_tensor(K_strikes):
        return K_strikes.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.as_tensor(K_strikes, dtype=torch.float32, device=device).reshape(-1)


def Heston_MC_Batched(kappa, theta, sigma, v0, rho, T, K_strikes, S, r, q, N_paths=2000, N_steps=100):
    dt = float(T) / N_steps
    sqrt_dt = math.sqrt(dt)
    num_samples = kappa.shape[0]

    kappa = kappa.view(-1, 1).float()
    theta = theta.view(-1, 1).float()
    sigma = sigma.view(-1, 1).float()
    v0 = v0.view(-1, 1).float()
    rho = rho.view(-1, 1).float()

    S_val = _to_float_scalar(S)
    r_val = _to_float_scalar(r)
    q_val = _to_float_scalar(q)
    T_val = float(T)

    Z1 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)
    Z2 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)

    log_S = torch.full((num_samples, N_paths), math.log(S_val), dtype=torch.float32, device=device)
    V_t = v0.expand(num_samples, N_paths).clone()

    rho_term = torch.sqrt(torch.clamp(1.0 - rho**2, min=1e-12))

    for t in range(N_steps):
        ZS = Z1[:, :, t]
        ZV = rho * ZS + rho_term * Z2[:, :, t]

        V_next = (
            V_t
            + kappa * (theta - V_t) * dt
            + sigma * torch.sqrt(torch.clamp(V_t, min=0.0)) * sqrt_dt * ZV
        )
        V_t = torch.clamp(V_next, min=1e-8)

        log_S = (
            log_S
            + (r_val - q_val - 0.5 * V_t) * dt
            + torch.sqrt(V_t) * sqrt_dt * ZS
        )

    S_T = torch.exp(log_S)
    K_tensor = _to_1d_strike_tensor(K_strikes, device).view(1, 1, -1)
    payoffs = torch.relu(S_T.unsqueeze(-1) - K_tensor)
    prices = torch.mean(payoffs, dim=1) * math.exp(-r_val * T_val)

    return torch.clamp(prices, min=1e-7)


def Bates_MC_Batched(kappa, theta, sigma, v0, rho, lambdJ, muJ, sigmaJ,
                     T, K_strikes, S, r, q, N_paths=4000, N_steps=100):
    dt = float(T) / N_steps
    sqrt_dt = math.sqrt(dt)
    num_samples = kappa.shape[0]

    kappa = kappa.view(-1, 1).float()
    theta = theta.view(-1, 1).float()
    sigma = sigma.view(-1, 1).float()
    v0 = v0.view(-1, 1).float()
    rho = rho.view(-1, 1).float()
    lambdJ = lambdJ.view(-1, 1).float()
    muJ = muJ.view(-1, 1).float()
    sigmaJ = sigmaJ.view(-1, 1).float()

    S_val = _to_float_scalar(S)
    r_val = _to_float_scalar(r)
    q_val = _to_float_scalar(q)
    T_val = float(T)

    Z1 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)
    Z2 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)

    log_S = torch.full((num_samples, N_paths), math.log(S_val), dtype=torch.float32, device=device)
    V_t = v0.expand(num_samples, N_paths).clone()

    rho_term = torch.sqrt(torch.clamp(1.0 - rho**2, min=1e-12))

    for t in range(N_steps):
        ZS = Z1[:, :, t]
        ZV = rho * ZS + rho_term * Z2[:, :, t]

        jump_prob = torch.clamp(lambdJ * dt, min=0.0, max=1.0)
        jump_mask = (
            torch.rand(num_samples, N_paths, dtype=torch.float32, device=device) < jump_prob
        ).float()
        jump_sizes = (
            torch.randn(num_samples, N_paths, dtype=torch.float32, device=device) * sigmaJ + muJ
        ) * jump_mask

        V_next = (
            V_t
            + kappa * (theta - V_t) * dt
            + sigma * torch.sqrt(torch.clamp(V_t, min=0.0)) * sqrt_dt * ZV
        )
        V_t = torch.clamp(V_next, min=1e-8)

        log_S = (
            log_S
            + (r_val - q_val - 0.5 * V_t) * dt
            + torch.sqrt(V_t) * sqrt_dt * ZS
            + jump_sizes
        )

    S_T = torch.exp(log_S)
    K_tensor = _to_1d_strike_tensor(K_strikes, device).view(1, 1, -1)
    payoffs = torch.relu(S_T.unsqueeze(-1) - K_tensor)
    prices = torch.mean(payoffs, dim=1) * math.exp(-r_val * T_val)

    return torch.clamp(prices, min=1e-7)


def Batesprice_Batched(kappa, theta, sigma, v0, rho, lambdJ, muJ, sigmaJ,
                       T, K, S, r, q, x_nodes=None, w_nodes=None, N_paths=4000, N_steps=100):
    prices = Bates_MC_Batched(
        kappa=kappa, theta=theta, sigma=sigma, v0=v0, rho=rho,
        lambdJ=lambdJ, muJ=muJ, sigmaJ=sigmaJ,
        T=float(T), K_strikes=K, S=S, r=r, q=q,
        N_paths=N_paths, N_steps=N_steps
    )
    return prices.to(torch.float32)


def Hestonprice_Batched(kappa, theta, sigma, v0, rho,
                        T, K, S, r, q, x_nodes=None, w_nodes=None, N_paths=2000, N_steps=100):
    prices = Heston_MC_Batched(
        kappa=kappa, theta=theta, sigma=sigma, v0=v0, rho=rho,
        T=float(T), K_strikes=K, S=S, r=r, q=q,
        N_paths=N_paths, N_steps=N_steps
    )
    return prices.to(torch.float32)


def Bergomi_1F_MC_Batched(xi, nu, rho, beta, T, K_strikes, S, r, q, N_paths=3000, N_steps=100):
    dt = float(T) / N_steps
    num_samples = xi.shape[0]

    xi = xi.view(-1, 1).float()
    nu = nu.view(-1, 1).float()
    beta = beta.view(-1, 1).float()
    rho = rho.view(-1, 1, 1).float()

    Z1 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)
    Z2 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)
    dW_S = Z1 * math.sqrt(dt)
    dW_V = (rho * Z1 + torch.sqrt(torch.clamp(1 - rho**2, min=1e-12)) * Z2) * math.sqrt(dt)

    X = torch.zeros(num_samples, N_paths, dtype=torch.float32, device=device)
    log_S = torch.full(
        (num_samples, N_paths),
        math.log(_to_float_scalar(S)),
        dtype=torch.float32,
        device=device
    )

    r_val = _to_float_scalar(r)
    q_val = _to_float_scalar(q)

    for t in range(N_steps):
        t_val = (t + 1) * dt
        E_X2 = (1 - torch.exp(-2 * beta * t_val)) / (2 * beta)
        V_t = xi * torch.exp(nu * X - 0.5 * nu**2 * E_X2)
        log_S = (
            log_S
            + (r_val - q_val - 0.5 * V_t) * dt
            + torch.sqrt(torch.clamp(V_t, min=1e-12)) * dW_S[:, :, t]
        )
        X = X - beta * X * dt + dW_V[:, :, t]

    S_T = torch.exp(log_S)
    K_tensor = _to_1d_strike_tensor(K_strikes, device).view(1, 1, -1)
    payoffs = torch.relu(S_T.unsqueeze(-1) - K_tensor)
    call_prices = torch.mean(payoffs, dim=1) * math.exp(-r_val * float(T))

    return torch.clamp(call_prices, min=1e-7)


def rBergomi_MC_Batched(xi, nu, rho, H, T, K_strikes, S, r, q, N_paths=4000, N_steps=150):
    dt = float(T) / N_steps
    num_samples = xi.shape[0]

    xi = xi.view(-1, 1).float()
    nu = nu.view(-1, 1).float()
    rho = rho.view(-1, 1, 1).float()
    H = H.view(-1, 1, 1).float()

    Z1 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)
    Z2 = torch.randn(num_samples, N_paths, N_steps, dtype=torch.float32, device=device)
    dW_S = Z1 * math.sqrt(dt)
    dW_V = (rho * Z1 + torch.sqrt(torch.clamp(1 - rho**2, min=1e-12)) * Z2) * math.sqrt(dt)

    log_S = torch.full(
        (num_samples, N_paths),
        math.log(_to_float_scalar(S)),
        dtype=torch.float32,
        device=device
    )

    r_val = _to_float_scalar(r)
    q_val = _to_float_scalar(q)

    for t in range(N_steps):
        t_val = (t + 1) * dt

        if t == 0:
            Z_t = torch.zeros(num_samples, N_paths, dtype=torch.float32, device=device)
        else:
            s_vals = torch.arange(0, t, device=device, dtype=torch.float32) * dt
            kernel = (t_val - s_vals).view(1, 1, -1) ** (H - 0.5)
            Z_t = torch.sum(kernel * dW_V[:, :, :t], dim=2)

        H_scalar = H.squeeze(-1)
        V_t = xi * torch.exp(
            nu * math.sqrt(2.0) * torch.sqrt(torch.clamp(H_scalar, min=1e-8)) * Z_t
            - 0.5 * nu**2 * (t_val ** (2 * H_scalar))
        )

        log_S = (
            log_S
            + (r_val - q_val - 0.5 * V_t) * dt
            + torch.sqrt(torch.clamp(V_t, min=1e-12)) * dW_S[:, :, t]
        )

    S_T = torch.exp(log_S)
    K_tensor = _to_1d_strike_tensor(K_strikes, device).view(1, 1, -1)
    payoffs = torch.relu(S_T.unsqueeze(-1) - K_tensor)
    call_prices = torch.mean(payoffs, dim=1) * math.exp(-r_val * float(T))

    return torch.clamp(call_prices, min=1e-7)


def Bergomiprice_Batched(xi, nu, rho_SV, beta, T, K_strikes, S, r, q, N_paths=3000, N_steps=100):
    prices = Bergomi_1F_MC_Batched(
        xi=xi.float(), nu=nu.float(), rho=rho_SV.float(), beta=beta.float(),
        T=float(T), K_strikes=K_strikes, S=S, r=r, q=q,
        N_paths=N_paths, N_steps=N_steps
    )
    return prices.to(torch.float32)


def rBergomiprice_Batched(xi, nu, rho_SV, H, T, K_strikes, S, r, q, N_paths=4000, N_steps=150):
    prices = rBergomi_MC_Batched(
        xi=xi.float(), nu=nu.float(), rho=rho_SV.float(), H=H.float(),
        T=float(T), K_strikes=K_strikes, S=S, r=r, q=q,
        N_paths=N_paths, N_steps=N_steps
    )
    return prices.to(torch.float32)


SVprice_Batched = Batesprice_Batched
