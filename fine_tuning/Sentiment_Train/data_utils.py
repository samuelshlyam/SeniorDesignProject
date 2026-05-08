import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from pathlib import Path
from collections import Counter

from model import VolatilitySurfaceCNN
from utils import interpolate_surface_linear_nearest

# Project root is the parent of the Non_Sentiment_Train folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Log-moneyness convention detection
# ---------------------------------------------------------------------------

def infer_logmoneyness_convention(df):
    if not {'Forward', 'Strike', 'LogMoneyness'}.issubset(df.columns):
        return 'unknown'

    tmp = df[['Forward', 'Strike', 'LogMoneyness']].dropna().copy()
    if len(tmp) == 0:
        return 'unknown'

    tmp = tmp.sample(min(len(tmp), 5000), random_state=42)

    err_k_over_f = np.median(np.abs(tmp['LogMoneyness'] - np.log(tmp['Strike'] / tmp['Forward'])))
    err_f_over_k = np.median(np.abs(tmp['LogMoneyness'] - np.log(tmp['Forward'] / tmp['Strike'])))

    return 'k_over_f' if err_k_over_f < err_f_over_k else 'f_over_k'


# ---------------------------------------------------------------------------
# Price filtering
# ---------------------------------------------------------------------------

def filter_extreme_prices(iv_tensor, price_tensor, context_tensor, dates,
                          global_price_cap_percentile=97,
                          per_point_low_pct=2, per_point_high_pct=98,
                          day_outlier_iqr_factor=2.0):
    n_days = price_tensor.shape[0]
    print(f"\n--- Filtering extreme prices ({n_days} days in) ---")

    prices_np = price_tensor.squeeze(1).numpy().copy()  # (N, 8, 11)
    ivs_np = iv_tensor.squeeze(1).numpy().copy()
    context_np = context_tensor.numpy().copy() if context_tensor is not None else None

    lo = np.percentile(prices_np, per_point_low_pct, axis=0)
    hi = np.percentile(prices_np, per_point_high_pct, axis=0)
    prices_np = np.clip(prices_np, lo[None, :, :], hi[None, :, :])
    print(f"  Per-point winsorized to [{per_point_low_pct}, {per_point_high_pct}] percentile")

    global_cap = np.percentile(prices_np, global_price_cap_percentile)
    global_floor = np.percentile(prices_np, 100 - global_price_cap_percentile)
    prices_np = np.clip(prices_np, max(global_floor, 1e-7), global_cap)
    print(f"  Global price cap: {global_cap:.6f} (p{global_price_cap_percentile})")

    day_means = prices_np.mean(axis=(1, 2))
    q1, q3 = np.percentile(day_means, 25), np.percentile(day_means, 75)
    iqr = q3 - q1
    lower_bound = q1 - day_outlier_iqr_factor * iqr
    upper_bound = q3 + day_outlier_iqr_factor * iqr

    keep_mask = (day_means >= lower_bound) & (day_means <= upper_bound)
    n_removed = (~keep_mask).sum()

    prices_np = prices_np[keep_mask]
    ivs_np = ivs_np[keep_mask]
    if context_np is not None:
        context_np = context_np[keep_mask]
    dates = [d for d, m in zip(dates, keep_mask) if m]

    print(f"  Day-level IQR filter: removed {n_removed} outlier days "
          f"(bounds: [{lower_bound:.4f}, {upper_bound:.4f}])")

    iv_out = torch.tensor(ivs_np, dtype=torch.float32).unsqueeze(1)
    price_out = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(1)
    context_out = torch.tensor(context_np, dtype=torch.float32) if context_np is not None else None

    print(f"  After filtering: {price_out.shape[0]} days remain")
    print(f"  Price range: [{price_out.min().item():.6f}, {price_out.max().item():.6f}]")
    print(f"  Price mean/std: {price_out.mean().item():.6f} / {price_out.std().item():.6f}")
    print(f"--- Done ---\n")

    return iv_out, price_out, context_out, dates


# ---------------------------------------------------------------------------
# IV + price surface construction from raw CSVs
# ---------------------------------------------------------------------------

def create_iv_and_price_grids_from_raw(
    source_csvs,
    max_days=None,
    use_calls_only=False,
    min_points_per_day=6,
    min_tau_per_day=2,
    min_lm_per_day=3
):
    if isinstance(source_csvs, (str, Path)):
        source_csvs = [str(source_csvs)]
    source_csvs = [str(p) for p in source_csvs]

    tau_grid = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]
    lm_grid = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    LM_mesh, T_mesh = np.meshgrid(lm_grid, tau_grid)

    frames = []
    lm_conventions = []

    for source_csv in source_csvs:
        df = pd.read_csv(source_csv, low_memory=False)

        required = ['Trade Date', 'LogMoneyness', 'Tau', 'Implied Volatility', 'Strike']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in {source_csv}")

        local_price_col = 'Mid' if 'Mid' in df.columns else 'OptionPrice'
        if local_price_col not in df.columns:
            raise ValueError(f"Missing Mid/OptionPrice column in {source_csv}")

        keep_cols = required + [local_price_col]
        for extra in ['OptionType', 'IsCall', 'S', 'Forward', 'r', 'q',
                       'daily_sentiment', 'articles_per_day']:
            if extra in df.columns:
                keep_cols.append(extra)

        local_df = df[keep_cols].copy()
        local_df = local_df.rename(columns={local_price_col: 'RawPrice'})
        local_df['SourceFile'] = Path(source_csv).name

        lm_convention = infer_logmoneyness_convention(local_df)
        lm_conventions.append(lm_convention)
        frames.append(local_df)

    df = pd.concat(frames, ignore_index=True)
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Trade Date', 'LogMoneyness', 'Tau', 'Implied Volatility', 'RawPrice', 'Strike'])

    lm_convention = lm_conventions[0]
    print("Detected LogMoneyness convention:", lm_convention)

    # Build per-date sentiment context mapping before further filtering
    has_sentiment = ('daily_sentiment' in df.columns and 'articles_per_day' in df.columns)
    if has_sentiment:
        df['daily_sentiment'] = pd.to_numeric(df['daily_sentiment'], errors='coerce').fillna(0.0)
        df['articles_per_day'] = pd.to_numeric(df['articles_per_day'], errors='coerce').fillna(0.0)
        a_min = df['articles_per_day'].min()
        a_max = df['articles_per_day'].max()
        df['articles_scaled'] = (df['articles_per_day'] - a_min) / (a_max - a_min + 1e-6)
        date_ctx = df.groupby('Trade Date').agg(
            daily_sentiment=('daily_sentiment', 'first'),
            articles_scaled=('articles_scaled', 'first')
        )
        date_to_context = {
            date: [float(row['daily_sentiment']), float(row['articles_scaled'])]
            for date, row in date_ctx.iterrows()
        }
        print("Sentiment context available for", len(date_to_context), "dates.")
    else:
        date_to_context = {}
        print("No sentiment columns found. Context will default to zeros.")

    total_rows_before_type_filter = len(df)

    if use_calls_only:
        if 'OptionType' in df.columns:
            df = df[df['OptionType'].astype(str).str.upper() == 'C'].copy()
            print("Using calls only for real-data fine tuning.")
        elif 'IsCall' in df.columns:
            df = df[df['IsCall'] == 1].copy()
            print("Using calls only for real-data fine tuning.")
        else:
            print("No call flag found. Keeping all rows.")
    else:
        print("Using both calls and puts. Puts will be converted to call-equivalent prices.")

    print(f"Rows before option-type filter: {total_rows_before_type_filter}")
    print(f"Rows after  option-type filter: {len(df)}")

    df = df[(df['Tau'] >= min(tau_grid)) & (df['Tau'] <= max(tau_grid))]
    df = df[(df['LogMoneyness'] >= min(lm_grid) - 0.05) & (df['LogMoneyness'] <= max(lm_grid) + 0.05)]

    iv_median = pd.to_numeric(df['Implied Volatility'], errors='coerce').median()
    if pd.notna(iv_median) and iv_median > 1.5:
        df['ImpliedVolUsed'] = pd.to_numeric(df['Implied Volatility'], errors='coerce') / 100.0
        print("Detected percentage IVs. Converting IVs from percent to decimal.")
    else:
        df['ImpliedVolUsed'] = pd.to_numeric(df['Implied Volatility'], errors='coerce')

    strike_numeric = pd.to_numeric(df['Strike'], errors='coerce')
    lm_numeric = pd.to_numeric(df['LogMoneyness'], errors='coerce')
    tau_numeric = pd.to_numeric(df['Tau'], errors='coerce')

    if 'Forward' in df.columns:
        forward_numeric = pd.to_numeric(df['Forward'], errors='coerce')
    else:
        forward_numeric = pd.Series(np.nan, index=df.index)

    if lm_convention == 'k_over_f':
        reconstructed_forward = strike_numeric * np.exp(-lm_numeric)
    else:
        reconstructed_forward = strike_numeric * np.exp(lm_numeric)

    df['ForwardProxy'] = forward_numeric.where(forward_numeric > 0, reconstructed_forward)

    if 'r' in df.columns:
        r_numeric = pd.to_numeric(df['r'], errors='coerce')
        df['r_used'] = np.where(r_numeric > 1.0, r_numeric / 100.0, r_numeric)
    else:
        df['r_used'] = 0.045

    if 'q' in df.columns:
        q_numeric = pd.to_numeric(df['q'], errors='coerce')
        df['q_used'] = np.where(q_numeric > 1.0, q_numeric / 100.0, q_numeric)
    else:
        df['q_used'] = 0.011

    df['r_used'] = pd.to_numeric(df['r_used'], errors='coerce').fillna(0.045)
    df['q_used'] = pd.to_numeric(df['q_used'], errors='coerce').fillna(0.011)

    df['SpotProxy'] = df['ForwardProxy'] * np.exp(-(df['r_used'] - df['q_used']) * tau_numeric)
    df = df.dropna(subset=['ImpliedVolUsed', 'RawPrice', 'ForwardProxy', 'SpotProxy'])
    df = df[(df['ForwardProxy'] > 1e-8) & (df['SpotProxy'] > 1e-8)].copy()

    raw_price_numeric = pd.to_numeric(df['RawPrice'], errors='coerce')
    df['NormalizedPrice'] = raw_price_numeric / df['SpotProxy']

    if 'S' in df.columns:
        s_numeric = pd.to_numeric(df['S'], errors='coerce').replace(0, np.nan)
        df['BadNormalization_Using_S'] = raw_price_numeric / s_numeric
        print("Raw S normalization q99.9:", df['BadNormalization_Using_S'].quantile(0.999))
        print("Raw S normalization max:", df['BadNormalization_Using_S'].max())

    print("SpotProxy normalization q99.9:", df['NormalizedPrice'].quantile(0.999))
    print("SpotProxy normalization max:", df['NormalizedPrice'].max())

    if not use_calls_only:
        if 'OptionType' in df.columns:
            option_flag = df['OptionType'].astype(str).str.upper()
        elif 'IsCall' in df.columns:
            option_flag = np.where(df['IsCall'] == 1, 'C', 'P')
            option_flag = pd.Series(option_flag, index=df.index)
        else:
            option_flag = pd.Series(['C'] * len(df), index=df.index)

        strike_over_spot = strike_numeric.loc[df.index] / df['SpotProxy']
        put_mask = option_flag == 'P'

        df.loc[put_mask, 'NormalizedPrice'] = (
            df.loc[put_mask, 'NormalizedPrice']
            + np.exp(-df.loc[put_mask, 'q_used'] * df.loc[put_mask, 'Tau'])
            - np.exp(-df.loc[put_mask, 'r_used'] * df.loc[put_mask, 'Tau']) * strike_over_spot.loc[put_mask]
        )
        print(f"Converted {put_mask.sum()} put rows to call-equivalent prices.")

    rows_before_clean_filter = len(df)
    df = df[
        df['NormalizedPrice'].notna() &
        df['ImpliedVolUsed'].notna() &
        (df['ImpliedVolUsed'] > 0.0) &
        (df['ImpliedVolUsed'] < 3.0) &
        (df['NormalizedPrice'] > -0.25) &
        (df['NormalizedPrice'] < 3.0)
    ].copy()
    print(f"Rows before broken-row filter: {rows_before_clean_filter}")
    print(f"Rows after  broken-row filter: {len(df)}")

    grouped = (
        df.groupby(['Trade Date', 'Tau', 'LogMoneyness'], as_index=False)
          .agg({'ImpliedVolUsed': 'mean', 'NormalizedPrice': 'mean'})
    )

    iv_surfaces = []
    price_surfaces = []
    context_surfaces = []
    used_dates = []

    unique_dates = sorted(grouped['Trade Date'].unique())
    print(f"Found {len(unique_dates)} days of raw market data across {len(source_csvs)} file(s)...")

    skip_reasons = Counter()
    used_days = 0
    skipped_days = 0

    for date in unique_dates:
        if max_days is not None and used_days >= max_days:
            break

        day_data = grouped[grouped['Trade Date'] == date].copy()

        if len(day_data) < min_points_per_day:
            skipped_days += 1
            skip_reasons['too_few_points'] += 1
            continue

        if day_data['Tau'].nunique() < min_tau_per_day:
            skipped_days += 1
            skip_reasons['too_few_maturities'] += 1
            continue

        if day_data['LogMoneyness'].nunique() < min_lm_per_day:
            skipped_days += 1
            skip_reasons['too_few_moneyness_points'] += 1
            continue

        iv_low, iv_high = day_data['ImpliedVolUsed'].quantile([0.01, 0.995])
        px_low, px_high = day_data['NormalizedPrice'].quantile([0.01, 0.995])

        day_data['ImpliedVolUsed'] = day_data['ImpliedVolUsed'].clip(
            lower=max(0.01, iv_low), upper=min(2.5, iv_high)
        )
        day_data['NormalizedPrice'] = day_data['NormalizedPrice'].clip(
            lower=max(1e-7, px_low), upper=min(2.5, px_high)
        )

        try:
            iv_interp = interpolate_surface_linear_nearest(
                day_data.rename(columns={'ImpliedVolUsed': 'IV_For_Interp'}),
                'IV_For_Interp', LM_mesh, T_mesh
            )
            price_interp = interpolate_surface_linear_nearest(
                day_data.rename(columns={'NormalizedPrice': 'Price_For_Interp'}),
                'Price_For_Interp', LM_mesh, T_mesh
            )

            if not np.isfinite(iv_interp).all():
                skipped_days += 1
                skip_reasons['iv_interp_nonfinite'] += 1
                continue
            if not np.isfinite(price_interp).all():
                skipped_days += 1
                skip_reasons['price_interp_nonfinite'] += 1
                continue

            iv_interp = np.clip(iv_interp, 0.01, 2.5)
            price_interp = np.clip(price_interp, 1e-7, 2.5)

            iv_surfaces.append(iv_interp)
            price_surfaces.append(price_interp)
            ctx = date_to_context.get(date, [0.0, 0.0])
            context_surfaces.append(ctx)
            used_dates.append(pd.Timestamp(date))
            used_days += 1

        except Exception:
            skipped_days += 1
            skip_reasons['interp_exception'] += 1
            continue

    print(f"Used days: {used_days}")
    print(f"Skipped days: {skipped_days}")
    print("Skip reason summary:", dict(skip_reasons))

    if len(iv_surfaces) == 0:
        raise ValueError("No valid surfaces were created from raw data.")

    iv_tensor = torch.tensor(np.array(iv_surfaces), dtype=torch.float32).unsqueeze(1)
    price_tensor = torch.tensor(np.array(price_surfaces), dtype=torch.float32).unsqueeze(1)
    context_tensor = torch.tensor(np.array(context_surfaces), dtype=torch.float32)

    print("Real IV mean/std/min/max:",
          iv_tensor.mean().item(), iv_tensor.std().item(),
          iv_tensor.min().item(), iv_tensor.max().item())
    print("Real price mean/std/min/max:",
          price_tensor.mean().item(), price_tensor.std().item(),
          price_tensor.min().item(), price_tensor.max().item())
    print("Context (sentiment/articles) shape:", context_tensor.shape)

    return iv_tensor, price_tensor, context_tensor, used_dates, lm_convention


# ---------------------------------------------------------------------------
# Scalers & loaders
# ---------------------------------------------------------------------------

def recover_mean_std_from_offline_csv(csv_path, grid_shape=(8, 11)):
    df = pd.read_csv(csv_path)

    iv_cols = [col for col in df.columns if col.startswith('IV_T')]
    X_flat = df[iv_cols].values
    X_surfaces = X_flat.reshape(-1, 1, grid_shape[0], grid_shape[1])

    X_tensor = torch.tensor(X_surfaces, dtype=torch.float32)
    dummy_y = torch.zeros(len(X_tensor), 1)
    dataset = TensorDataset(X_tensor, dummy_y)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, _, _ = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    X_train = train_dataset.dataset.tensors[0][train_dataset.indices]
    X_train_mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    X_train_std = X_train.std(dim=(0, 2, 3), keepdim=True)
    X_train_std[X_train_std < 1e-7] = 1e-7

    return X_train_mean, X_train_std


def make_blended_scaler(offline_mean, offline_std, real_iv_tensor, offline_weight=0.70):
    real_mean = real_iv_tensor.mean(dim=(0, 2, 3), keepdim=True)
    real_std = real_iv_tensor.std(dim=(0, 2, 3), keepdim=True)
    real_std[real_std < 1e-7] = 1e-7

    blended_mean = offline_weight * offline_mean + (1.0 - offline_weight) * real_mean
    blended_std = offline_weight * offline_std + (1.0 - offline_weight) * real_std
    blended_std[blended_std < 1e-7] = 1e-7

    print("Offline mean/std:", offline_mean.item(), offline_std.item())
    print("Real    mean/std:", real_mean.item(), real_std.item())
    print("Blended mean/std:", blended_mean.item(), blended_std.item())

    return blended_mean, blended_std


def build_realworld_loaders(real_iv_tensor, real_price_tensor, context_tensor,
                            batch_size=16, val_fraction=0.15, seed=42):
    dataset = TensorDataset(real_iv_tensor, real_price_tensor, context_tensor)
    total_size = len(dataset)

    if total_size < 40:
        print("Dataset is small. Using all days for training and disabling validation.")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader, None

    val_size = max(1, int(val_fraction * total_size))
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def load_bounds_from_state_dict(weights_path):
    state_dict = torch.load(weights_path, map_location='cpu')
    mins = state_dict['param_mins'].cpu().tolist()
    maxs = state_dict['param_maxs'].cpu().tolist()
    return list(zip(mins, maxs))


def get_model_registry(base_dir=None):
    if base_dir is None:
        base_dir = PROJECT_ROOT
    ft = os.path.join(base_dir, 'Fine_Tuning_Files')
    registry = {
        "Bates": {
            "model_type": "Bates",
            "weights_path": os.path.join(ft, "CNN_Bates_In1_OutBase.pth"),
            "offline_csv": os.path.join(ft, "Bates_Sentiment_IV_Surface_Data_Final.csv")
        },
        "Heston": {
            "model_type": "Heston",
            "weights_path": os.path.join(ft, "CNN_Heston_In1_OutBase.pth"),
            "offline_csv": os.path.join(ft, "Heston_Sentiment_IV_Surface_Data_Final.csv")
        },
        "Bergomi": {
            "model_type": "Bergomi",
            "weights_path": os.path.join(ft, "CNN_Bergomi_In1_OutBase.pth"),
            "offline_csv": os.path.join(ft, "Bergomi_Sentiment_IV_Surface_Data_Final.csv")
        },
        "rBergomi": {
            "model_type": "rBergomi",
            "weights_path": os.path.join(ft, "CNN_rBergomi_In1_OutBase.pth"),
            "offline_csv": os.path.join(ft, "rBergomi_Sentiment_IV_Surface_Data_Final.csv")
        }
    }

    for model_name in registry:
        registry[model_name]["param_bounds"] = load_bounds_from_state_dict(
            registry[model_name]["weights_path"]
        )

    return registry


def build_model_from_config(model_config, device):
    model = VolatilitySurfaceCNN(
        param_bounds=model_config["param_bounds"], context_size=2
    ).to(device)

    state_dict = torch.load(model_config["weights_path"], map_location=device)

    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in model_state:
            continue
        if model_state[name].shape == param.shape:
            # Shapes match — copy directly
            model_state[name].copy_(param)
        elif name == "dense_block.0.weight":
            # Old weights have 2816 input features, new model has 2818 (2816 + 2 context)
            # Copy the CNN feature weights and leave context weights randomly initialized
            model_state[name][:, :param.shape[1]].copy_(param)
        # Any other size mismatch — leave as random init

    model.load_state_dict(model_state)
    return model


def run_zero_shot_evaluation(model_config, real_iv_tensor, real_price_tensor,
                             context_tensor, lm_convention, device, offline_weight=0.70):
    # Local import prevents circular dependency (trainer imports pricers, not data_utils)
    from trainer import RealWorldFineTuner

    model = build_model_from_config(model_config, device)

    offline_mean, offline_std = recover_mean_std_from_offline_csv(model_config["offline_csv"])
    X_mean, X_std = make_blended_scaler(offline_mean, offline_std, real_iv_tensor,
                                        offline_weight=offline_weight)
    X_mean = X_mean.to(device)
    X_std = X_std.to(device)

    eval_loader = DataLoader(
        TensorDataset(real_iv_tensor, real_price_tensor, context_tensor),
        batch_size=32,
        shuffle=False
    )

    tuner = RealWorldFineTuner(
        model=model,
        x_mean=X_mean,
        x_std=X_std,
        device=device,
        model_config=model_config,
        lm_convention=lm_convention
    )

    metrics = tuner.evaluate_loader(eval_loader)

    return {
        "model_name": model_config["model_type"],
        "huber": metrics["abs"],
        "relative": metrics["rel"],
        "x_mean": X_mean.detach().cpu(),
        "x_std": X_std.detach().cpu()
    }
