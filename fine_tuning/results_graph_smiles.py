"""
results_graph_smiles.py — Cross-model smile / term-structure comparison.

For each date range loads real-world market data and every available trained
checkpoint (non-sentiment + sentiment variants), then plots smile and ATM
term-structure curves for all models overlaid on the same representative day.

Run from the full_project directory:
    python results_graph_smiles.py

Plots are saved to:
    <workspace>/full_project/fine_tuning_output/base_output/plots/smiles/<date_range>/
        all_models_smile.png       — Actual + all 8 model predictions
        nonsent_models_smile.png   — Actual + 4 non-sentiment predictions
        sent_models_smile.png      — Actual + 4 sentiment predictions
"""

import glob as _glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE           = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT    = os.path.join(HERE, "fine_tuning_output")
NONSENT_DIR    = os.path.join(HERE, "Non_Sentiment_Train")
SENT_DIR       = os.path.join(HERE, "Sentiment_Train")
PLOT_ROOT      = os.path.join(OUTPUT_ROOT, "base_output", "plots", "smiles")
NONSENT_MODELS = os.path.join(OUTPUT_ROOT, "base_output", "models")
SENT_MODELS    = os.path.join(OUTPUT_ROOT, "sentiment_output", "models")
DATA_FILES_DIR = os.path.join(HERE, "Fine_Tuning_Files", "SPX_Data")

CANDIDATE_MODELS      = ["Heston", "Bates", "Bergomi", "rBergomi"]
CANDIDATE_DATE_RANGES = ["2010-2012", "2013-2015", "2016-2019", "2020-2022"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module names that both packages share — must be managed carefully in sys.modules.
_PKG_NAMES = ["utils", "pricers", "model", "data_utils", "trainer", "config"]


# ---------------------------------------------------------------------------
# Package loading helpers
# ---------------------------------------------------------------------------

def _clear_pkg_modules():
    """Remove shared-name modules from sys.modules so the next import is fresh."""
    for name in _PKG_NAMES:
        sys.modules.pop(name, None)


def _load_pkg(pkg_dir):
    """
    Insert pkg_dir at front of sys.path, clear cached same-named modules,
    import the package's core modules, return dict of module objects, then
    pop pkg_dir from sys.path.

    The returned module objects remain valid (and use their own globals)
    even after pkg_dir is removed from sys.path and sys.modules is cleared.
    """
    _clear_pkg_modules()
    sys.path.insert(0, pkg_dir)
    try:
        import importlib
        mods = {}
        for name in ("utils", "model", "data_utils", "trainer"):
            fpath = os.path.join(pkg_dir, f"{name}.py")
            if os.path.exists(fpath):
                mods[name] = importlib.import_module(name)
        return mods
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# One-time load: capture stable function / class references from each package.
# Once captured, these references remain correct regardless of what is in
# sys.modules, because Python functions reference their own module's __globals__.
# ---------------------------------------------------------------------------

print("Loading Non-Sentiment package ...")
_ns = _load_pkg(NONSENT_DIR)
ns_get_registry       = _ns["data_utils"].get_model_registry
ns_build_model        = _ns["data_utils"].build_model_from_config
ns_RealWorldFineTuner = _ns["trainer"].RealWorldFineTuner

print("Loading Sentiment package ...")
_st = _load_pkg(SENT_DIR)
st_get_registry              = _st["data_utils"].get_model_registry
st_build_model               = _st["data_utils"].build_model_from_config
st_RealWorldFineTuner        = _st["trainer"].RealWorldFineTuner
st_create_iv_and_price_grids = _st["data_utils"].create_iv_and_price_grids_from_raw
st_filter_extreme_prices     = _st["data_utils"].filter_extreme_prices

# Build model registries (reads param_bounds from pre-trained .pth files).
NS_REGISTRY = ns_get_registry(base_dir=HERE)
ST_REGISTRY = st_get_registry(base_dir=HERE)

# Leave sys.modules in a neutral state so nothing leaks later.
_clear_pkg_modules()


# ---------------------------------------------------------------------------
# SPX file discovery (replicated to avoid importing config.py side-effects)
# ---------------------------------------------------------------------------

def get_spx_files_for_range(date_range: str):
    start_yr, end_yr = [int(p) for p in date_range.split("-")]
    files = sorted(
        [
            Path(f)
            for f in _glob.glob(
                os.path.join(DATA_FILES_DIR, "cleanerSPXData_*_merged.csv")
            )
            if start_yr <= int(Path(f).stem.split("_")[1]) <= end_yr
        ],
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for {date_range} in: {DATA_FILES_DIR}"
        )
    return files


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_nonsent_ckpt(model_name, date_range):
    path = os.path.join(NONSENT_MODELS, model_name, date_range, "best_model.pth")
    return path if os.path.exists(path) else None


def _find_sent_ckpt(model_name, date_range):
    model_dir  = os.path.join(SENT_MODELS, model_name, date_range)
    candidates = sorted(_glob.glob(os.path.join(model_dir, "best_model_epoch_*.pth")))
    if not candidates:
        fb = os.path.join(model_dir, "best_model.pth")
        return fb if os.path.exists(fb) else None
    return candidates[-1]  # highest epoch


# ---------------------------------------------------------------------------
# Single-day prediction
# ---------------------------------------------------------------------------

def _predict_nonsent(model_name, ckpt_path, iv_day, lm_conv):
    """
    Return predicted price surface [8, 11] for *iv_day* using the
    non-sentiment fine-tuned checkpoint.

    iv_day : torch.Tensor  [1, 1, 8, 11]
    """
    ckpt            = torch.load(ckpt_path, map_location=device)
    selected_config = NS_REGISTRY[model_name]
    model           = ns_build_model(selected_config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tuner = ns_RealWorldFineTuner(
        model=model,
        x_mean=ckpt["x_mean"].to(device),
        x_std=ckpt["x_std"].to(device),
        device=device,
        model_config=selected_config,
        lm_convention=lm_conv,
        train_mc_settings=ckpt.get("train_mc_settings"),
        eval_mc_settings=ckpt.get("eval_mc_settings"),
    )

    x = iv_day.to(device, dtype=torch.float32)
    with torch.no_grad():
        scaled      = tuner.scale_input(x)
        _, params   = tuner.model(scaled)
        surface     = tuner.calculate_model_price_surface(params, for_eval=True)

    return surface.detach().cpu().numpy()[0, 0]  # [8, 11]


def _predict_sent(model_name, ckpt_path, iv_day, context_day, lm_conv):
    """
    Return predicted price surface [8, 11] for *iv_day* using the
    sentiment-aware fine-tuned checkpoint.

    iv_day      : torch.Tensor  [1, 1, 8, 11]
    context_day : torch.Tensor  [1, 2]
    """
    ckpt            = torch.load(ckpt_path, map_location=device)
    selected_config = ST_REGISTRY[model_name]
    model           = st_build_model(selected_config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tuner = st_RealWorldFineTuner(
        model=model,
        x_mean=ckpt["x_mean"].to(device),
        x_std=ckpt["x_std"].to(device),
        device=device,
        model_config=selected_config,
        lm_convention=lm_conv,
        train_mc_settings=ckpt.get("train_mc_settings"),
        eval_mc_settings=ckpt.get("eval_mc_settings"),
    )

    x   = iv_day.to(device, dtype=torch.float32)
    ctx = context_day.to(device, dtype=torch.float32)
    with torch.no_grad():
        scaled      = tuner.scale_input(x)
        _, params   = tuner.model(scaled, ctx)
        surface     = tuner.calculate_model_price_surface(params, for_eval=True)

    return surface.detach().cpu().numpy()[0, 0]  # [8, 11]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

LM_GRID     = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
TAU_GRID    = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])
ATM_IDX     = int(np.argmin(np.abs(LM_GRID - 0.0)))
MID_TAU_IDX = len(TAU_GRID) // 2


def _plot_smile_comparison(actual_surface, predictions, date_label,
                            title_prefix, save_path):
    """
    Two-panel figure: volatility smile slice (left) and ATM term structure (right).

    actual_surface : np.ndarray [8, 11]
    predictions    : dict  label -> np.ndarray [8, 11]
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # --- Left: smile slice at mid maturity ---
    axes[0].plot(
        LM_GRID, actual_surface[MID_TAU_IDX],
        marker="o", linewidth=2.5, color="black", label="Actual",
    )
    for label, surf in predictions.items():
        axes[0].plot(LM_GRID, surf[MID_TAU_IDX], marker="s", linewidth=1.8, label=label)
    axes[0].set_title(
        f"{title_prefix} | {date_label} | "
        f"Smile Slice  (τ = {TAU_GRID[MID_TAU_IDX]:.2f})"
    )
    axes[0].set_xlabel("Log-Moneyness")
    axes[0].set_ylabel("Normalized Price")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend(fontsize=8)

    # --- Right: ATM term structure ---
    axes[1].plot(
        TAU_GRID, actual_surface[:, ATM_IDX],
        marker="o", linewidth=2.5, color="black", label="Actual",
    )
    for label, surf in predictions.items():
        axes[1].plot(TAU_GRID, surf[:, ATM_IDX], marker="s", linewidth=1.8, label=label)
    axes[1].set_title(f"{title_prefix} | {date_label} | ATM Term Structure")
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("Normalized Price")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nDevice: {device}")
    print(f"Plots will be saved under: {PLOT_ROOT}\n")

    for date_range in CANDIDATE_DATE_RANGES:
        print(f"\n{'=' * 64}")
        print(f"  Date range: {date_range}")
        print(f"{'=' * 64}")

        # --- Load market data (sentiment loader also extracts context) ---
        spx_files = get_spx_files_for_range(date_range)
        iv, price, context, dates, lm_conv = st_create_iv_and_price_grids(
            spx_files,
            max_days=None,
            use_calls_only=False,
            min_points_per_day=6,
            min_tau_per_day=2,
            min_lm_per_day=3,
        )
        iv, price, context, dates = st_filter_extreme_prices(
            iv, price, context, dates,
            global_price_cap_percentile=97,
            per_point_low_pct=2,
            per_point_high_pct=98,
            day_outlier_iqr_factor=2.0,
        )

        # Representative day: middle of the dataset by date order
        day_idx        = len(dates) // 2
        date_label     = pd.to_datetime(dates[day_idx]).date()
        actual_surface = price[day_idx, 0].numpy()       # [8, 11]
        iv_day         = iv[day_idx : day_idx + 1]        # [1, 1, 8, 11]
        context_day    = context[day_idx : day_idx + 1]   # [1, 2]

        print(f"\n  Representative day: {date_label}  "
              f"(index {day_idx} / {len(dates) - 1})")

        plot_dir      = os.path.join(PLOT_ROOT, date_range)
        nonsent_preds = {}   # model_name -> surface [8, 11]
        sent_preds    = {}

        for model_name in CANDIDATE_MODELS:

            # Non-sentiment
            ckpt_ns = _find_nonsent_ckpt(model_name, date_range)
            if ckpt_ns:
                print(f"  [NS] {model_name}: {os.path.basename(ckpt_ns)}", end=" ... ")
                try:
                    nonsent_preds[model_name] = _predict_nonsent(
                        model_name, ckpt_ns, iv_day, lm_conv
                    )
                    print("OK")
                except Exception as exc:
                    print(f"FAILED — {exc}")
            else:
                print(f"  [NS] {model_name}: checkpoint not found — skipping")

            # Sentiment
            ckpt_st = _find_sent_ckpt(model_name, date_range)
            if ckpt_st:
                print(f"  [S]  {model_name}: {os.path.basename(ckpt_st)}", end=" ... ")
                try:
                    sent_preds[model_name] = _predict_sent(
                        model_name, ckpt_st, iv_day, context_day, lm_conv
                    )
                    print("OK")
                except Exception as exc:
                    print(f"FAILED — {exc}")
            else:
                print(f"  [S]  {model_name}: checkpoint not found — skipping")

        # Build combined prediction dict (NS first, then S)
        all_preds = {f"{m} (NS)": s for m, s in nonsent_preds.items()}
        all_preds.update({f"{m} (S)": s for m, s in sent_preds.items()})

        os.makedirs(plot_dir, exist_ok=True)

        if all_preds:
            _plot_smile_comparison(
                actual_surface, all_preds, date_label,
                title_prefix="All Models",
                save_path=os.path.join(plot_dir, "all_models_smile.png"),
            )

        if nonsent_preds:
            _plot_smile_comparison(
                actual_surface, nonsent_preds, date_label,
                title_prefix="Non-Sentiment Models",
                save_path=os.path.join(plot_dir, "nonsent_models_smile.png"),
            )

        if sent_preds:
            _plot_smile_comparison(
                actual_surface, sent_preds, date_label,
                title_prefix="Sentiment Models",
                save_path=os.path.join(plot_dir, "sent_models_smile.png"),
            )

    print(f"\nAll smile plots saved under: {PLOT_ROOT}")


if __name__ == "__main__":
    main()