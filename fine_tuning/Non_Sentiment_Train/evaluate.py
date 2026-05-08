"""
evaluate.py — Post-training evaluation and plotting.

Run from the SeniorDesign project root:
    python Non_Sentiment_Train/evaluate.py

Edit SELECTED_MODEL_NAME and PLOT_DATE_RANGE at the bottom of this file
to choose which trained model to inspect.
"""

import sys
import os

# Ensure imports from this folder work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from config import (
    MODEL_OUTPUT_ROOT,
    PLOT_OUTPUT_ROOT,
    OFFLINE_WEIGHT,
    RUN_ZERO_SHOT_COMPARISON,
    CANDIDATE_MODELS,
    CANDIDATE_DATE_RANGES,
    get_spx_files_for_range,
)
from data_utils import (
    get_model_registry,
    build_model_from_config,
    recover_mean_std_from_offline_csv,
    make_blended_scaler,
    create_iv_and_price_grids_from_raw,
    filter_extreme_prices,
)
from trainer import RealWorldFineTuner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_eval_seed(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training history plot (standalone / for .npz files)
# ---------------------------------------------------------------------------

def _valid_points(values):
    xs, ys = [], []
    for i, v in enumerate(values, start=1):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            xs.append(i)
            ys.append(v)
    return xs, ys


def plot_training_history(history_dict, model_name, zero_shot_results=None,
                          save_path=None, zero_shot_save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(
        range(1, len(history_dict["train_abs"]) + 1),
        history_dict["train_abs"],
        marker='o', label="Train Huber"
    )
    x_val_abs, y_val_abs = _valid_points(history_dict.get("val_abs", []))
    if y_val_abs:
        axes[0].plot(x_val_abs, y_val_abs, marker='s', label="Validation Huber")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].set_title(f"{model_name} Huber Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        range(1, len(history_dict["train_rel"]) + 1),
        history_dict["train_rel"],
        marker='o', label="Train Relative Metric"
    )
    x_val_rel, y_val_rel = _valid_points(history_dict.get("val_rel", []))
    if y_val_rel:
        axes[1].plot(x_val_rel, y_val_rel, marker='s', label="Validation Relative Metric")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_title(f"{model_name} Relative Error")
    axes[1].grid(True)
    axes[1].legend()

    lr_hist = history_dict.get("lr", [])
    if len(lr_hist) > 0:
        axes[2].plot(range(1, len(lr_hist) + 1), lr_hist, marker='o')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title(f"{model_name} Learning Rate")
        axes[2].grid(True)
    else:
        axes[2].text(0.5, 0.5, "No LR history found", ha='center', va='center')
        axes[2].set_axis_off()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    if zero_shot_results is not None and len(zero_shot_results) > 0:
        zero_shot_sorted = sorted(zero_shot_results, key=lambda x: x["huber"])
        names    = [row["model_name"] for row in zero_shot_sorted]
        hubers   = [row["huber"]      for row in zero_shot_sorted]
        relatives = [row["relative"]  for row in zero_shot_sorted]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(names, hubers)
        axes[0].set_title("Zero-Shot Huber Comparison")
        axes[0].set_ylabel("Huber Loss")
        axes[0].grid(True, axis='y', alpha=0.3)

        axes[1].bar(names, relatives)
        axes[1].set_title("Zero-Shot Relative Error Comparison")
        axes[1].set_ylabel("Relative Error")
        axes[1].grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        if zero_shot_save_path:
            plt.savefig(zero_shot_save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


# ---------------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------------

def extract_predictions_and_surfaces(tuner_obj, data_loader, seed=1234):
    set_eval_seed(seed)
    tuner_obj.model.eval()

    iv_surfaces     = []
    actual_surfaces = []
    pred_surfaces   = []

    with torch.no_grad():
        for market_iv_surface, market_price_surface in data_loader:
            market_iv_surface    = market_iv_surface.to(tuner_obj.device, dtype=torch.float32)
            market_price_surface = market_price_surface.to(tuner_obj.device, dtype=torch.float32)

            scaled_input = tuner_obj.scale_input(market_iv_surface)
            _, predicted_params = tuner_obj.model(scaled_input)
            model_generated_price_surface = tuner_obj.calculate_model_price_surface(
                predicted_params, for_eval=True
            )

            iv_surfaces.append(market_iv_surface.cpu())
            actual_surfaces.append(market_price_surface.cpu())
            pred_surfaces.append(model_generated_price_surface.cpu())

    iv_surfaces     = torch.cat(iv_surfaces, dim=0).numpy()
    actual_surfaces = torch.cat(actual_surfaces, dim=0).numpy()
    pred_surfaces   = torch.cat(pred_surfaces, dim=0).numpy()

    return iv_surfaces, actual_surfaces, pred_surfaces


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize_metrics(actual_surfaces, pred_surfaces):
    actual_t = torch.tensor(actual_surfaces, dtype=torch.float32)
    pred_t   = torch.tensor(pred_surfaces,   dtype=torch.float32)

    diff  = pred_surfaces - actual_surfaces
    denom = np.clip(np.abs(actual_surfaces), a_min=0.05, a_max=None)

    return {
        "huber":    float(F.huber_loss(pred_t, actual_t, delta=0.05).item()),
        "mae":      float(np.mean(np.abs(diff))),
        "rmse":     float(np.sqrt(np.mean(diff ** 2))),
        "relative": float(np.mean(np.abs(diff) / denom))
    }


def flatten_prices(actual_surfaces, pred_surfaces):
    actual_flat = actual_surfaces.reshape(-1)
    pred_flat   = pred_surfaces.reshape(-1)
    denom       = np.clip(np.abs(actual_flat), a_min=0.05, a_max=None)
    rel_errors  = np.abs(actual_flat - pred_flat) / denom
    return actual_flat, pred_flat, rel_errors


def surface_level_relative_errors(actual_surfaces, pred_surfaces):
    denom = np.clip(np.abs(actual_surfaces), a_min=0.05, a_max=None)
    rel = np.abs(pred_surfaces - actual_surfaces) / denom
    return rel.mean(axis=(1, 2, 3))


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_parity_hist_and_metric_bars(actual_flat, pred_flat, rel_errors,
                                     fine_metrics, zero_metrics=None, save_path=None):
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lower = float(min(actual_flat.min(), pred_flat.min()))
    upper = float(max(actual_flat.max(), pred_flat.max()))
    h = axes[0].hist2d(
        actual_flat, pred_flat,
        bins=150,
        range=[[lower, upper], [lower, upper]],
        cmap='inferno',
        norm=LogNorm(),
        cmin=1
    )
    plt.colorbar(h[3], ax=axes[0], label='Count')
    axes[0].plot([lower, upper], [lower, upper], 'w--', lw=2, label='Perfect Fit')
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xlabel("Actual Prices")
    axes[0].set_ylabel("Predicted Prices")
    axes[0].set_title("Actual vs Predicted Prices (Density)")
    axes[0].legend()

    axes[1].hist(rel_errors, bins=50, edgecolor='black')
    axes[1].set_title("Distribution of Relative Errors")
    axes[1].set_xlabel("Relative Error")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    metric_names = ["Huber", "RMSE", "Relative"]
    fine_vals = [fine_metrics["huber"], fine_metrics["rmse"], fine_metrics["relative"]]

    if zero_metrics is not None:
        zero_vals = [zero_metrics["huber"], zero_metrics["rmse"], zero_metrics["relative"]]
        x = np.arange(len(metric_names))
        width = 0.35
        axes[2].bar(x - width / 2, zero_vals, width, label="Zero-Shot")
        axes[2].bar(x + width / 2, fine_vals, width, label="Fine-Tuned")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(metric_names)
        axes[2].legend()
        axes[2].set_title("Zero-Shot vs Fine-Tuned")
    else:
        axes[2].bar(metric_names, fine_vals)
        axes[2].set_title("Fine-Tuned Metrics")

    axes[2].grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_surface_heatmaps(iv_surface, actual_surface, pred_surface, tuner,
                          title_prefix="", save_path=None):
    error_surface = pred_surface - actual_surface

    vmin = min(actual_surface.min(), pred_surface.min())
    vmax = max(actual_surface.max(), pred_surface.max())
    err_abs = max(float(np.max(np.abs(error_surface))), 1e-8)

    extent = [
        float(tuner.log_moneyness.min().cpu().item()),
        float(tuner.log_moneyness.max().cpu().item()),
        float(tuner.maturities.min().cpu().item()),
        float(tuner.maturities.max().cpu().item())
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    im0 = axes[0].imshow(iv_surface, aspect='auto', origin='lower', extent=extent)
    axes[0].set_title(f"{title_prefix}IV Surface")
    axes[0].set_xlabel("Log-Moneyness")
    axes[0].set_ylabel("Maturity")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(actual_surface, aspect='auto', origin='lower',
                         extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix}Actual Prices")
    axes[1].set_xlabel("Log-Moneyness")
    axes[1].set_ylabel("Maturity")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(pred_surface, aspect='auto', origin='lower',
                         extent=extent, vmin=vmin, vmax=vmax)
    axes[2].set_title(f"{title_prefix}Predicted Prices")
    axes[2].set_xlabel("Log-Moneyness")
    axes[2].set_ylabel("Maturity")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(error_surface, aspect='auto', origin='lower',
                         extent=extent, vmin=-err_abs, vmax=err_abs)
    axes[3].set_title(f"{title_prefix}Prediction Error")
    axes[3].set_xlabel("Log-Moneyness")
    axes[3].set_ylabel("Maturity")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_smile_and_term_structure(actual_surface, pred_surface, tuner,
                                  title_prefix="", save_path=None):
    lm_grid  = tuner.log_moneyness.detach().cpu().numpy()
    tau_grid = tuner.maturities.detach().cpu().numpy()

    atm_idx      = int(np.argmin(np.abs(lm_grid - 0.0)))
    mid_tau_idx  = len(tau_grid) // 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(lm_grid, actual_surface[mid_tau_idx], marker='o', label='Actual')
    axes[0].plot(lm_grid, pred_surface[mid_tau_idx],   marker='s', label='Predicted')
    axes[0].set_title(f"{title_prefix}Smile Slice at Tau={tau_grid[mid_tau_idx]:.2f}")
    axes[0].set_xlabel("Log-Moneyness")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    axes[1].plot(tau_grid, actual_surface[:, atm_idx], marker='o', label='Actual')
    axes[1].plot(tau_grid, pred_surface[:, atm_idx],   marker='s', label='Predicted')
    axes[1].set_title(f"{title_prefix}ATM Term Structure")
    axes[1].set_xlabel("Maturity")
    axes[1].set_ylabel("Price")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_aggregate_error_heatmaps(actual_surfaces, pred_surfaces, tuner, save_path=None):
    abs_err = np.mean(np.abs(pred_surfaces - actual_surfaces), axis=0).squeeze(0)
    denom   = np.clip(np.abs(actual_surfaces), a_min=0.05, a_max=None)
    rel_err = np.mean(np.abs(pred_surfaces - actual_surfaces) / denom, axis=0).squeeze(0)

    extent = [
        float(tuner.log_moneyness.min().cpu().item()),
        float(tuner.log_moneyness.max().cpu().item()),
        float(tuner.maturities.min().cpu().item()),
        float(tuner.maturities.max().cpu().item())
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(abs_err, aspect='auto', origin='lower', extent=extent)
    axes[0].set_title("Mean Absolute Error Heatmap")
    axes[0].set_xlabel("Log-Moneyness")
    axes[0].set_ylabel("Maturity")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rel_err, aspect='auto', origin='lower', extent=extent)
    axes[1].set_title("Mean Relative Error Heatmap")
    axes[1].set_xlabel("Log-Moneyness")
    axes[1].set_ylabel("Maturity")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _evaluate_one(model_name, date_range, MODEL_REGISTRY, data_cache):
    """Evaluate a single model/date_range combination and save all plots."""
    model_dir       = os.path.join(MODEL_OUTPUT_ROOT, model_name, date_range)
    checkpoint_path = os.path.join(model_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] No checkpoint: {checkpoint_path}")
        return None

    plot_dir = os.path.join(PLOT_OUTPUT_ROOT, model_name, date_range)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n=== {model_name} | {date_range} ===")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"  Loaded epoch {checkpoint['epoch']} | loss {checkpoint['best_loss']:.6f}")

    selected_config = MODEL_REGISTRY[model_name]
    plot_model = build_model_from_config(selected_config, device)
    plot_model.load_state_dict(checkpoint["model_state_dict"])
    X_mean = checkpoint["x_mean"].to(device)
    X_std  = checkpoint["x_std"].to(device)

    # Reuse cached market data for this date range
    real_iv_tensor    = data_cache["real_iv_tensor"]
    real_price_tensor = data_cache["real_price_tensor"]
    real_dates        = data_cache["real_dates"]
    lm_convention     = data_cache["lm_convention"]

    tuner = RealWorldFineTuner(
        model=plot_model,
        x_mean=X_mean,
        x_std=X_std,
        device=device,
        model_config=selected_config,
        lm_convention=lm_convention,
        train_mc_settings=checkpoint.get("train_mc_settings"),
        eval_mc_settings=checkpoint.get("eval_mc_settings")
    )

    full_loader = DataLoader(
        TensorDataset(real_iv_tensor, real_price_tensor),
        batch_size=16, shuffle=False
    )

    iv_surfaces, actual_surfaces, pred_surfaces = extract_predictions_and_surfaces(
        tuner, full_loader, seed=1234
    )

    fine_metrics = summarize_metrics(actual_surfaces, pred_surfaces)
    actual_flat, pred_flat, rel_errors = flatten_prices(actual_surfaces, pred_surfaces)

    print(f"  Metrics: " + " | ".join(f"{k}: {v:.6f}" for k, v in fine_metrics.items()))

    # Zero-shot comparison
    zero_metrics = None
    if RUN_ZERO_SHOT_COMPARISON:
        zero_model = build_model_from_config(selected_config, device)
        zero_tuner = RealWorldFineTuner(
            model=zero_model, x_mean=X_mean, x_std=X_std,
            device=device, model_config=selected_config,
            lm_convention=lm_convention,
            train_mc_settings=checkpoint.get("train_mc_settings"),
            eval_mc_settings=checkpoint.get("eval_mc_settings")
        )
        _, actual_zero, pred_zero = extract_predictions_and_surfaces(
            zero_tuner, full_loader, seed=1234
        )
        zero_metrics = summarize_metrics(actual_zero, pred_zero)

    # Representative day selection
    surface_rel = surface_level_relative_errors(actual_surfaces, pred_surfaces)
    rep_idx   = int(np.argsort(surface_rel)[len(surface_rel) // 2])
    best_idx  = int(np.argmin(surface_rel))
    worst_idx = int(np.argmax(surface_rel))
    date_array = pd.to_datetime(real_dates)

    print(f"  Best day:  {date_array[best_idx].date()} (rel={surface_rel[best_idx]:.4f})")
    print(f"  Median day:{date_array[rep_idx].date()}  (rel={surface_rel[rep_idx]:.4f})")
    print(f"  Worst day: {date_array[worst_idx].date()} (rel={surface_rel[worst_idx]:.4f})")

    tag = f"{model_name}_{date_range}"

    # Training history
    history = checkpoint.get("history", {})
    if history:
        plot_training_history(
            history, model_name,
            save_path=os.path.join(plot_dir, "training_history.png"),
            zero_shot_save_path=os.path.join(plot_dir, "zero_shot_comparison.png")
        )

    # Parity / histogram / metric bars
    plot_parity_hist_and_metric_bars(
        actual_flat, pred_flat, rel_errors, fine_metrics, zero_metrics=zero_metrics,
        save_path=os.path.join(plot_dir, "parity_and_metrics.png")
    )

    # Surface heatmaps for best / median / worst days
    for label, idx in [("best", best_idx), ("median", rep_idx), ("worst", worst_idx)]:
        day_str = str(date_array[idx].date())
        plot_surface_heatmaps(
            iv_surface=iv_surfaces[idx, 0],
            actual_surface=actual_surfaces[idx, 0],
            pred_surface=pred_surfaces[idx, 0],
            tuner=tuner,
            title_prefix=f"{model_name} | {day_str} | ",
            save_path=os.path.join(plot_dir, f"surface_heatmap_{label}.png")
        )
        plot_smile_and_term_structure(
            actual_surface=actual_surfaces[idx, 0],
            pred_surface=pred_surfaces[idx, 0],
            tuner=tuner,
            title_prefix=f"{model_name} | {day_str} | ",
            save_path=os.path.join(plot_dir, f"smile_term_{label}.png")
        )

    # Aggregate error heatmaps
    plot_aggregate_error_heatmaps(
        actual_surfaces, pred_surfaces, tuner,
        save_path=os.path.join(plot_dir, "aggregate_error_heatmaps.png")
    )

    print(f"  Plots saved to: {plot_dir}")
    return fine_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving

    print(f"Using device: {device}")
    MODEL_REGISTRY = get_model_registry()

    # Cache market data per date range (shared across models)
    data_cache_by_range = {}
    for date_range in CANDIDATE_DATE_RANGES:
        spx_files = get_spx_files_for_range(date_range)
        iv, price, dates, lm_conv = create_iv_and_price_grids_from_raw(
            spx_files, max_days=None, use_calls_only=False,
            min_points_per_day=6, min_tau_per_day=2, min_lm_per_day=3
        )
        iv, price, dates = filter_extreme_prices(
            iv, price, dates,
            global_price_cap_percentile=97, per_point_low_pct=2,
            per_point_high_pct=98, day_outlier_iqr_factor=2.0
        )
        data_cache_by_range[date_range] = {
            "real_iv_tensor":    iv,
            "real_price_tensor": price,
            "real_dates":        dates,
            "lm_convention":     lm_conv,
        }

    # Evaluate all models × all date ranges
    all_metrics = {}
    for model_name in CANDIDATE_MODELS:
        for date_range in CANDIDATE_DATE_RANGES:
            metrics = _evaluate_one(
                model_name, date_range,
                MODEL_REGISTRY, data_cache_by_range[date_range]
            )
            if metrics is not None:
                all_metrics[f"{model_name}_{date_range}"] = metrics

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Model':<12} {'Range':<12} {'Huber':>10} {'MAE':>10} {'RMSE':>10} {'Relative':>10}")
    print("-" * 72)
    for key, m in sorted(all_metrics.items()):
        parts = key.rsplit("_", 1)
        mname, drange = parts[0], parts[1] if len(parts) == 2 else ("", key)
        # Handle model names without underscore ambiguity
        for mn in CANDIDATE_MODELS:
            if key.startswith(mn):
                mname  = mn
                drange = key[len(mn)+1:]
                break
        print(f"{mname:<12} {drange:<12} {m['huber']:>10.6f} {m['mae']:>10.6f} "
              f"{m['rmse']:>10.6f} {m['relative']:>10.6f}")
    print("=" * 72)
    print("\nAll evaluations complete.")
