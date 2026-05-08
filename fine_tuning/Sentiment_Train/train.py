"""
train.py — Batched fine-tuning entry point.

Run from the SeniorDesign project root:
    python Non_Sentiment_Train/train.py
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure imports from this folder work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import matplotlib
matplotlib.use("Agg")   # non-interactive backend safe for long runs
import matplotlib.pyplot as plt
import torch

from config import (
    RUN_ZERO_SHOT_COMPARISON,
    CANDIDATE_MODELS,
    CANDIDATE_DATE_RANGES,
    OFFLINE_WEIGHT,
    EPOCHS,
    LR,
    BATCH_SIZE,
    VAL_FRACTION,
    SPLIT_SEED,
    MODEL_OUTPUT_ROOT,
    PLOT_OUTPUT_ROOT,
    get_spx_files_for_range,
)
from data_utils import (
    get_model_registry,
    build_model_from_config,
    recover_mean_std_from_offline_csv,
    make_blended_scaler,
    build_realworld_loaders,
    create_iv_and_price_grids_from_raw,
    filter_extreme_prices,
    run_zero_shot_evaluation,
)
from trainer import RealWorldFineTuner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Which models to train e.g. --models Heston Bates")
    parser.add_argument("--date-ranges", nargs="+", default=None,
                        help="Which date ranges to train e.g. --date-ranges 2010-2012 2013-2015")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def _save_training_history_plot(history_dict, model_name, date_range, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train_abs = history_dict.get("train_abs", [])
    train_rel = history_dict.get("train_rel", [])
    val_abs   = history_dict.get("val_abs", [])
    val_rel   = history_dict.get("val_rel", [])
    lr_hist   = history_dict.get("lr", [])

    axes[0].plot(range(1, len(train_abs) + 1), train_abs, marker="o", label="Train Huber")
    x_val_abs = [i + 1 for i, v in enumerate(val_abs) if v is not None]
    y_val_abs = [v for v in val_abs if v is not None]
    if y_val_abs:
        axes[0].plot(x_val_abs, y_val_abs, marker="s", label="Validation Huber")
    axes[0].set_title(f"{model_name} | {date_range} | Huber")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(train_rel) + 1), train_rel, marker="o", label="Train Relative")
    x_val_rel = [i + 1 for i, v in enumerate(val_rel) if v is not None]
    y_val_rel = [v for v in val_rel if v is not None]
    if y_val_rel:
        axes[1].plot(x_val_rel, y_val_rel, marker="s", label="Validation Relative")
    axes[1].set_title(f"{model_name} | {date_range} | Relative")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Relative")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if lr_hist:
        axes[2].plot(range(1, len(lr_hist) + 1), lr_hist, marker="o")
        axes[2].set_title(f"{model_name} | {date_range} | Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No LR history", ha="center", va="center")
        axes[2].set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

def validate_registry(model_registry):
    print("\n=== Model registry validation ===")
    all_valid = True
    for model_name in CANDIDATE_MODELS:
        if model_name not in model_registry:
            print(f"  [ERROR] {model_name} not found in MODEL_REGISTRY")
            all_valid = False
            continue
        cfg = model_registry[model_name]
        w_ok = os.path.exists(cfg["weights_path"])
        c_ok = os.path.exists(cfg["offline_csv"])
        status = "OK" if (w_ok and c_ok) else "MISSING FILES"
        n_params = len(cfg.get("param_bounds", {}))
        print(
            f"  {model_name}: {status} | {n_params} params | "
            f"weights={os.path.basename(cfg['weights_path'])} ({'OK' if w_ok else 'MISSING'}) | "
            f"csv={os.path.basename(cfg['offline_csv'])} ({'OK' if c_ok else 'MISSING'})"
        )
        if not (w_ok and c_ok):
            all_valid = False

    if not all_valid:
        raise FileNotFoundError(
            "One or more model files are missing. "
            "Check the paths printed above before starting training."
        )
    print("\nAll registry checks passed. Ready to run batched training.")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training():
    MODEL_REGISTRY = get_model_registry()
    validate_registry(MODEL_REGISTRY)

    TRAINING_RUNS     = {m: {} for m in CANDIDATE_MODELS}
    MODEL_BEFORE_RUNS = {m: {} for m in CANDIDATE_MODELS}
    DATA_CACHE_BY_RANGE = {}

    os.makedirs(MODEL_OUTPUT_ROOT, exist_ok=True)
    os.makedirs(PLOT_OUTPUT_ROOT, exist_ok=True)

    print("\nRunning one batched loop across all date ranges and all models...")

    for DATE_RANGE in CANDIDATE_DATE_RANGES:
        print(f"\n=== Preparing data for range {DATE_RANGE} ===")
        spx_files = get_spx_files_for_range(DATE_RANGE)

        print("Using SPX files:")
        for f in spx_files:
            print("  ", f)

        real_iv_tensor, real_price_tensor, context_tensor, real_dates, lm_convention = \
            create_iv_and_price_grids_from_raw(
                spx_files,
                max_days=None,
                use_calls_only=False,
                min_points_per_day=6,
                min_tau_per_day=2,
                min_lm_per_day=3
            )

        real_iv_tensor, real_price_tensor, context_tensor, real_dates = filter_extreme_prices(
            real_iv_tensor, real_price_tensor, context_tensor, real_dates,
            global_price_cap_percentile=97,
            per_point_low_pct=2,
            per_point_high_pct=98,
            day_outlier_iqr_factor=2.0
        )

        print("IV tensor shape:", real_iv_tensor.shape)
        print("Price tensor shape:", real_price_tensor.shape)
        print("Context tensor shape:", context_tensor.shape)
        print("Number of dates used:", len(real_dates))
        print("Using LogMoneyness convention:", lm_convention)

        DATA_CACHE_BY_RANGE[DATE_RANGE] = {
            "real_iv_tensor":    real_iv_tensor,
            "real_price_tensor": real_price_tensor,
            "context_tensor":    context_tensor,
            "real_dates":        real_dates,
            "lm_convention":     lm_convention,
        }

        if RUN_ZERO_SHOT_COMPARISON:
            print("\nZero-shot leaderboard for", DATE_RANGE)
            zero_rows = []
            for model_name in CANDIDATE_MODELS:
                cfg = MODEL_REGISTRY[model_name]
                z = run_zero_shot_evaluation(
                    cfg, real_iv_tensor, real_price_tensor, context_tensor,
                    lm_convention=lm_convention, device=device,
                    offline_weight=OFFLINE_WEIGHT
                )
                zero_rows.append(z)
                print(
                    f"{z['model_name']} | "
                    f"Huber: {z['huber']:.6f} | Relative: {z['relative']:.6f}"
                )
            zero_rows = sorted(zero_rows, key=lambda x: x["huber"])
            for rank, row in enumerate(zero_rows, start=1):
                print(
                    f"{rank}. {row['model_name']} | "
                    f"Huber: {row['huber']:.6f} | Relative: {row['relative']:.6f}"
                )

        for selected_model_name in CANDIDATE_MODELS:
            print(f"\n--- Training {selected_model_name} on {DATE_RANGE} ---")
            selected_config = MODEL_REGISTRY[selected_model_name]

            model_output_dir = os.path.join(MODEL_OUTPUT_ROOT, selected_model_name, DATE_RANGE)
            plot_output_dir  = os.path.join(PLOT_OUTPUT_ROOT,  selected_model_name, DATE_RANGE)
            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(plot_output_dir,  exist_ok=True)

            model = build_model_from_config(selected_config, device)
            model_before = copy.deepcopy(model)
            MODEL_BEFORE_RUNS[selected_model_name][DATE_RANGE] = {
                "model_state_dict": copy.deepcopy(model_before.state_dict()),
                "model_config":     copy.deepcopy(selected_config),
            }

            offline_mean, offline_std = recover_mean_std_from_offline_csv(
                selected_config["offline_csv"]
            )
            X_mean, X_std = make_blended_scaler(
                offline_mean, offline_std, real_iv_tensor, offline_weight=OFFLINE_WEIGHT
            )
            X_mean = X_mean.to(device)
            X_std  = X_std.to(device)

            train_loader, val_loader = build_realworld_loaders(
                real_iv_tensor, real_price_tensor, context_tensor,
                batch_size=BATCH_SIZE,
                val_fraction=VAL_FRACTION,
                seed=SPLIT_SEED
            )

            tuner = RealWorldFineTuner(
                model=model,
                x_mean=X_mean,
                x_std=X_std,
                device=device,
                model_config=selected_config,
                lm_convention=lm_convention
            )

            best_model_path = os.path.join(model_output_dir, "best_model.pth")
            history, best_epoch, best_loss = tuner.fine_tune(
                train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LR,
                weight_decay=1e-5,
                scheduler_start_epoch=15,
                scheduler_patience=20,
                scheduler_factor=0.7,
                scheduler_cooldown=2,
                scheduler_min_lr=1e-6,
                best_model_path=best_model_path
            )

            training_plot_path = os.path.join(plot_output_dir, "training_history.png")
            _save_training_history_plot(history, selected_model_name, DATE_RANGE, training_plot_path)

            TRAINING_RUNS[selected_model_name][DATE_RANGE] = {
                "history":            history,
                "best_epoch":         best_epoch,
                "best_loss":          best_loss,
                "best_model_path":    best_model_path,
                "training_plot_path": training_plot_path,
                "x_mean": X_mean.detach().cpu(),
                "x_std":  X_std.detach().cpu(),
            }

            print(f"Saved model artifacts to: {model_output_dir}")
            print(f"Saved plot artifacts to:  {plot_output_dir}")
            print(f"Best epoch: {best_epoch} | Best loss: {best_loss:.6f}")

    print("\n=== Batched training summary ===")
    for model_name in CANDIDATE_MODELS:
        model_runs = TRAINING_RUNS.get(model_name, {})
        if not model_runs:
            continue
        print(f"\n{model_name}:")
        for date_range, run_info in sorted(model_runs.items()):
            print(
                f"  {date_range} | Best epoch: {run_info['best_epoch']} | "
                f"Best loss: {run_info['best_loss']:.6f}"
            )

    return TRAINING_RUNS, MODEL_BEFORE_RUNS, DATA_CACHE_BY_RANGE


if __name__ == "__main__":
    args = parse_args()
    if args.models:
        CANDIDATE_MODELS[:] = [m for m in CANDIDATE_MODELS if m in args.models]
    if args.date_ranges:
        CANDIDATE_DATE_RANGES[:] = [d for d in CANDIDATE_DATE_RANGES if d in args.date_ranges]

    print(f"Using device: {device}")
    print(f"Training models: {CANDIDATE_MODELS}")
    print(f"Date ranges: {CANDIDATE_DATE_RANGES}")
    run_training()
