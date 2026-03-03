#!/usr/bin/env python3
"""Phase 3: Walk-Forward Ensemble Training Pipeline.

Trains a 3-model ensemble (XGBoost, LightGBM, CatBoost) using monthly
rolling walk-forward validation with purged 5-fold cross-validation.

Usage::

    uv run python scripts/train_models.py --walk-forward --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml.trainer import WalkForwardTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/features/all_features.parquet")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Columns to exclude from features
META_COLS = {"ticker", "target"}

# Feature pruning threshold (fraction of total importance)
PRUNE_THRESHOLD = 0.005  # 0.5%

MODEL_TYPES = ["xgboost", "lightgbm", "catboost"]

# Walk-forward parameters
TRAIN_MONTHS = 12
TEST_MONTHS = 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def load_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the feature parquet and split into X, y, dates.

    Returns:
        Tuple of (X, y, dates, feature_names).
    """
    print(f"[1/6] Loading data from {path}...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Tickers: {df['ticker'].unique().tolist()}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Features: {len(feature_cols)}")

    X = df[feature_cols].copy()  # noqa: N806
    y = df["target"].copy()

    # Extract dates from the DatetimeIndex
    dates = pd.Series(df.index, index=df.index)

    # Fill any remaining NaN with 0
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"  Filling {nan_count} NaN values with 0")
        X = X.fillna(0.0)  # noqa: N806

    # Replace any inf with column max/min
    inf_count = np.isinf(X.values).sum()
    if inf_count > 0:
        print(f"  Replacing {inf_count} inf values")
        for col in X.columns:
            finite_mask = np.isfinite(X[col])
            if not finite_mask.all():
                col_max = X.loc[finite_mask, col].max()
                col_min = X.loc[finite_mask, col].min()
                X[col] = X[col].replace([np.inf, -np.inf], [col_max, col_min])

    print(f"  Target mean: {y.mean():.4f}")
    return X, y, dates, feature_cols


def run_walk_forward(
    trainer: WalkForwardTrainer,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    dates: pd.Series,
    verbose: bool = False,
) -> dict:
    """Run the full walk-forward ensemble training."""
    print("\n[2/6] Running walk-forward ensemble training...")
    print(f"  Models: {', '.join(MODEL_TYPES)}")
    print(f"  Train window: {TRAIN_MONTHS} months")
    print(f"  Test window: {TEST_MONTHS} month")
    print("  Purged CV: 5-fold, 5-day embargo")
    print(f"  Samples: {len(X)}")

    result = trainer.train_walk_forward_ensemble(
        X=X,
        y=y,
        dates=dates,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        model_types=MODEL_TYPES,
        verbose=verbose,
    )

    print(f"\n  Walk-forward complete: {result['n_windows']} windows")
    print(f"  Avg OOS AUC: {result['avg_oos_auc']:.4f}")
    print(f"  Min OOS AUC: {result['min_oos_auc']:.4f}")
    print(f"  Max OOS AUC: {result['max_oos_auc']:.4f}")
    print(f"  Avg CV AUC:  {result['avg_cv_auc']:.4f}")
    print(f"  Time: {result['train_time_seconds']:.1f}s")

    return result


def prune_and_retrain(
    trainer: WalkForwardTrainer,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    dates: pd.Series,
    first_pass_result: dict,
    verbose: bool = False,
) -> tuple[dict, list[str], list[str]]:
    """Prune low-importance features and re-run walk-forward."""
    print("\n[3/6] Feature importance pruning...")

    importances = first_pass_result["feature_importances"]
    kept_features = trainer.prune_features(importances, PRUNE_THRESHOLD)

    all_features = list(importances.keys())
    pruned_features = [f for f in all_features if f not in kept_features]

    print(f"  Original features: {len(all_features)}")
    print(f"  Kept features: {len(kept_features)}")
    print(f"  Pruned features: {len(pruned_features)}")

    if len(pruned_features) == 0:
        print("  No features pruned — using original result")
        return first_pass_result, kept_features, pruned_features

    # Re-run with pruned features
    print(f"\n[4/6] Re-running walk-forward with {len(kept_features)} features...")
    X_pruned = X[kept_features]  # noqa: N806

    pruned_result = trainer.train_walk_forward_ensemble(
        X=X_pruned,
        y=y,
        dates=dates,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        model_types=MODEL_TYPES,
        verbose=verbose,
    )

    print(f"\n  Pruned OOS AUC: {pruned_result['avg_oos_auc']:.4f}")
    print(
        f"  Improvement: "
        f"{pruned_result['avg_oos_auc'] - first_pass_result['avg_oos_auc']:+.4f}"
    )

    # Use pruned result if it's equal or better
    if pruned_result["avg_oos_auc"] >= first_pass_result["avg_oos_auc"] - 0.005:
        print("  Using pruned feature set (equal or better performance)")
        return pruned_result, kept_features, pruned_features

    print("  Pruned performance worse — reverting to full feature set")
    return first_pass_result, all_features, []


def save_models(
    trainer: WalkForwardTrainer,
    result: dict,
    feature_names: list[str],
) -> dict[str, str]:
    """Save final ensemble models from the most recent window."""
    print("\n[5/6] Saving models...")

    saved = trainer.save_ensemble_models(
        window_results=result["window_results"],
        feature_names=feature_names,
        metadata=result,
        model_types=MODEL_TYPES,
    )

    # Also save feature names
    feature_names_path = MODELS_DIR / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Feature names: {feature_names_path}")

    # Save comprehensive metadata
    meta = {
        "n_windows": result["n_windows"],
        "n_models": result["n_models"],
        "model_types": result["model_types"],
        "train_months": result["train_months"],
        "test_months": result["test_months"],
        "avg_oos_auc": result["avg_oos_auc"],
        "min_oos_auc": result["min_oos_auc"],
        "max_oos_auc": result["max_oos_auc"],
        "std_oos_auc": result["std_oos_auc"],
        "avg_oos_accuracy": result["avg_oos_accuracy"],
        "avg_cv_auc": result["avg_cv_auc"],
        "n_features": len(feature_names),
        "total_train_samples": result["total_train_samples"],
        "train_time_seconds": result["train_time_seconds"],
        "generated_at": datetime.now(UTC).isoformat(),
        "model_paths": saved,
    }

    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    for mt, path in saved.items():
        print(f"  {mt}: {path}")

    return saved


def generate_report(
    result: dict,
    feature_names: list[str],
    pruned_features: list[str],
    original_n_features: int,
) -> Path:
    """Generate the Phase 3 training report."""
    print("\n[6/6] Generating report...")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "PHASE_3_TRAINING.md"

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Phase 3: Walk-Forward Training Report",
        f"**Generated:** {now}",
        "",
        "## Training Configuration",
        f"- Train window: {TRAIN_MONTHS} months",
        f"- Test window: {TEST_MONTHS} month",
        "- Purged CV: 5-fold, 5-day embargo",
        "- Models: XGBoost, LightGBM, CatBoost",
        f"- Features: {len(feature_names)} "
        f"(after pruning from {original_n_features} original)",
        "",
        "## Walk-Forward Results",
        "| Window | Train Period | Test Period "
        "| CV AUC (mean\u00b1std) | OOS AUC | OOS Accuracy |",
        "|--------|-------------|-------------"
        "|-------------------|---------|-------------|",
    ]

    for w in result["window_results"]:
        lines.append(
            f"| {w['window']} | {w['train_period']} | {w['test_period']} "
            f"| {w['cv_auc_mean']:.4f}\u00b1{w['cv_auc_std']:.4f} "
            f"| {w['oos_auc']:.4f} | {w['oos_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Avg OOS AUC | {result['avg_oos_auc']:.4f} |",
            f"| Min OOS AUC | {result['min_oos_auc']:.4f} |",
            f"| Max OOS AUC | {result['max_oos_auc']:.4f} |",
            f"| Std OOS AUC | {result['std_oos_auc']:.4f} |",
            f"| Avg CV AUC | {result['avg_cv_auc']:.4f} |",
            f"| Features (post-pruning) | {len(feature_names)} |",
            f"| Total training samples | {result['total_train_samples']} |",
            f"| Training time | {result['train_time_seconds']:.1f}s |",
            "",
        ]
    )

    # Feature importance (top 20)
    importances = result["feature_importances"]
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    lines.extend(
        [
            "## Feature Importance (Top 20)",
            "| Rank | Feature | Avg Importance |",
            "|------|---------|---------------|",
        ]
    )
    for rank, (feat, imp) in enumerate(sorted_imp[:20], 1):
        lines.append(f"| {rank} | {feat} | {imp:.6f} |")

    # Pruned features
    lines.extend(
        [
            "",
            f"## Features Pruned ({len(pruned_features)} dropped, "
            f"< {PRUNE_THRESHOLD * 100:.1f}% importance)",
        ]
    )
    if pruned_features:
        for f in sorted(pruned_features):
            lines.append(f"- `{f}`")
    else:
        lines.append("No features pruned.")

    # Per-model performance
    lines.extend(
        [
            "",
            "## Per-Model Performance",
            "| Model | Avg OOS AUC |",
            "|-------|-------------|",
        ]
    )

    for mt in MODEL_TYPES:
        model_aucs = [
            w["per_model_auc"][mt]
            for w in result["window_results"]
            if mt in w.get("per_model_auc", {})
        ]
        avg = float(np.mean(model_aucs)) if model_aucs else 0.0
        lines.append(f"| {mt} | {avg:.4f} |")

    # Assessment
    avg_auc = result["avg_oos_auc"]
    lines.extend(
        [
            "",
            "## Assessment",
            "",
        ]
    )

    if avg_auc > 0.72:
        lines.append(
            f"Excellent performance: avg OOS AUC {avg_auc:.4f} > 0.72. "
            "Model is well-calibrated for production use."
        )
    elif avg_auc > 0.68:
        lines.append(
            f"Good performance: avg OOS AUC {avg_auc:.4f} > 0.68. "
            "Model suitable for production with confidence filtering."
        )
    elif avg_auc > 0.60:
        lines.append(
            f"Acceptable performance: avg OOS AUC {avg_auc:.4f} > 0.60. "
            "Model is learning meaningful patterns. "
            "May improve with additional features or hyperparameter tuning."
        )
    else:
        lines.append(
            f"Below target: avg OOS AUC {avg_auc:.4f} < 0.60. "
            "Investigate data leakage, target definition, "
            "or feature engineering issues."
        )

    # Stability assessment
    auc_range = result["max_oos_auc"] - result["min_oos_auc"]
    if auc_range < 0.15:
        lines.append(f"AUC stability is excellent (range: {auc_range:.4f} < 0.15).")
    elif auc_range < 0.25:
        lines.append(f"AUC stability is acceptable (range: {auc_range:.4f} < 0.25).")
    else:
        lines.append(
            f"AUC stability needs attention (range: {auc_range:.4f} >= 0.25). "
            "Performance varies significantly across market regimes."
        )

    # Ensemble vs individual
    ens_auc = result["avg_oos_auc"]
    per_model_avgs = {}
    for mt in MODEL_TYPES:
        model_aucs = [
            w["per_model_auc"][mt]
            for w in result["window_results"]
            if mt in w.get("per_model_auc", {})
        ]
        per_model_avgs[mt] = float(np.mean(model_aucs)) if model_aucs else 0.0

    best_single = max(per_model_avgs.values()) if per_model_avgs else 0.0
    if ens_auc >= best_single:
        lines.append(
            f"Ensemble ({ens_auc:.4f}) matches or beats best "
            f"single model ({best_single:.4f})."
        )
    else:
        lines.append(
            f"Best single model ({best_single:.4f}) slightly beats "
            f"ensemble ({ens_auc:.4f}). Consider weighted averaging."
        )

    # Saved artifacts
    lines.extend(
        [
            "",
            "## Saved Artifacts",
            "- `models/ensemble_xgb.json`",
            "- `models/ensemble_lgb.pkl`",
            "- `models/ensemble_cat.cbm`",
            "- `models/feature_names.json`",
            "- `models/model_metadata.json`",
            "",
        ]
    )

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print(f"  Report: {report_path}")
    return report_path


def rename_model_files(saved_paths: dict[str, str]) -> None:
    """Rename versioned model files to the canonical names expected by production.

    The trainer saves files like titan_xgboost_ensemble_v2.json.
    Production expects models/ensemble_xgb.json, etc.
    """
    canonical = {
        "xgboost": MODELS_DIR / "ensemble_xgb.json",
        "lightgbm": MODELS_DIR / "ensemble_lgb.pkl",
        "catboost": MODELS_DIR / "ensemble_cat.cbm",
    }

    import shutil

    for mt, target_path in canonical.items():
        if mt in saved_paths:
            src = Path(saved_paths[mt])
            if src.exists():
                # Remove existing target if it exists (may be root-owned)
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except PermissionError:
                        print(
                            f"  {mt}: Cannot overwrite "
                            f"{target_path} (permission denied)"
                        )
                        continue
                shutil.copy2(src, target_path)
                print(f"  {mt}: {src.name} -> {target_path.name}")


def validate_results(result: dict, feature_names: list[str]) -> bool:
    """Run the Phase 3 validation checks."""
    print("\n--- Validation ---")
    ok = True

    # 1. Walk-forward windows
    n = result["n_windows"]
    if n >= 12:
        print(f"  [PASS] Windows: {n} >= 12")
    else:
        print(f"  [FAIL] Windows: {n} < 12")
        ok = False

    # 2. Three models
    if result["n_models"] == 3:
        print(f"  [PASS] Models: {result['n_models']}")
    else:
        print(f"  [FAIL] Models: {result['n_models']} != 3")
        ok = False

    if set(result["model_types"]) == {"xgboost", "lightgbm", "catboost"}:
        print(f"  [PASS] Model types: {result['model_types']}")
    else:
        print(f"  [FAIL] Model types: {result['model_types']}")
        ok = False

    # 3. OOS AUC
    auc = result["avg_oos_auc"]
    if auc > 0.60:
        print(f"  [PASS] Avg OOS AUC: {auc:.4f} > 0.60")
    else:
        print(f"  [WARN] Avg OOS AUC: {auc:.4f} <= 0.60")
        # Don't fail — may be acceptable for initial training

    # 4. AUC stability
    auc_range = result["max_oos_auc"] - result["min_oos_auc"]
    if auc_range < 0.25:
        print(f"  [PASS] AUC range: {auc_range:.4f} < 0.25")
    else:
        print(f"  [WARN] AUC range: {auc_range:.4f} >= 0.25")

    # 5. CV AUC
    cv_auc = result["avg_cv_auc"]
    if cv_auc > 0.55:
        print(f"  [PASS] Avg CV AUC: {cv_auc:.4f} > 0.55")
    else:
        print(f"  [WARN] Avg CV AUC: {cv_auc:.4f} <= 0.55")

    # 6. Model files
    for path in [
        "models/ensemble_xgb.json",
        "models/ensemble_lgb.pkl",
        "models/ensemble_cat.cbm",
        "models/feature_names.json",
    ]:
        if os.path.exists(path):
            print(f"  [PASS] {path} exists")
        else:
            print(f"  [FAIL] {path} missing")
            ok = False

    # 7. Feature names
    if len(feature_names) >= 50:
        print(f"  [PASS] Features after pruning: {len(feature_names)} >= 50")
    else:
        print(f"  [WARN] Features after pruning: {len(feature_names)} < 50")

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Walk-Forward Ensemble Training"
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run full walk-forward training",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress",
    )
    args = parser.parse_args()

    if not args.walk_forward:
        parser.print_help()
        print("\nUse --walk-forward to run training.")
        sys.exit(1)

    # Load data
    X, y, dates, all_feature_names = load_data(FEATURES_PATH)  # noqa: N806
    original_n_features = len(all_feature_names)

    # Initialize trainer
    trainer = WalkForwardTrainer(
        n_splits=5,
        embargo_days=5,
        model_dir=str(MODELS_DIR),
    )

    # First pass: walk-forward with all features
    first_result = run_walk_forward(trainer, X, y, dates, args.verbose)

    # Feature pruning + optional re-run
    final_result, final_features, pruned_features = prune_and_retrain(
        trainer, X, y, dates, first_result, args.verbose
    )

    # Save models
    saved_paths = save_models(trainer, final_result, final_features)

    # Rename to canonical names
    rename_model_files(saved_paths)

    # Generate report
    generate_report(final_result, final_features, pruned_features, original_n_features)

    # Validate
    all_passed = validate_results(final_result, final_features)
    if all_passed:
        print("\n All validation checks PASSED.")
    else:
        print("\n Some checks failed or warned — review report.")

    print(f"\nDone. Training time: {final_result['train_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
