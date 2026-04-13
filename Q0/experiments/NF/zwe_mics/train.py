import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set before importing torch/probaforms

import json
import random
import pickle
import numpy as np
import pandas as pd

from probaforms.models import RealNVP


# -----------------------
# CONFIG
# -----------------------
SAMPLE_CHUNK = 50_000  # generation chunk size for memory stability

SCALE_METHOD = "iqr"   # "iqr" or "zscore"
EPS = 1e-12

# RealNVP hyperparams
N_EPOCHS = 400
HIDDEN = (32, 32)

# light clipping before scaling/training
CLIP_BEFORE_TRAINING = True
CLIP_LOW_Q = 0.005
CLIP_HIGH_Q = 0.995

# ============================================================
# ETH NF MICRON (conditional RealNVP)
# Full dataset training, conditioned on one-hot(adm1name) only
# Generate n_syn rows by sampling ADM1 from empirical frequency
# distribution with replacement
# ============================================================

DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/zwe_mics/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q0/experiments/NF/zwe_mics/results"
SEEDS = [1, 2, 3, 4, 5]

target_cols = [
    "space_per_person", "avg_adult_education", "wscore"
]

adm1_col = "adm1name"

# columns kept only for reference in outputs; not used as conditions
extra_id_candidates = [
    "adm1geometry"
]

psu_col = "psu"


# -----------------------
# GPU: FORCE USAGE (FAIL FAST IF NOT USABLE)
# -----------------------
def assert_cuda_usable():
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False. GPU not available.")
        _ = torch.randn(1, device="cuda:0")
        return True
    except Exception as e:
        raise RuntimeError(
            "CUDA is not usable by this PyTorch build on this machine. "
            "You requested to force GPU usage, so exiting. "
            f"Original error: {repr(e)}"
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -----------------------
# LINEAR SCALER (MONOTONE, STRICT)
# -----------------------
class MonotoneLinearScaler:
    """
    Monotone linear per-feature scaling:
      x_scaled = (x - center) / scale
    where scale > 0 (fallback to 1 if degenerate).

    Strict version:
    - fails on non-finite input
    - fails if transform/inverse_transform produce non-finite output
    """
    def __init__(self, method: str = "iqr", eps: float = 1e-12):
        if method not in ("zscore", "iqr"):
            raise ValueError("method must be one of: 'zscore', 'iqr'")
        self.method = method
        self.eps = eps
        self.center_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")

        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            i, j = bad[0]
            raise ValueError(
                f"Scaler received non-finite values. "
                f"Example at row={i}, col={j}, value={X[i, j]}"
            )

        if self.method == "zscore":
            center = np.mean(X, axis=0)
            scale = np.std(X, axis=0, ddof=0)
        else:  # iqr
            center = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            scale = q75 - q25

        scale = np.where(np.isfinite(scale) & (scale > self.eps), scale, 1.0)

        if not np.isfinite(center).all():
            raise ValueError("Scaler center contains non-finite values.")
        if not np.isfinite(scale).all():
            raise ValueError("Scaler scale contains non-finite values.")

        self.center_ = center
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted yet.")

        X = np.asarray(X, dtype=float)
        Xs = (X - self.center_) / self.scale_

        if not np.isfinite(Xs).all():
            bad = np.argwhere(~np.isfinite(Xs))
            i, j = bad[0]
            raise ValueError(
                f"Scaled data contains non-finite values. "
                f"Example at row={i}, col={j}, value={Xs[i, j]}"
            )

        return Xs

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted yet.")

        Xs = np.asarray(Xs, dtype=float)
        X = Xs * self.scale_ + self.center_

        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            i, j = bad[0]
            raise ValueError(
                f"Inverse-transformed data contains non-finite values. "
                f"Example at row={i}, col={j}, value={X[i, j]}"
            )

        return X

    def to_dict(self):
        return {
            "method": self.method,
            "center": self.center_.tolist() if self.center_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }


# -----------------------
# HELPERS
# -----------------------
def build_onehot_adm1_conditions(df: pd.DataFrame, adm1_col: str):
    """
    Returns:
      cond_df_full: one-hot(adm1_col)
      onehot_cols: list of one-hot columns
    """
    adm1_dummies_full = pd.get_dummies(df[adm1_col].astype(str), prefix=adm1_col)
    adm1_dummies_full = adm1_dummies_full.astype(float)
    return adm1_dummies_full, list(adm1_dummies_full.columns)


def sample_adm1_conditions_from_empirical(df: pd.DataFrame, adm1_col: str, onehot_cols, n: int, seed: int):
    """
    Sample ADM1 labels with replacement according to their empirical frequencies,
    then build a one-hot condition dataframe with the same columns/order as training.
    """
    rng = np.random.RandomState(seed)

    adm1_probs = df[adm1_col].value_counts(normalize=True).sort_index()
    adm1_values = adm1_probs.index.to_numpy()
    probs = adm1_probs.values.astype(float)

    sampled_adm1 = rng.choice(adm1_values, size=n, replace=True, p=probs)

    sampled_df = pd.DataFrame({adm1_col: sampled_adm1})
    cond_df = pd.get_dummies(sampled_df[adm1_col].astype(str), prefix=adm1_col)
    cond_df = cond_df.reindex(columns=onehot_cols, fill_value=0)
    cond_df = cond_df.astype(float)

    return sampled_df.reset_index(drop=True), cond_df.reset_index(drop=True), adm1_probs


def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def chunk_writer(csv_path: str, df_iterable, header_cols, add_seed: int = None):
    wrote_header = False
    for chunk_idx, df_chunk in enumerate(df_iterable, start=1):
        out = df_chunk.copy()
        if add_seed is not None:
            out["seed"] = add_seed
        out = out[header_cols + (["seed"] if add_seed is not None else [])]
        out.to_csv(csv_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        print(f"   wrote chunk {chunk_idx} ({out.shape[0]} rows)")


def report_numeric_columns(df: pd.DataFrame, cols, title: str):
    print(f"\n{title}")
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        arr = s.to_numpy(dtype=float, na_value=np.nan)
        finite_mask = np.isfinite(arr)
        finite_vals = arr[finite_mask]

        finite_min = finite_vals.min() if finite_vals.size > 0 else "NA"
        finite_max = finite_vals.max() if finite_vals.size > 0 else "NA"

        print(
            f"{c}: "
            f"nan={int(np.isnan(arr).sum())}, "
            f"inf={int(np.isinf(arr).sum())}, "
            f"finite_min={finite_min}, "
            f"finite_max={finite_max}"
        )


def assert_all_finite(name: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))
        i, j = bad[0]
        raise ValueError(
            f"{name} contains non-finite values. "
            f"Example at row={i}, col={j}, value={arr[i, j]}"
        )


def winsorize_training_targets(
    X: np.ndarray,
    low_q: float = 0.005,
    high_q: float = 0.995
):
    """
    Clip each target column to its empirical [low_q, high_q] interval.
    Returns:
      X_clip, clip_info
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be 2D")

    lower = np.quantile(X, low_q, axis=0)
    upper = np.quantile(X, high_q, axis=0)

    X_clip = np.clip(X, lower, upper)

    clip_info = {
        "low_q": float(low_q),
        "high_q": float(high_q),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "n_clipped_total": int((X_clip != X).sum()),
        "n_rows_any_clipped": int(np.any(X_clip != X, axis=1).sum()),
    }
    return X_clip, clip_info


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Force GPU usage
    assert_cuda_usable()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    required_cols = list(dict.fromkeys(target_cols + [adm1_col]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keep_cols = list(dict.fromkeys(
        target_cols + [adm1_col] + extra_id_candidates + ([psu_col] if psu_col in df.columns else [])
    ))
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Raw diagnostics
    report_numeric_columns(df, target_cols, title="Raw diagnostics before cleaning:")

    # Safe numeric conversion for targets
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace inf/-inf explicitly, then drop invalid rows
    df[target_cols] = df[target_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=target_cols + [adm1_col]).copy()

    # Type safety
    if psu_col in df.columns:
        df[psu_col] = df[psu_col].astype(str)
    df[adm1_col] = df[adm1_col].astype(str)

    # Diagnostics after cleaning
    report_numeric_columns(df, target_cols, title="Diagnostics after cleaning:")

    n_syn = df.shape[0]

    print(f"\nRows after cleaning on needed cols: {df.shape[0]}")
    print(f"Unique {adm1_col}: {df[adm1_col].nunique()}")
    print(f"Synthetic rows to generate: {n_syn}")

    # Training conditions: one-hot(adm1)
    cond_df_full, onehot_cols = build_onehot_adm1_conditions(df, adm1_col)
    C_full = cond_df_full.to_numpy(dtype=float)

    # Optional reference columns
    keep_extra = [c for c in extra_id_candidates if c in df.columns]

    # Targets
    X_full_raw = df[target_cols].to_numpy(dtype=float)

    # Strict finite checks
    assert_all_finite("X_full_raw", X_full_raw)
    assert_all_finite("C_full", C_full)

    # Optional clipping before scaling/training
    if CLIP_BEFORE_TRAINING:
        X_full, clip_info = winsorize_training_targets(
            X_full_raw,
            low_q=CLIP_LOW_Q,
            high_q=CLIP_HIGH_Q
        )
        print("\nApplied clipping before scaling/training:")
        print(clip_info)
    else:
        X_full = X_full_raw.copy()
        clip_info = {
            "enabled": False,
            "low_q": None,
            "high_q": None,
            "lower": None,
            "upper": None,
            "n_clipped_total": 0,
            "n_rows_any_clipped": 0,
        }

    # Strict finite checks
    assert_all_finite("X_full", X_full)

    # Fit scaler on FULL DATA
    x_scaler = MonotoneLinearScaler(method=SCALE_METHOD, eps=EPS).fit(X_full)
    X_full_s = x_scaler.transform(X_full)

    # Check scaled data too
    assert_all_finite("X_full_s", X_full_s)

    # Save cleaned training data actually used
    cleaned_training = pd.concat(
        [
            df[[adm1_col] + keep_extra].reset_index(drop=True),
            cond_df_full.reset_index(drop=True),
            pd.DataFrame(X_full, columns=target_cols).reset_index(drop=True),
        ],
        axis=1
    )
    cleaned_train_path = os.path.join(OUTPUT_DIR, "cleaned_training_data.csv")
    cleaned_training.to_csv(cleaned_train_path, index=False)
    print(f"Saved cleaned training data to: {cleaned_train_path}")

    # Save original unclipped cleaned training data for comparison
    raw_training = pd.concat(
        [
            df[[adm1_col] + keep_extra].reset_index(drop=True),
            cond_df_full.reset_index(drop=True),
            pd.DataFrame(X_full_raw, columns=target_cols).reset_index(drop=True),
        ],
        axis=1
    )
    raw_train_path = os.path.join(OUTPUT_DIR, "cleaned_training_data_unclipped.csv")
    raw_training.to_csv(raw_train_path, index=False)
    print(f"Saved unclipped cleaned training data to: {raw_train_path}")

    # Save scaled training debug table
    scaled_debug = pd.DataFrame(X_full_s, columns=[f"{c}_scaled" for c in target_cols])
    scaled_debug = pd.concat(
        [df[[adm1_col]].reset_index(drop=True), cond_df_full.reset_index(drop=True), scaled_debug],
        axis=1
    )
    scaled_debug_path = os.path.join(OUTPUT_DIR, "scaled_training_debug.csv")
    scaled_debug.to_csv(scaled_debug_path, index=False)
    print(f"Saved scaled training debug data to: {scaled_debug_path}")

    # Save experiment-level metadata
    meta = {
        "data_path": DATA_PATH,
        "n_rows_train": int(df.shape[0]),
        "n_syn": int(n_syn),
        "generation_mode": "same_n_as_cleaned_real_data",
        "condition_sampling": "adm1_empirical_frequency_distribution_with_replacement",
        "conditioning_mode": "onehot_adm1_only",
        "target_cols": target_cols,
        "adm1_col": adm1_col,
        "cond_onehot_cols": onehot_cols,
        "sample_chunk": int(SAMPLE_CHUNK),
        "scale_method": SCALE_METHOD,
        "realnvp_params": {"n_epochs": N_EPOCHS, "hidden": list(HIDDEN)},
        "clip_before_training": CLIP_BEFORE_TRAINING,
        "clip_low_q": CLIP_LOW_Q if CLIP_BEFORE_TRAINING else None,
        "clip_high_q": CLIP_HIGH_Q if CLIP_BEFORE_TRAINING else None,
        "clip_info": clip_info,
        "target_scaler": x_scaler.to_dict(),
    }
    save_json(os.path.join(OUTPUT_DIR, "run_metadata.json"), meta)
    print(f"Saved run metadata to: {os.path.join(OUTPUT_DIR, 'run_metadata.json')}")

    for seed in SEEDS:
        print("\n==============================")
        print(f"Seed: {seed}")
        print("==============================")

        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        seed_everything(seed)

        # Train conditional RealNVP on full dataset
        model = RealNVP(n_epochs=N_EPOCHS, hidden=HIDDEN)
        print("-> Fitting RealNVP on FULL dataset (scaled targets, conditioned on one-hot adm1)...")
        model.fit(X_full_s, C_full)

        # Save model if possible
        try:
            with open(os.path.join(seed_dir, "realnvp_model.pkl"), "wb") as f:
                pickle.dump(model, f)
            print("-> Saved model pickle.")
        except Exception as e:
            print(f"-> Could not pickle model (non-fatal): {repr(e)}")

        # Sample ADM1 conditions from empirical frequency distribution
        sampled_adm1_df, cond_df_syn, adm1_probs = sample_adm1_conditions_from_empirical(
            df=df,
            adm1_col=adm1_col,
            onehot_cols=onehot_cols,
            n=n_syn,
            seed=seed,
        )

        C_syn = cond_df_syn.to_numpy(dtype=float)
        assert_all_finite("C_syn", C_syn)

        # Output base contains sampled ADM1 + optional metadata + one-hot columns
        out_base_raw = sampled_adm1_df.copy()
        for c in keep_extra:
            out_base_raw[c] = np.nan

        out_base = pd.concat(
            [out_base_raw.reset_index(drop=True), cond_df_syn.reset_index(drop=True)],
            axis=1
        )

        # Save sampled ADM1 frequency table
        adm1_probs_df = adm1_probs.rename("probability").reset_index().rename(columns={"index": adm1_col})
        adm1_probs_df.to_csv(os.path.join(seed_dir, "adm1_empirical_probs.csv"), index=False)

        sampled_counts_df = sampled_adm1_df[adm1_col].value_counts().rename("n_sampled").reset_index()
        sampled_counts_df.columns = [adm1_col, "n_sampled"]
        sampled_counts_df.to_csv(os.path.join(seed_dir, "adm1_sampled_counts.csv"), index=False)

        # Generate raw-scale synthetic pool
        out_csv = os.path.join(seed_dir, "synthetic_pool.csv")
        if os.path.exists(out_csv):
            os.remove(out_csv)

        ordered_cols = list(out_base.columns) + target_cols

        def chunk_iter():
            remaining = n_syn
            start = 0
            while remaining > 0:
                n_now = min(SAMPLE_CHUNK, remaining)

                C_chunk = C_syn[start:start + n_now]
                assert_all_finite("C_chunk", C_chunk)

                X_gen_s = model.sample(C_chunk)
                X_gen_s = np.asarray(X_gen_s, dtype=float)
                assert_all_finite("X_gen_s", X_gen_s)

                X_gen = x_scaler.inverse_transform(X_gen_s)
                assert_all_finite("X_gen", X_gen)

                syn_targets = pd.DataFrame(X_gen, columns=target_cols)
                syn_chunk = pd.concat(
                    [out_base.iloc[start:start + n_now].reset_index(drop=True), syn_targets],
                    axis=1
                )
                yield syn_chunk

                start += n_now
                remaining -= n_now

        print(f"-> Generating {n_syn} rows; writing to: {out_csv}")
        chunk_writer(out_csv, chunk_iter(), header_cols=ordered_cols, add_seed=seed)
        print("-> Done.")

        # Generate scaled synthetic pool
        out_csv_scaled = os.path.join(seed_dir, "synthetic_pool_scaled.csv")
        if os.path.exists(out_csv_scaled):
            os.remove(out_csv_scaled)

        def chunk_iter_scaled():
            remaining = n_syn
            start = 0
            while remaining > 0:
                n_now = min(SAMPLE_CHUNK, remaining)

                C_chunk = C_syn[start:start + n_now]
                assert_all_finite("C_chunk", C_chunk)

                X_gen_s = model.sample(C_chunk)
                X_gen_s = np.asarray(X_gen_s, dtype=float)
                assert_all_finite("X_gen_s", X_gen_s)

                syn_targets_s = pd.DataFrame(X_gen_s, columns=target_cols)
                syn_chunk_s = pd.concat(
                    [out_base.iloc[start:start + n_now].reset_index(drop=True), syn_targets_s],
                    axis=1
                )
                yield syn_chunk_s

                start += n_now
                remaining -= n_now

        print(f"-> Writing scaled synthetic pool to: {out_csv_scaled}")
        chunk_writer(out_csv_scaled, chunk_iter_scaled(), header_cols=ordered_cols, add_seed=seed)
        print("-> Done.")

        # Save full real data with targets scaled by this run's scaler
        complete_eth_scaled = df.copy()
        complete_eth_scaled[target_cols] = X_full_s
        complete_eth_scaled["seed"] = seed
        complete_eth_scaled["scaler_method"] = SCALE_METHOD

        complete_scaled_path = os.path.join(seed_dir, f"full_zwe_scaled_seed{seed}.csv")
        complete_eth_scaled.to_csv(complete_scaled_path, index=False)
        print(f"-> Saved full_zwe_scaled to:\n   {complete_scaled_path}")

        # Save per-seed config
        save_json(
            os.path.join(seed_dir, "seed_config.json"),
            {
                "seed": seed,
                "n_syn": int(n_syn),
                "sample_chunk": int(SAMPLE_CHUNK),
                "generation_mode": "same_n_as_cleaned_real_data",
                "condition_sampling": "adm1_empirical_frequency_distribution_with_replacement",
                "conditioning_mode": "onehot_adm1_only",
                "realnvp_params": {"n_epochs": N_EPOCHS, "hidden": list(HIDDEN)},
                "clip_before_training": CLIP_BEFORE_TRAINING,
                "clip_low_q": CLIP_LOW_Q if CLIP_BEFORE_TRAINING else None,
                "clip_high_q": CLIP_HIGH_Q if CLIP_BEFORE_TRAINING else None,
            },
        )

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()