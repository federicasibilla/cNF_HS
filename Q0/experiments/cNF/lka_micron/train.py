import os
import json
import random
import numpy as np
import pandas as pd

from probaforms.models import RealNVP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# -----------------------
# CONFIG
# -----------------------
SAMPLE_CHUNK = 1_000  # generation chunk size for memory stability

SCALE_METHOD = "iqr"
EPS = 1e-12

# RealNVP hyperparams
N_EPOCHS = 600
HIDDEN = (64, 64)


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
# LINEAR SCALER (MONOTONE)
# -----------------------
class MonotoneLinearScaler:
    """
    Monotone linear per-feature scaling:
      x_scaled = (x - center) / scale
    where scale > 0 (fallback to 1 if degenerate).
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

        if self.method == "zscore":
            center = np.nanmean(X, axis=0)
            scale = np.nanstd(X, axis=0, ddof=0)
        else:  # iqr
            center = np.nanmedian(X, axis=0)
            q75 = np.nanpercentile(X, 75, axis=0)
            q25 = np.nanpercentile(X, 25, axis=0)
            scale = q75 - q25

        scale = np.where(np.isfinite(scale) & (scale > self.eps), scale, 1.0)
        self.center_ = center
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted yet.")
        X = np.asarray(X, dtype=float)
        return (X - self.center_) / self.scale_

    def inverse_transform(self, Xs: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fitted yet.")
        Xs = np.asarray(Xs, dtype=float)
        return Xs * self.scale_ + self.center_

    def to_dict(self):
        return {
            "method": self.method,
            "center": self.center_.tolist() if self.center_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }


def build_onehot_conditions(df: pd.DataFrame, cond_base_cols, sector_col: str):
    """
    Returns:
      cond_df_full: columns = base continuous + one-hot(sector_col)
      onehot_cols: list of one-hot columns
    """
    cond_base_full = df[cond_base_cols].copy()
    sector_dummies_full = pd.get_dummies(df[sector_col].astype(int), prefix=sector_col)
    cond_df_full = pd.concat([cond_base_full, sector_dummies_full], axis=1)
    return cond_df_full, list(sector_dummies_full.columns)


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


# ============================================================
# ETH NF MICRON (conditional RealNVP)
# Full dataset training, conditional RealNVP
# Generate EXACTLY one synthetic row per cleaned real row
# using the original empirical condition table once each
# ============================================================

DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/lka_micron/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q0/experiments/cNF/lka_micron/results"
SEEDS = [1, 2, 3, 4, 5]

target_cols = [
    'vita_rae_mcg',
    'folate_mcg', 'vitb12_mcg', 'fe_mg', 'zn_mg',
  'avg_adult_education', 'log_exp'
]

cond_base_cols = [
    "entropy_2", "rwi_2",
    "r3q", "rfh_avg", "vim_avg","sector"
]
sector_col = "sector"

id_cols_base = [
    "adm1name"
]
extra_id_candidates = [
    "adm1geometry", "adm2name", "adm2geometry", "psu"
]

psu_col = "psu"
adm1_col = "adm1name"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Force GPU usage
    assert_cuda_usable()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    required_cols = list(set(target_cols + cond_base_cols + [sector_col] + id_cols_base))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        strict_required = list(set(target_cols + cond_base_cols + [sector_col]))
        strict_missing = [c for c in strict_required if c not in df.columns]
        if strict_missing:
            raise ValueError(f"Missing required columns (strict): {strict_missing}")
        else:
            print(f"Warning: some id_cols_base not found and will be skipped: {missing}")

    need_cols = list(set(target_cols + cond_base_cols + [sector_col]))
    df = df.dropna(subset=need_cols).copy()

    # type safety
    if psu_col in df.columns:
        df[psu_col] = df[psu_col].astype(str)
    if adm1_col in df.columns:
        df[adm1_col] = df[adm1_col].astype(str)

    for c in target_cols:
        df[c] = df[c].astype(float)

    for c in cond_base_cols:
        df[c] = df[c].astype(float)

    df[sector_col] = df[sector_col].astype(int)

    n_syn = df.shape[0]

    print(f"Rows after dropna on needed cols: {df.shape[0]}")
    print(f"Unique {sector_col}: {df[sector_col].nunique()}")
    print(f"Synthetic rows to generate: {n_syn} (same as cleaned real dataset)")

    # Conditions for model input: continuous + one-hot(sector)
    cond_df_full, onehot_cols = build_onehot_conditions(df, cond_base_cols, sector_col)
    C_full = cond_df_full.values.astype(float)

    # IDs / raw-condition output columns
    id_cols = [c for c in id_cols_base if c in df.columns]
    keep_extra = [c for c in extra_id_candidates if c in df.columns]

    # Keep raw sector + raw base conditions for readability
    out_cond_cols_raw = cond_base_cols + [sector_col]
    out_base_raw = df[id_cols + keep_extra + out_cond_cols_raw].reset_index(drop=True)
    out_base = pd.concat(
        [out_base_raw, cond_df_full[onehot_cols].reset_index(drop=True)],
        axis=1
    )

    # Targets
    X_full = df[target_cols].values.astype(float)

    # Fit scaler on FULL DATA
    x_scaler = MonotoneLinearScaler(method=SCALE_METHOD, eps=EPS).fit(X_full)
    X_full_s = x_scaler.transform(X_full)

    # Save experiment-level metadata
    meta = {
        "data_path": DATA_PATH,
        "n_rows_train": int(df.shape[0]),
        "n_syn": int(n_syn),
        "generation_mode": "same_n_as_cleaned_real_data",
        "condition_sampling": "original_empirical_condition_table_once_each",
        "target_cols": target_cols,
        "cond_base_cols": cond_base_cols,
        "sector_col": sector_col,
        "cond_onehot_cols": onehot_cols,
        "sample_chunk": int(SAMPLE_CHUNK),
        "scale_method": SCALE_METHOD,
        "realnvp_params": {"n_epochs": N_EPOCHS, "hidden": list(HIDDEN)},
        "scaler": x_scaler.to_dict(),
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
        print("-> Fitting RealNVP on FULL dataset (scaled targets, conditional)...")
        model.fit(X_full_s, C_full)

        # Save model if possible
        try:
            import pickle
            with open(os.path.join(seed_dir, "realnvp_model.pkl"), "wb") as f:
                pickle.dump(model, f)
            print("-> Saved model pickle.")
        except Exception as e:
            print(f"-> Could not pickle model (non-fatal): {repr(e)}")

        # Use original condition rows exactly once each
        C_syn = C_full
        out_base_syn = out_base.reset_index(drop=True)

        # Generate in chunks
        out_csv = os.path.join(seed_dir, f"synthetic_pool.csv")
        if os.path.exists(out_csv):
            os.remove(out_csv)

        ordered_cols = list(out_base_syn.columns) + target_cols

        def chunk_iter():
            remaining = n_syn
            start = 0
            while remaining > 0:
                n_now = min(SAMPLE_CHUNK, remaining)
                C_chunk = C_syn[start:start + n_now]
                X_gen_s = model.sample(C_chunk)
                X_gen = x_scaler.inverse_transform(X_gen_s)

                syn_targets = pd.DataFrame(X_gen, columns=target_cols)
                syn_chunk = pd.concat(
                    [out_base_syn.iloc[start:start + n_now].reset_index(drop=True), syn_targets],
                    axis=1
                )
                yield syn_chunk

                start += n_now
                remaining -= n_now

        print(f"-> Generating {n_syn} rows; writing to: {out_csv}")
        chunk_writer(out_csv, chunk_iter(), header_cols=ordered_cols, add_seed=seed)
        print("-> Done.")

        # Optional: also save scaled synthetic pool
        out_csv_scaled = os.path.join(seed_dir, f"synthetic_pool_scaled.csv")
        if os.path.exists(out_csv_scaled):
            os.remove(out_csv_scaled)

        def chunk_iter_scaled():
            remaining = n_syn
            start = 0
            while remaining > 0:
                n_now = min(SAMPLE_CHUNK, remaining)
                C_chunk = C_syn[start:start + n_now]
                X_gen_s = model.sample(C_chunk)

                syn_targets_s = pd.DataFrame(X_gen_s, columns=target_cols)
                syn_chunk_s = pd.concat(
                    [out_base_syn.iloc[start:start + n_now].reset_index(drop=True), syn_targets_s],
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

        complete_scaled_path = os.path.join(seed_dir, f"full_lka_scaled_seed{seed}.csv")
        complete_eth_scaled.to_csv(complete_scaled_path, index=False)
        print(f"-> Saved full_lka_scaled to:\n   {complete_scaled_path}")

        # Save per-seed config
        save_json(
            os.path.join(seed_dir, "seed_config.json"),
            {
                "seed": seed,
                "n_syn": int(n_syn),
                "sample_chunk": int(SAMPLE_CHUNK),
                "generation_mode": "same_n_as_cleaned_real_data",
                "condition_sampling": "original_empirical_condition_table_once_each",
            },
        )

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()