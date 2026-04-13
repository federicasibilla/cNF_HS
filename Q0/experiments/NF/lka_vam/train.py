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
SAMPLE_CHUNK = 50_000  # generation chunk size for memory stability

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


def build_onehot_adm1_conditions(df: pd.DataFrame, adm1_col: str):
    """
    Returns:
      cond_df_full: one-hot(adm1_col)
      onehot_cols: list of one-hot columns
    """
    adm1_dummies_full = pd.get_dummies(df[adm1_col].astype(str), prefix=adm1_col)
    return adm1_dummies_full, list(adm1_dummies_full.columns)


def sample_adm1_conditions_from_empirical(df: pd.DataFrame, adm1_col: str, onehot_cols, n: int, seed: int):
    """
    Sample ADM1 labels with replacement according to their empirical frequencies,
    then build a one-hot condition dataframe with the SAME columns/order as training.
    """
    rng = np.random.RandomState(seed)

    adm1_probs = df[adm1_col].value_counts(normalize=True).sort_index()
    adm1_values = adm1_probs.index.to_numpy()
    probs = adm1_probs.values.astype(float)

    sampled_adm1 = rng.choice(adm1_values, size=n, replace=True, p=probs)

    sampled_df = pd.DataFrame({adm1_col: sampled_adm1})
    cond_df = pd.get_dummies(sampled_df[adm1_col].astype(str), prefix=adm1_col)

    # force exact training column set/order
    cond_df = cond_df.reindex(columns=onehot_cols, fill_value=0)

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


# ============================================================
# ETH NF MICRON (conditional RealNVP)
# Full dataset training, conditioned on one-hot(adm1name) only
# Generate n_syn rows by sampling ADM1 from empirical frequency
# distribution with replacement
# ============================================================

DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/lka_vam/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q0/experiments/NF/lka_vam/results"
SEEDS = [1, 2, 3, 4, 5]

target_cols = [
    'education_score', 'log_income', 'space_per_person', 'FES', 'FCS',
       'rCSI'
]

adm1_col = "adm1name"

# columns kept only for reference in outputs; note they are not used as conditions
extra_id_candidates = [
    "adm1geometry"
]

psu_col = "psu"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Force GPU usage
    assert_cuda_usable()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    required_cols = list(set(target_cols + [adm1_col]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    need_cols = list(set(target_cols + [adm1_col]))
    df = df.dropna(subset=need_cols).copy()

    # type safety
    if psu_col in df.columns:
        df[psu_col] = df[psu_col].astype(str)
    df[adm1_col] = df[adm1_col].astype(str)

    for c in target_cols:
        df[c] = df[c].astype(float)

    n_syn = df.shape[0]

    print(f"Rows after dropna on needed cols: {df.shape[0]}")
    print(f"Unique {adm1_col}: {df[adm1_col].nunique()}")
    print(f"Synthetic rows to generate: {n_syn}")

    # Training conditions: one-hot(adm1)
    cond_df_full, onehot_cols = build_onehot_adm1_conditions(df, adm1_col)
    C_full = cond_df_full.values.astype(float)

    # Output base for training data reference
    keep_extra = [c for c in extra_id_candidates if c in df.columns]

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
        "condition_sampling": "adm1_empirical_frequency_distribution_with_replacement",
        "conditioning_mode": "onehot_adm1_only",
        "target_cols": target_cols,
        "adm1_col": adm1_col,
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
        print("-> Fitting RealNVP on FULL dataset (scaled targets, conditioned on one-hot adm1)...")
        model.fit(X_full_s, C_full)

        # Save model if possible
        try:
            import pickle
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

        C_syn = cond_df_syn.values.astype(float)

        # Output base contains sampled ADM1 + optional metadata + one-hot columns
        out_base_raw = sampled_adm1_df.copy()
        for c in keep_extra:
            # no exact row-level mapping here; keep as missing placeholders if desired
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
        out_csv = os.path.join(seed_dir, f"synthetic_pool.csv")
        if os.path.exists(out_csv):
            os.remove(out_csv)

        ordered_cols = list(out_base.columns) + target_cols

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
                "condition_sampling": "adm1_empirical_frequency_distribution_with_replacement",
                "conditioning_mode": "onehot_adm1_only",
            },
        )

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()