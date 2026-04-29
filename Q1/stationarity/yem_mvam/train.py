import os
import json
import random
import numpy as np
import pandas as pd

from probaforms.models import RealNVP  


# ============================================================
# SCRIPT 1/2 (yem mvam): conditioning on continuous ADM1 features +
#                        household sampling by exact
#                        households-per-ADM1 levels:
#                        - always >=1 household per ADM1
#                        - additional households sampled proportionally
#                          to remaining household counts per ADM1
# ============================================================

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/yem_mvam/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q1/stationarity/yem_mvam/results"


# === stationarity test config ===
HELD_OUT_ADM1_RATIO = 0.5     # fraction of ADM1s held out from training
STATIONARITY_SEED   = 7       # seed for the random ADM1 partition (independent of training seeds)
SAVE_SPLIT_FILE     = True    # write stationarity_split.csv with adm1 -> in_sample/held_out
HH_PER_ADM1_LEVELS = [
    10,
    20,
    40,
    80
]

SEEDS = [1, 2, 3, 4, 5]

target_cols = [
    "log_exp_pp", "rCSI", "FCS"
]

# continuous ADM1-level conditioning features (already present per row)
cond_cols_adm1 = [
    "entropy_2", "wscore_1",
]

adm1_name_col = "adm1name"

# no PSU; sample directly at household level using this id
hh_id_col = "id"

extra_id_candidates = ["adm1geometry", 'adm2name', 'adm2geometry']

SCALE_METHOD = "iqr"
EPS = 1e-12


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
        else:  # "iqr"
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


def modal_value(series: pd.Series):
    m = series.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan


# -----------------------
# NEW HOUSEHOLD SAMPLING HELPERS
# -----------------------
def sample_one_hh_per_adm1(hh_info_adm1: pd.DataFrame, rng: np.random.RandomState):
    """
    Select exactly 1 household within one ADM1, uniformly among households in that ADM1.
    """
    if hh_info_adm1.empty:
        raise ValueError("hh_info_adm1 is empty")

    pool = hh_info_adm1[hh_id_col].values
    return rng.choice(pool, size=1, replace=False)[0]


def sample_additional_hhs_weighted_by_remaining_adm1(hh_info_df: pd.DataFrame,
                                                     already_selected,
                                                     n_extra: int,
                                                     rng: np.random.RandomState):
    """
    Sequentially sample extra households without replacement.
    At each step, choose among remaining ADM1s with probability proportional
    to the number of remaining households in that ADM1, then sample one
    household uniformly from that ADM1.
    """
    if n_extra <= 0:
        return np.array([], dtype=object)

    selected_set = set(already_selected)
    remaining_df = hh_info_df.loc[~hh_info_df[hh_id_col].isin(selected_set)].copy()

    sampled = []

    for _ in range(n_extra):
        if remaining_df.empty:
            break

        adm1_counts = (
            remaining_df
            .groupby(["hh_modal_adm1"], dropna=False)[hh_id_col]
            .count()
            .reset_index(name="n_remaining")
        )

        weights = adm1_counts["n_remaining"].values.astype(float)
        probs = weights / weights.sum()

        chosen_idx = rng.choice(np.arange(len(adm1_counts)), p=probs)
        chosen_adm1 = adm1_counts.iloc[chosen_idx]["hh_modal_adm1"]

        pool = remaining_df.loc[
            remaining_df["hh_modal_adm1"] == chosen_adm1,
            hh_id_col
        ].values

        if len(pool) == 0:
            pool = remaining_df[hh_id_col].values

        chosen_hh = rng.choice(pool, size=1, replace=False)[0]
        sampled.append(chosen_hh)

        remaining_df = remaining_df.loc[remaining_df[hh_id_col] != chosen_hh].copy()

    return np.array(sampled, dtype=object)


# -----------------------
# LOAD & BASIC CLEANING
# -----------------------
df = pd.read_csv(DATA_PATH)

required_cols = [adm1_name_col, hh_id_col] + target_cols + cond_cols_adm1
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

need_cols = list(set(target_cols + cond_cols_adm1 + [adm1_name_col, hh_id_col]))
df = df.dropna(subset=need_cols).copy()

# NOTE: your original script casts ALL targets to float.
for col in target_cols:
    df[col] = df[col].astype(float)

for col in cond_cols_adm1:
    df[col] = df[col].astype(float)

# household id + ADM1 safe types
df[hh_id_col] = df[hh_id_col].astype(str)
df[adm1_name_col] = df[adm1_name_col].astype(str)

print(f"Total rows after cleaning: {df.shape[0]}")
print(f"Unique households: {df[hh_id_col].nunique()}")
print(f"Unique ADM1s: {df[adm1_name_col].nunique()}")

# -----------------------
# HOUSEHOLD -> modal(adm1) (in case of duplicates per id)
# -----------------------
hh_info_df = (
    df.groupby(hh_id_col)
      .agg(
          hh_modal_adm1=(adm1_name_col, modal_value),
      )
      .reset_index()
)

hh_info_df = hh_info_df.dropna(subset=["hh_modal_adm1"]).copy()

# === stationarity test: split ADM1s into in-sample and held-out ===
_stationarity_rng = np.random.RandomState(STATIONARITY_SEED)
_all_adm1s = np.sort(hh_info_df["hh_modal_adm1"].unique())
_n_held = max(1, int(round(len(_all_adm1s) * HELD_OUT_ADM1_RATIO)))
_held_out_adm1s = set(_stationarity_rng.choice(_all_adm1s, size=_n_held, replace=False).tolist())
_in_sample_adm1s = set(_all_adm1s) - _held_out_adm1s
print(f"  Stationarity split: {len(_in_sample_adm1s)} in-sample / {len(_held_out_adm1s)} held-out ADM1s "
      f"(seed={STATIONARITY_SEED}, ratio={HELD_OUT_ADM1_RATIO})")

# Filter household pool to in-sample ADM1s only
hh_info_df = hh_info_df.loc[hh_info_df["hh_modal_adm1"].isin(_in_sample_adm1s)].copy()

if SAVE_SPLIT_FILE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _split_df = pd.DataFrame({
        "adm1_name": list(_all_adm1s),
        "split": ["held_out" if a in _held_out_adm1s else "in_sample" for a in _all_adm1s],
    })
    _split_df.to_csv(os.path.join(OUTPUT_DIR, "stationarity_split.csv"), index=False)
    print(f"  Saved partition to {os.path.join(OUTPUT_DIR, 'stationarity_split.csv')}")
hh_info_df["hh_modal_adm1"] = hh_info_df["hh_modal_adm1"].astype(str)

unique_hhs = hh_info_df[hh_id_col].values

print("\nHousehold ADM1 counts:")
print(hh_info_df["hh_modal_adm1"].value_counts().sort_index())

# -----------------------
# CONDITIONS: continuous ADM1 features only
# -----------------------
cond_df_full = df[cond_cols_adm1].copy()
C_full = cond_df_full.values.astype(float)

# -----------------------
# PREPARE IDs (for output)
# -----------------------
id_cols = [hh_id_col, adm1_name_col]
keep_extra = [col for col in extra_id_candidates if col in df.columns]
id_df_full = df[id_cols + keep_extra].reset_index(drop=True)

# -----------------------
# MAIN LOOP
# -----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for hh_per_adm1 in HH_PER_ADM1_LEVELS:
    print(f"\n==============================")
    print(f"Training size level: {hh_per_adm1} households per ADM1")
    print(f"==============================")

    level_dir = os.path.join(OUTPUT_DIR, f"train_{hh_per_adm1}_scaled")
    os.makedirs(level_dir, exist_ok=True)

    for seed in SEEDS:
        print(f"\n--- Seed {seed} for HH-per-ADM1 level {hh_per_adm1} ---")

        seed_dir = os.path.join(level_dir, f"seed_{seed}_scaled")
        os.makedirs(seed_dir, exist_ok=True)

        # -----------------------
        # Household sampling
        # -----------------------
        rng = np.random.RandomState(seed)

        n_hh_total = len(unique_hhs)
        unique_adm1s = np.sort(hh_info_df["hh_modal_adm1"].unique())
        n_adm1 = len(unique_adm1s)

        n_hh_train_target = int(hh_per_adm1 * n_adm1)
        n_hh_train_target = max(n_hh_train_target, n_adm1)
        n_hh_train_target = min(n_hh_train_target, n_hh_total)

        # First guaranteed household per ADM1
        mandatory_hhs = []
        for adm1 in unique_adm1s:
            adm1_hh_df = hh_info_df.loc[hh_info_df["hh_modal_adm1"] == adm1].copy()
            pick = sample_one_hh_per_adm1(adm1_hh_df, rng)
            mandatory_hhs.append(pick)

        mandatory_hhs = np.array(mandatory_hhs, dtype=object)

        remaining_needed = int(n_hh_train_target - len(mandatory_hhs))

        if remaining_needed > 0:
            sampled_remaining = sample_additional_hhs_weighted_by_remaining_adm1(
                hh_info_df=hh_info_df,
                already_selected=mandatory_hhs,
                n_extra=remaining_needed,
                rng=rng
            )
            train_hhs = np.concatenate([mandatory_hhs, sampled_remaining])
        else:
            train_hhs = mandatory_hhs.copy()

        train_hhs = np.unique(train_hhs)
        n_hh_train = len(train_hhs)

        df_train = df[df[hh_id_col].isin(train_hhs)].copy()
        train_adm1_covered = df_train[adm1_name_col].nunique()

        print(
            f"  -> Training on {df_train.shape[0]} rows from {n_hh_train}/{n_hh_total} households "
            f"(target {n_hh_train_target} households; level={hh_per_adm1} households per ADM1), "
            f"guaranteeing >=1 household per {adm1_name_col} "
            f"(covered ADM1s: {train_adm1_covered}/{n_adm1})."
        )

        # -----------------------
        # ARRAYS
        # -----------------------
        X_train = df_train[target_cols].values.astype(float)
        C_train = cond_df_full.loc[df_train.index].values.astype(float)

        # -----------------------
        # SCALER FIT ON TRAIN
        # -----------------------
        x_scaler = MonotoneLinearScaler(method=SCALE_METHOD, eps=EPS).fit(X_train)
        X_train_s = x_scaler.transform(X_train)

        # Save scaler params + households used
        scaler_path = os.path.join(
            seed_dir,
            f"x_scaler_{SCALE_METHOD}_train{hh_per_adm1}_seed{seed}.json"
        )
        with open(scaler_path, "w") as f:
            json.dump(
                {
                    "target_cols": target_cols,
                    "train_frac_hh": hh_per_adm1,
                    "train_percent_hh": hh_per_adm1,
                    "hh_per_adm1": hh_per_adm1,
                    "seed": seed,
                    "hh_id_col": hh_id_col,
                    "adm1_col": adm1_name_col,
                    "sampling_rule": "guarantee_1_per_adm1_then_sample_remaining_by_adm1_remaining_counts",
                    "hh_adm1_rule": "modal",
                    "n_hh_total": int(n_hh_total),
                    "n_hh_train_target": int(n_hh_train_target),
                    "n_hh_train_actual": int(n_hh_train),
                    "n_adm1_total": int(n_adm1),
                    "n_adm1_covered": int(train_adm1_covered),
                    "train_households": train_hhs.tolist(),
                    **x_scaler.to_dict(),
                },
                f,
                indent=2,
            )
        print(f"  -> Saved scaler params to:\n     {scaler_path}")

        # -----------------------
        # SAVE TRAINING DF (RAW + SCALED TARGETS)
        # -----------------------
        subsample_raw_path = os.path.join(
            seed_dir,
            f"train_subset_{hh_per_adm1}_seed{seed}.csv"
        )
        df_train.to_csv(subsample_raw_path, index=False)
        print(f"  -> Saved training data (raw) to:\n     {subsample_raw_path}")

        df_train_scaled = df_train.copy()
        df_train_scaled[target_cols] = X_train_s
        subsample_scaled_path = os.path.join(
            seed_dir,
            f"train_subset_{hh_per_adm1}_seed{seed}_scaled.csv"
        )
        df_train_scaled.to_csv(subsample_scaled_path, index=False)
        print(f"  -> Saved training data (scaled targets) to:\n     {subsample_scaled_path}")

        # -----------------------
        # SET RANDOM SEEDS
        # -----------------------
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        # -----------------------
        # TRAIN MODEL ON SCALED TARGETS
        # -----------------------
        model = RealNVP(
            n_epochs=600,
            hidden=(64, 64),
        )

        print("  -> Fitting RealNVP model (on scaled targets)...")
        model.fit(X_train_s, C_train)

        # -----------------------
        # GENERATE FOR FULL CONDITIONS DATASET
        # -----------------------
        print("  -> Generating samples for FULL conditions dataset...")

        # bounds from the training set, variable by variable
        q_low = 0.00
        q_high = 1.
        max_rounds = 50

        lower_bounds = df_train[target_cols].quantile(q_low)
        upper_bounds = df_train[target_cols].quantile(q_high)

        # optional hard floors for variables that should not be negative
        for c in ["education_score", "space_per_person", "FES", "FCS", "rCSI"]:
            if c in lower_bounds.index:
                lower_bounds[c] = max(lower_bounds[c], 0.0)

        X_gen_s = model.sample(C_full)
        X_gen = x_scaler.inverse_transform(X_gen_s)

        round_idx = 0
        while True:
            round_idx += 1

            syn_check = pd.DataFrame(X_gen, columns=target_cols)

            valid_mask = np.ones(len(syn_check), dtype=bool)
            for c in target_cols:
                valid_mask &= syn_check[c].between(lower_bounds[c], upper_bounds[c])

            n_bad = (~valid_mask).sum()
            print(f"     Round {round_idx}: invalid rows = {n_bad}")

            if n_bad == 0 or round_idx >= max_rounds:
                break

            bad_idx = np.where(~valid_mask)[0]

            # regenerate only the bad rows
            X_bad_s = model.sample(C_full[bad_idx])
            X_bad = x_scaler.inverse_transform(X_bad_s)

            X_gen_s[bad_idx] = X_bad_s
            X_gen[bad_idx] = X_bad

        if n_bad > 0:
            print(f"     Warning: still {n_bad} invalid rows after {max_rounds} rounds.")

        # -----------------------
        # SAVE GENERATED POOL (RAW)
        # -----------------------
        syn_targets = pd.DataFrame(X_gen, columns=target_cols)
        syn_cond = cond_df_full.copy().reset_index(drop=True)

        syn = pd.concat([id_df_full.copy(), syn_cond, syn_targets], axis=1)
        syn["train_frac_hh"] = hh_per_adm1
        syn["train_percent_hh"] = hh_per_adm1
        syn["seed"] = seed
        syn["scaler_method"] = SCALE_METHOD

        out_raw_path = os.path.join(
            seed_dir,
            f"generated_pool_{hh_per_adm1}_seed{seed}.csv"
        )
        syn.to_csv(out_raw_path, index=False)
        print(f"  -> Saved generated pool (raw units) to:\n     {out_raw_path}")
        print(f"     Total synthetic rows: {syn.shape[0]}")

        # -----------------------
        # SAVE GENERATED POOL (SCALED)
        # -----------------------
        syn_scaled = syn.copy()
        syn_scaled[target_cols] = X_gen_s
        out_scaled_path = os.path.join(
            seed_dir,
            f"generated_pool_{hh_per_adm1}_seed{seed}_scaled.csv"
        )
        syn_scaled.to_csv(out_scaled_path, index=False)
        print(f"  -> Saved generated pool (scaled targets) to:\n     {out_scaled_path}")

        # -----------------------
        # SAVE complete_yem_scaled (FULL REAL DF, TARGETS SCALED USING THIS RUN'S SCALER)
        # -----------------------
        X_full = df[target_cols].values.astype(float)
        X_full_s = x_scaler.transform(X_full)

        complete_yem_scaled = df.copy()
        complete_yem_scaled[target_cols] = X_full_s
        complete_yem_scaled["train_frac_hh"] = hh_per_adm1
        complete_yem_scaled["train_percent_hh"] = hh_per_adm1
        complete_yem_scaled["seed"] = seed
        complete_yem_scaled["scaler_method"] = SCALE_METHOD

        complete_scaled_path = os.path.join(
            seed_dir,
            f"full_yem_scaled_train{hh_per_adm1}_seed{seed}.csv"
        )
        complete_yem_scaled.to_csv(complete_scaled_path, index=False)
        print(f"  -> Saved complete_yem_scaled to:\n     {complete_scaled_path}")

print("\nAll runs completed.")