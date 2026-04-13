import os
import json
import random
import numpy as np
import pandas as pd

from probaforms.models import RealNVP  


# ============================================================
# SCRIPT (ZWE MICS): conditioning on continuous ADM1 features +
#                        one-hot(area), PSU sampling by exact
#                        PSU-per-ADM1 levels:
#                        - always >=1 PSU per ADM1
#                        - first PSU per ADM1 always guaranteed
#                        - sector chosen proportionally within ADM1
#                        - additional PSUs sampled proportionally
#                          to remaining (ADM1, sector) PSU counts
# ============================================================

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/zwe_mics/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q1/experiments/zwe_mics/results"

PSU_PER_ADM1_LEVELS = [
    #1,
    #2,
    4,
    8
]

SEEDS = [1, 2, 3, 4, 5]

target_cols = [
    "space_per_person", "avg_adult_education", "wscore"
]

cond_cols_adm1 = [
    "entropy_1", "rwi_1",
    "sector"
]
sector_col = "sector"  

adm1_name_col = "adm1name"

psu_col = "psu"

extra_id_candidates = ["adm1geometry"]

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


def clip_to_extremes_plus_margin(X_gen_s, X_ref_s, margin=0.10, eps=1e-12):
    """
    Clip each dimension of X_gen_s to [min_ref - margin*range_ref, max_ref + margin*range_ref],
    where min_ref/max_ref are computed from X_ref_s per dimension.
    """
    ref_min = np.min(X_ref_s, axis=0)
    ref_max = np.max(X_ref_s, axis=0)
    ref_range = np.maximum(ref_max - ref_min, eps)

    lo = ref_min - margin * ref_range
    hi = ref_max + margin * ref_range
    return np.clip(X_gen_s, lo, hi)


def modal_value(series: pd.Series):
    m = series.mode(dropna=True)
    return m.iloc[0] if not m.empty else np.nan


# -----------------------
# NEW PSU SAMPLING HELPERS
# -----------------------
def sample_one_psu_per_adm1_proportional_sector(psu_info_adm1: pd.DataFrame,
                                                rng: np.random.RandomState):
    """
    Select exactly 1 PSU within one ADM1.
    Sector is chosen proportionally to PSU counts in that ADM1.
    Then one PSU is drawn uniformly within the chosen sector.
    """
    if psu_info_adm1.empty:
        raise ValueError("psu_info_adm1 is empty")

    sector_counts = psu_info_adm1["psu_modal_sector"].value_counts().sort_index()
    sectors = sector_counts.index.to_numpy()
    probs = (sector_counts.values / sector_counts.values.sum()).astype(float)

    chosen_sector = rng.choice(sectors, p=probs)
    pool = psu_info_adm1.loc[
        psu_info_adm1["psu_modal_sector"] == chosen_sector, psu_col
    ].values

    if len(pool) == 0:
        pool = psu_info_adm1[psu_col].values

    return rng.choice(pool, size=1, replace=False)[0]


def sample_additional_psus_weighted_by_remaining_adm1_sector(psu_info_df: pd.DataFrame,
                                                             already_selected,
                                                             n_extra: int,
                                                             rng: np.random.RandomState):
    """
    Sequentially sample extra PSUs without replacement.
    At each step, choose among remaining (adm1, sector) cells with probability
    proportional to the number of remaining PSUs in that cell, then sample one PSU
    uniformly from that chosen cell.

    This makes extra PSU allocation depend jointly on:
      - how many PSUs remain in each ADM1
      - sector prevalence within each ADM1
    """
    if n_extra <= 0:
        return np.array([], dtype=object)

    selected_set = set(already_selected)
    remaining_df = psu_info_df.loc[~psu_info_df[psu_col].isin(selected_set)].copy()

    sampled = []

    for _ in range(n_extra):
        if remaining_df.empty:
            break

        cell_counts = (
            remaining_df
            .groupby(["psu_modal_adm1", "psu_modal_sector"], dropna=False)[psu_col]
            .count()
            .reset_index(name="n_remaining")
        )

        weights = cell_counts["n_remaining"].values.astype(float)
        probs = weights / weights.sum()

        chosen_idx = rng.choice(np.arange(len(cell_counts)), p=probs)
        chosen_adm1 = cell_counts.iloc[chosen_idx]["psu_modal_adm1"]
        chosen_sector = cell_counts.iloc[chosen_idx]["psu_modal_sector"]

        pool = remaining_df.loc[
            (remaining_df["psu_modal_adm1"] == chosen_adm1) &
            (remaining_df["psu_modal_sector"] == chosen_sector),
            psu_col
        ].values

        if len(pool) == 0:
            pool = remaining_df[psu_col].values

        chosen_psu = rng.choice(pool, size=1, replace=False)[0]
        sampled.append(chosen_psu)

        remaining_df = remaining_df.loc[remaining_df[psu_col] != chosen_psu].copy()

    return np.array(sampled, dtype=object)


# -----------------------
# LOAD & BASIC CLEANING
# -----------------------
df = pd.read_csv(DATA_PATH)

required_cols = [adm1_name_col, sector_col, psu_col] + target_cols + cond_cols_adm1
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

need_cols = list(set(target_cols + cond_cols_adm1 + [adm1_name_col, sector_col, psu_col]))
df = df.dropna(subset=need_cols).copy()

# NOTE: original behavior preserved
for col in target_cols:
    df[col] = df[col].astype(float)

for col in cond_cols_adm1:
    if col != sector_col:
        df[col] = df[col].astype(float)


# PSU + ADM1 safe types
df[psu_col] = df[psu_col].astype(str)
df[adm1_name_col] = df[adm1_name_col].astype(str)

print(f"Total rows after cleaning: {df.shape[0]}")
print(f"Sectors present (int codes): {sorted(df[sector_col].unique())}")
print(f"Unique PSUs: {df[psu_col].nunique()}")
print(f"Unique ADM1s: {df[adm1_name_col].nunique()}")

# -----------------------
# PSU -> modal(sector), modal(adm1)
# -----------------------
psu_info_df = (
    df.groupby(psu_col)
      .agg(
          psu_modal_sector=(sector_col, modal_value),
          psu_modal_adm1=(adm1_name_col, modal_value),
      )
      .reset_index()
)

psu_info_df = psu_info_df.dropna(subset=["psu_modal_sector", "psu_modal_adm1"]).copy()
psu_info_df["psu_modal_sector"] = psu_info_df["psu_modal_sector"].astype(int)
psu_info_df["psu_modal_adm1"] = psu_info_df["psu_modal_adm1"].astype(str)

unique_psus = psu_info_df[psu_col].values

print("\nPSU modal area counts:")
print(psu_info_df["psu_modal_sector"].value_counts().sort_index())
print("\nPSU ADM1 counts:")
print(psu_info_df["psu_modal_adm1"].value_counts().sort_index())

# -----------------------
# CONDITIONS: (continuous base) + one-hot(area)
# -----------------------
cond_base_cols = [c for c in cond_cols_adm1 if c != sector_col]
cond_base_full = df[cond_base_cols]
sector_dummies_full = pd.get_dummies(df[sector_col], prefix=sector_col)

cond_df_full = pd.concat([cond_base_full, sector_dummies_full], axis=1)
C_full = cond_df_full.values.astype(float)

# -----------------------
# PREPARE IDs (for output)
# -----------------------
id_cols = [adm1_name_col]
if adm1_name_col in df.columns:
    id_cols.append(adm1_name_col)

keep_extra = [col for col in extra_id_candidates if col in df.columns]
id_df_full = df[id_cols + keep_extra].reset_index(drop=True)

# -----------------------
# MAIN LOOP
# -----------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for psu_per_adm1 in PSU_PER_ADM1_LEVELS:
    print(f"\n==============================")
    print(f"Training size level: {psu_per_adm1} PSU per ADM1")
    print(f"==============================")

    level_dir = os.path.join(OUTPUT_DIR, f"train_{psu_per_adm1}_scaled")
    os.makedirs(level_dir, exist_ok=True)

    for seed in SEEDS:
        print(f"\n--- Seed {seed} for PSU-per-ADM1 level {psu_per_adm1} ---")

        seed_dir = os.path.join(level_dir, f"seed_{seed}_scaled")
        os.makedirs(seed_dir, exist_ok=True)

        # -----------------------
        # PSU sampling
        # -----------------------
        rng = np.random.RandomState(seed)

        n_psu_total = len(unique_psus)
        unique_adm1s = np.sort(psu_info_df["psu_modal_adm1"].unique())
        n_adm1 = len(unique_adm1s)

        n_psu_train_target = int(psu_per_adm1 * n_adm1)
        n_psu_train_target = max(n_psu_train_target, n_adm1)
        n_psu_train_target = min(n_psu_train_target, n_psu_total)

        # First guaranteed PSU per ADM1
        mandatory_psus = []
        for adm1 in unique_adm1s:
            adm1_psu_df = psu_info_df.loc[psu_info_df["psu_modal_adm1"] == adm1].copy()
            pick = sample_one_psu_per_adm1_proportional_sector(adm1_psu_df, rng)
            mandatory_psus.append(pick)

        mandatory_psus = np.array(mandatory_psus, dtype=object)

        remaining_needed = int(n_psu_train_target - len(mandatory_psus))

        if remaining_needed > 0:
            sampled_remaining = sample_additional_psus_weighted_by_remaining_adm1_sector(
                psu_info_df=psu_info_df,
                already_selected=mandatory_psus,
                n_extra=remaining_needed,
                rng=rng
            )
            train_psus = np.concatenate([mandatory_psus, sampled_remaining])
        else:
            train_psus = mandatory_psus.copy()

        train_psus = np.unique(train_psus)
        n_psu_train = len(train_psus)

        df_train = df[df[psu_col].isin(train_psus)].copy()
        train_adm1_covered = df_train[adm1_name_col].nunique()

        print(
            f"  -> Training on {df_train.shape[0]} rows from {n_psu_train}/{n_psu_total} PSUs "
            f"(target {n_psu_train_target} PSUs; level={psu_per_adm1} PSU per ADM1), "
            f"guaranteeing >=1 PSU per {adm1_name_col} "
            f"(covered ADM1s: {train_adm1_covered}/{n_adm1})."
        )

        # -----------------------
        # ARRAYS
        # -----------------------
        X_train_df = df_train[target_cols].copy()

        low_q = 0.005
        high_q = 0.995
        
        for c in target_cols:
            lo = X_train_df[c].quantile(low_q)
            hi = X_train_df[c].quantile(high_q)
            X_train_df[c] = X_train_df[c].clip(lo, hi)
        
        X_train = X_train_df.values.astype(float)
        C_train = cond_df_full.loc[df_train.index].values.astype(float)

        # -----------------------
        # SCALER FIT ON TRAIN
        # -----------------------
        x_scaler = MonotoneLinearScaler(method=SCALE_METHOD, eps=EPS).fit(X_train)
        X_train_s = x_scaler.transform(X_train)

        # Save scaler params + PSUs used
        scaler_path = os.path.join(
            seed_dir,
            f"x_scaler_{SCALE_METHOD}_train{psu_per_adm1}_seed{seed}.json"
        )
        with open(scaler_path, "w") as f:
            json.dump(
                {
                    "target_cols": target_cols,
                    "train_frac_psu": psu_per_adm1,
                    "train_percent_psu": psu_per_adm1,
                    "psu_per_adm1": psu_per_adm1,
                    "seed": seed,
                    "psu_col": psu_col,
                    "sector_col": sector_col,
                    "adm1_col": adm1_name_col,
                    "sampling_rule": "guarantee_1_per_adm1_then_sample_remaining_by_adm1_sector_remaining_counts",
                    "first_psu_rule": "within_each_adm1_sector_sampled_proportionally_to_psu_sector_counts",
                    "extra_psu_rule": "sequential_sampling_proportional_to_remaining_psu_counts_over_adm1_sector_cells",
                    "psu_sector_rule": "modal",
                    "psu_adm1_rule": "modal",
                    "n_psu_total": int(n_psu_total),
                    "n_psu_train_target": int(n_psu_train_target),
                    "n_psu_train_actual": int(n_psu_train),
                    "n_adm1_total": int(n_adm1),
                    "n_adm1_covered": int(train_adm1_covered),
                    "train_psus": train_psus.tolist(),
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
            f"train_subset_{psu_per_adm1}_seed{seed}.csv"
        )
        df_train.to_csv(subsample_raw_path, index=False)
        print(f"  -> Saved training data (raw) to:\n     {subsample_raw_path}")

        df_train_scaled = df_train.copy()
        df_train_scaled[target_cols] = X_train_s
        subsample_scaled_path = os.path.join(
            seed_dir,
            f"train_subset_{psu_per_adm1}_seed{seed}_scaled.csv"
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
            n_epochs=400,
            hidden=(32, 32),
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
        syn["train_frac_psu"] = psu_per_adm1
        syn["train_percent_psu"] = psu_per_adm1
        syn["seed"] = seed
        syn["scaler_method"] = SCALE_METHOD

        out_raw_path = os.path.join(
            seed_dir,
            f"generated_pool_{psu_per_adm1}_seed{seed}.csv"
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
            f"generated_pool_{psu_per_adm1}_seed{seed}_scaled.csv"
        )
        syn_scaled.to_csv(out_scaled_path, index=False)
        print(f"  -> Saved generated pool (scaled targets) to:\n     {out_scaled_path}")

        # -----------------------
        # SAVE complete_zwe_scaled
        # -----------------------
        X_full = df[target_cols].values.astype(float)
        X_full_s = x_scaler.transform(X_full)

        complete_nga_scaled = df.copy()
        complete_nga_scaled[target_cols] = X_full_s
        complete_nga_scaled["train_frac_psu"] = psu_per_adm1
        complete_nga_scaled["train_percent_psu"] = psu_per_adm1
        complete_nga_scaled["seed"] = seed
        complete_nga_scaled["scaler_method"] = SCALE_METHOD

        complete_scaled_path = os.path.join(
            seed_dir,
            f"full_zwe_scaled_train{psu_per_adm1}_seed{seed}.csv"
        )
        complete_nga_scaled.to_csv(complete_scaled_path, index=False)
        print(f"  -> Saved complete_zwe_scaled to:\n     {complete_scaled_path}")

print("\nAll runs completed.")