import os
import json
import random
import numpy as np
import pandas as pd

from probaforms.models import RealNVP  


# ============================================================
# SCRIPT (eth micron): conditioning on continuous ADM1 features
#                        (NO sector one-hot — model trained on
#                        urban-biased sample should not condition
#                        on a sector it may never have seen).
#
#                        PSU sampling is URBAN-BIASED:
#                        - For each ADM1, sample PSUs from urban
#                          sector only (sector == 1).
#                        - If an ADM1 has NO urban PSUs, fall back
#                          to random PSU selection regardless of sector.
#                        - Additional PSUs (beyond 1-per-ADM1) also
#                          drawn from urban pool where possible, with
#                          the same fallback logic.
# ============================================================

# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/lka_vam/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q1/biased/lka_vam/results"

URBAN_SECTOR_CODE = 1  # sector value that identifies urban PSUs

PSU_PER_ADM1_LEVELS = [
    1,
]

SEEDS = [1, 2, 3, 4, 5]

target_cols = [
    'education_score', 'log_income', 'space_per_person', 'FES', 'FCS','rCSI'
]

# NOTE: sector_col is intentionally EXCLUDED from conditioning columns.
# The model is trained on an urban-biased sample and should not condition
# on sector, since rural sector may be unseen or underrepresented.
cond_cols_adm1 = [
    "entropy_2", "rwi_2",
]

sector_col = "sector"   # used only for PSU sampling, NOT for conditioning

adm1_name_col = "adm1name"

psu_col = "psu"

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
# URBAN-BIASED PSU SAMPLING HELPERS
# -----------------------
def sample_one_psu_urban_biased(psu_info_adm1: pd.DataFrame,
                                rng: np.random.RandomState,
                                urban_code: int = URBAN_SECTOR_CODE):
    """
    Select exactly 1 PSU within one ADM1, preferring urban PSUs.
    - If urban PSUs exist: draw uniformly from them.
    - Fallback: draw uniformly from ALL PSUs in the ADM1 (any sector).
    """
    if psu_info_adm1.empty:
        raise ValueError("psu_info_adm1 is empty")

    urban_pool = psu_info_adm1.loc[
        psu_info_adm1["psu_modal_sector"] == urban_code, psu_col
    ].values

    if len(urban_pool) > 0:
        pool = urban_pool
        used_fallback = False
    else:
        pool = psu_info_adm1[psu_col].values
        used_fallback = True

    return rng.choice(pool, size=1, replace=False)[0], used_fallback


def sample_additional_psus_urban_biased(psu_info_df: pd.DataFrame,
                                        already_selected,
                                        n_extra: int,
                                        rng: np.random.RandomState,
                                        urban_code: int = URBAN_SECTOR_CODE):
    """
    Sequentially sample extra PSUs without replacement, urban-biased.

    At each step:
      1. Build the remaining urban pool across all ADM1s.
      2. If non-empty, sample uniformly from it.
      3. If empty (all urban PSUs exhausted), sample uniformly from
         all remaining PSUs regardless of sector.
    """
    if n_extra <= 0:
        return np.array([], dtype=object), 0

    selected_set = set(already_selected)
    remaining_df = psu_info_df.loc[~psu_info_df[psu_col].isin(selected_set)].copy()

    sampled = []
    n_fallback = 0

    for _ in range(n_extra):
        if remaining_df.empty:
            break

        urban_remaining = remaining_df.loc[
            remaining_df["psu_modal_sector"] == urban_code
        ]

        if len(urban_remaining) > 0:
            pool = urban_remaining[psu_col].values
        else:
            pool = remaining_df[psu_col].values
            n_fallback += 1

        chosen_psu = rng.choice(pool, size=1, replace=False)[0]
        sampled.append(chosen_psu)
        remaining_df = remaining_df.loc[remaining_df[psu_col] != chosen_psu].copy()

    return np.array(sampled, dtype=object), n_fallback


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

for col in target_cols:
    df[col] = df[col].astype(float)

for col in cond_cols_adm1:
    df[col] = df[col].astype(float)

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

print("\nPSU modal sector counts:")
print(psu_info_df["psu_modal_sector"].value_counts().sort_index())
print("\nPSU ADM1 counts:")
print(psu_info_df["psu_modal_adm1"].value_counts().sort_index())

# Report which ADM1s have no urban PSUs (will use fallback)
adm1_urban_counts = (
    psu_info_df[psu_info_df["psu_modal_sector"] == URBAN_SECTOR_CODE]
    .groupby("psu_modal_adm1")[psu_col].count()
)
all_adm1s_in_psu = psu_info_df["psu_modal_adm1"].unique()
no_urban_adm1s = [a for a in all_adm1s_in_psu if a not in adm1_urban_counts.index]
if no_urban_adm1s:
    print(f"\nADM1s with NO urban PSUs (will use random fallback): {no_urban_adm1s}")
else:
    print("\nAll ADM1s have at least one urban PSU.")

# -----------------------
# CONDITIONS: continuous ADM1 features only (NO sector one-hot)
# -----------------------
# cond_cols_adm1 already excludes sector_col — just use directly.
C_full = df[cond_cols_adm1].values.astype(float)

# Keep a DataFrame version for output alignment
cond_df_full = df[cond_cols_adm1].copy().reset_index(drop=True)

# -----------------------
# PREPARE IDs (for output)
# -----------------------
id_cols = [adm1_name_col]
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

    level_dir = os.path.join(OUTPUT_DIR, f"train_{psu_per_adm1}_scaled_urban")
    os.makedirs(level_dir, exist_ok=True)

    for seed in SEEDS:
        print(f"\n--- Seed {seed} for PSU-per-ADM1 level {psu_per_adm1} ---")

        seed_dir = os.path.join(level_dir, f"seed_{seed}_scaled_urban")
        os.makedirs(seed_dir, exist_ok=True)

        # -----------------------
        # PSU SAMPLING (urban-biased)
        # -----------------------
        rng = np.random.RandomState(seed)

        n_psu_total = len(unique_psus)
        unique_adm1s = np.sort(psu_info_df["psu_modal_adm1"].unique())
        n_adm1 = len(unique_adm1s)

        n_psu_train_target = int(psu_per_adm1 * n_adm1)
        n_psu_train_target = max(n_psu_train_target, n_adm1)
        n_psu_train_target = min(n_psu_train_target, n_psu_total)

        # Guaranteed first PSU per ADM1: urban if available, else random fallback
        mandatory_psus = []
        n_mandatory_fallbacks = 0
        for adm1 in unique_adm1s:
            adm1_psu_df = psu_info_df.loc[psu_info_df["psu_modal_adm1"] == adm1].copy()
            pick, used_fallback = sample_one_psu_urban_biased(adm1_psu_df, rng)
            mandatory_psus.append(pick)
            if used_fallback:
                n_mandatory_fallbacks += 1

        mandatory_psus = np.array(mandatory_psus, dtype=object)

        if n_mandatory_fallbacks > 0:
            print(f"  -> Fallback (no urban PSU) triggered for {n_mandatory_fallbacks} ADM1(s) "
                  f"during mandatory 1-per-ADM1 selection.")

        remaining_needed = int(n_psu_train_target - len(mandatory_psus))

        n_extra_fallbacks = 0
        if remaining_needed > 0:
            sampled_remaining, n_extra_fallbacks = sample_additional_psus_urban_biased(
                psu_info_df=psu_info_df,
                already_selected=mandatory_psus,
                n_extra=remaining_needed,
                rng=rng
            )
            train_psus = np.concatenate([mandatory_psus, sampled_remaining])
            if n_extra_fallbacks > 0:
                print(f"  -> Fallback triggered for {n_extra_fallbacks} additional PSU(s) "
                      f"(urban pool exhausted during extra sampling).")
        else:
            train_psus = mandatory_psus.copy()

        train_psus = np.unique(train_psus)
        n_psu_train = len(train_psus)

        df_train = df[df[psu_col].isin(train_psus)].copy()
        train_adm1_covered = df_train[adm1_name_col].nunique()

        # Report sector composition of the training set
        train_sector_counts = df_train[sector_col].value_counts().sort_index()
        print(
            f"  -> Training on {df_train.shape[0]} rows from {n_psu_train}/{n_psu_total} PSUs "
            f"(target {n_psu_train_target} PSUs; level={psu_per_adm1} PSU per ADM1), "
            f"guaranteeing >=1 PSU per {adm1_name_col} "
            f"(covered ADM1s: {train_adm1_covered}/{n_adm1})."
        )
        print(f"     Sector composition in training set: {train_sector_counts.to_dict()}")

        # -----------------------
        # ARRAYS
        # -----------------------
        X_train = df_train[target_cols].values.astype(float)
        C_train = df_train[cond_cols_adm1].values.astype(float)

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
                    "cond_cols": cond_cols_adm1,

                    # kept for backward compatibility of structure
                    "train_frac_psu": psu_per_adm1,
                    "train_percent_psu": psu_per_adm1,

                    "psu_per_adm1": psu_per_adm1,
                    "seed": seed,
                    "psu_col": psu_col,
                    "sector_col": sector_col,
                    "adm1_col": adm1_name_col,
                    "sampling_rule": "urban_biased_guarantee_1_per_adm1_fallback_random_if_no_urban",
                    "first_psu_rule": "urban_only_if_available_else_random_any_sector",
                    "extra_psu_rule": "urban_pool_first_else_random_any_sector",
                    "sector_in_conditioning": False,
                    "urban_sector_code": URBAN_SECTOR_CODE,
                    "n_mandatory_fallbacks": int(n_mandatory_fallbacks),
                    "n_extra_fallbacks": int(n_extra_fallbacks),
                    "psu_sector_rule": "modal",
                    "psu_adm1_rule": "modal",
                    "n_psu_total": int(n_psu_total),
                    "n_psu_train_target": int(n_psu_train_target),
                    "n_psu_train_actual": int(n_psu_train),
                    "n_adm1_total": int(n_adm1),
                    "n_adm1_covered": int(train_adm1_covered),
                    "train_psus": train_psus.tolist(),
                    "train_sector_composition": train_sector_counts.to_dict(),
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
            n_epochs=600,
            hidden=(64, 64),
        )

        print("  -> Fitting RealNVP model (on scaled targets, no sector conditioning)...")
        model.fit(X_train_s, C_train)

        # -----------------------
        # GENERATE FOR FULL CONDITIONS DATASET
        # -----------------------
        print("  -> Generating samples for FULL conditions dataset...")

        q_low = 0.00
        q_high = 1.
        max_rounds = 50

        lower_bounds = df_train[target_cols].quantile(q_low)
        upper_bounds = df_train[target_cols].quantile(q_high)

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

        syn = pd.concat([id_df_full.copy(), cond_df_full.copy(), syn_targets], axis=1)
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
        # SAVE complete_eth_scaled
        # -----------------------
        X_full = df[target_cols].values.astype(float)
        X_full_s = x_scaler.transform(X_full)

        complete_eth_scaled = df.copy()
        complete_eth_scaled[target_cols] = X_full_s
        complete_eth_scaled["train_frac_psu"] = psu_per_adm1
        complete_eth_scaled["train_percent_psu"] = psu_per_adm1
        complete_eth_scaled["seed"] = seed
        complete_eth_scaled["scaler_method"] = SCALE_METHOD

        complete_scaled_path = os.path.join(
            seed_dir,
            f"full_lka_scaled_train{psu_per_adm1}_seed{seed}.csv"
        )
        complete_eth_scaled.to_csv(complete_scaled_path, index=False)
        print(f"  -> Saved complete_lka_scaled to:\n     {complete_scaled_path}")

print("\nAll runs completed.")