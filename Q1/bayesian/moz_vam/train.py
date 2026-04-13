import os
import json
import random
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from probaforms.models import RealNVP


# =============================================================================
# CONFIG
# =============================================================================

DATA_PATH   = "/data/shared/fsibilla/clean_code/Q1/experiments/moz_vam/full.csv"
RESULTS_DIR = "/data/shared/fsibilla/clean_code/Q1/experiments/moz_vam/results"
OUTPUT_DIR  = "/data/shared/fsibilla/clean_code/Q1/bayesian/moz_vam/results"

PSU_LEVEL = 1
SEEDS     = [1, 2, 3, 4, 5]    # one training subset per seed
N_RNVP    = 5                  # how many times to reinitialize and retrain RealNVP

TARGET_COLS    = ['FCS', 'rCSI', 'FGVitA', 'FGProtein', 'FGHIron']
COND_BASE_COLS = ["entropy_2", "rwi_2"]
SECTOR_COL     = "sector"
ADM1_COL       = "adm1name"
PSU_COL        = "psu"
SCALE_METHOD   = "iqr"

RNVP_EPOCHS = 600
RNVP_HIDDEN = (64, 64)

# ---- bounds rejection settings ----
Q_LOW = 0.00
Q_HIGH = 1.00
MAX_ROUNDS = 50


# =============================================================================
# SCALER  (same as training script)
# =============================================================================

class MonotoneLinearScaler:
    def __init__(self, method="iqr", eps=1e-12):
        self.method  = method
        self.eps     = eps
        self.center_ = None
        self.scale_  = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.method == "zscore":
            self.center_ = np.nanmean(X, axis=0)
            self.scale_  = np.nanstd(X, axis=0, ddof=0)
        else:
            self.center_ = np.nanmedian(X, axis=0)
            self.scale_  = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
        self.scale_ = np.where(self.scale_ > self.eps, self.scale_, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.center_) / self.scale_

    def inverse_transform(self, Xs):
        return np.asarray(Xs, dtype=float) * self.scale_ + self.center_

    def from_dict(self, d):
        self.center_ = np.array(d["center"])
        self.scale_  = np.array(d["scale"])
        return self


# =============================================================================
# BAYESIAN MVN
# =============================================================================

class BayesianMVN:
    """
    Multivariate Bayesian linear regression with flat prior.
    Posterior is analytic — no randomness, no seeds.
    Given a training set, mu and std are fully determined.
    """

    def __init__(self):
        self.B_hat   = None   # (p, q) posterior mean weights
        self.Sigma   = None   # (q, q) residual covariance
        self.XtX_inv = None
        self.p       = None

    def fit(self, Y, C):
        n, q  = Y.shape
        C_aug = np.hstack([np.ones((n, 1)), C])
        self.p       = C_aug.shape[1]
        self.XtX_inv = np.linalg.pinv(C_aug.T @ C_aug)
        self.B_hat   = self.XtX_inv @ C_aug.T @ Y
        resid        = Y - C_aug @ self.B_hat
        self.Sigma   = (resid.T @ resid) / max(n - self.p, 1)
        return self

    def predict_mean(self, C_new):
        C_aug = np.hstack([np.ones((len(C_new), 1)), C_new])
        return C_aug @ self.B_hat                                   # (m, q)

    def predict_std(self, C_new):
        # predictive std per target per row: sqrt((1 + leverage) * diag(Sigma))
        C_aug = np.hstack([np.ones((len(C_new), 1)), C_new])
        lev   = np.einsum('ij,jk,ik->i', C_aug, self.XtX_inv, C_aug)  # (m,)
        return np.sqrt((1.0 + lev)[:, None] * np.diag(self.Sigma))     # (m, q)


# =============================================================================
# LOAD FULL DATASET AND BUILD CONDITION MATRIX
# =============================================================================

df_full = pd.read_csv(DATA_PATH)

need_cols = TARGET_COLS + COND_BASE_COLS + [SECTOR_COL, ADM1_COL, PSU_COL]
df_full   = df_full.dropna(subset=need_cols).copy()

for c in TARGET_COLS + COND_BASE_COLS:
    df_full[c] = df_full[c].astype(float)

# build condition matrix once — same columns will be reused for every model
sector_dummies_full = pd.get_dummies(df_full[SECTOR_COL], prefix=SECTOR_COL)
cond_df_full        = pd.concat([df_full[COND_BASE_COLS], sector_dummies_full], axis=1)
C_full              = cond_df_full.values.astype(float)
Y_full              = df_full[TARGET_COLS].values.astype(float)

print(f"Full dataset: {df_full.shape[0]} rows, {C_full.shape[1]} condition features")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# MAIN LOOP — one iteration per training subset (seed)
# =============================================================================

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    seed_dir     = os.path.join(RESULTS_DIR, f"train_{PSU_LEVEL}_scaled", f"seed_{seed}_scaled")
    out_seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(out_seed_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load the already-subsampled training data
    # -------------------------------------------------------------------------
    train_raw_path    = os.path.join(seed_dir, f"train_subset_{PSU_LEVEL}_seed{seed}.csv")
    train_scaled_path = os.path.join(seed_dir, f"train_subset_{PSU_LEVEL}_seed{seed}_scaled.csv")

    df_train_raw    = pd.read_csv(train_raw_path)
    df_train_scaled = pd.read_csv(train_scaled_path)

    # align sector dummies with full-data columns (some sectors may be absent in subset)
    train_sector_dummies = pd.get_dummies(df_train_scaled[SECTOR_COL], prefix=SECTOR_COL)
    train_sector_dummies = train_sector_dummies.reindex(
        columns=sector_dummies_full.columns, fill_value=0
    )

    C_train   = pd.concat([df_train_scaled[COND_BASE_COLS], train_sector_dummies], axis=1).values.astype(float)
    Y_train_s = df_train_scaled[TARGET_COLS].values.astype(float)   # already scaled

    # raw version needed for empirical bounds
    Y_train_raw = df_train_raw[TARGET_COLS].values.astype(float)

    # load scaler fitted on this training subset
    scaler_path = os.path.join(seed_dir, f"x_scaler_{SCALE_METHOD}_train{PSU_LEVEL}_seed{seed}.json")
    with open(scaler_path) as f:
        scaler = MonotoneLinearScaler(method=SCALE_METHOD).from_dict(json.load(f))

    print(f"  Training rows: {df_train_scaled.shape[0]}  |  PSUs: {df_train_scaled[PSU_COL].nunique()}")

    # -------------------------------------------------------------------------
    # Bounds from raw training data
    # -------------------------------------------------------------------------
    lower_bounds = df_train_raw[TARGET_COLS].quantile(Q_LOW)
    upper_bounds = df_train_raw[TARGET_COLS].quantile(Q_HIGH)

    # optional hard floors for variables that should not be negative
    for c in ["education_score", "space_per_person", "FES", "FCS", "rCSI"]:
        if c in lower_bounds.index:
            lower_bounds[c] = max(lower_bounds[c], 0.0)

    print("  Bounds for rejection sampling:")
    for c in TARGET_COLS:
        print(f"    {c}: [{lower_bounds[c]:.6f}, {upper_bounds[c]:.6f}]")

    # =========================================================================
    # PART 1 — RealNVP  (retrained N_RNVP times with different init seeds)
    # =========================================================================

    for run_idx in range(N_RNVP):

        # each run gets a deterministic but distinct seed
        run_seed = seed * 100 + run_idx + 1

        random.seed(run_seed)
        np.random.seed(run_seed)
        try:
            import torch
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
        except ImportError:
            pass

        print(f"  [RealNVP] seed={seed}  run={run_idx}  (init seed={run_seed})")

        model = RealNVP(n_epochs=RNVP_EPOCHS, hidden=RNVP_HIDDEN)
        model.fit(Y_train_s, C_train)

        # ---------------------------------------------------------------------
        # Generate one sample per full-data row with rejection/resampling
        # ---------------------------------------------------------------------
        Y_gen_s = model.sample(C_full)
        Y_gen   = scaler.inverse_transform(Y_gen_s)

        round_idx = 0
        n_bad = None

        while True:
            round_idx += 1

            syn_check = pd.DataFrame(Y_gen, columns=TARGET_COLS)

            valid_mask = np.ones(len(syn_check), dtype=bool)
            for c in TARGET_COLS:
                valid_mask &= syn_check[c].between(lower_bounds[c], upper_bounds[c])

            n_bad = (~valid_mask).sum()
            print(f"    Rejection round {round_idx}: invalid rows = {n_bad}")

            if n_bad == 0 or round_idx >= MAX_ROUNDS:
                break

            bad_idx = np.where(~valid_mask)[0]

            # regenerate only the invalid rows
            Y_bad_s = model.sample(C_full[bad_idx])
            Y_bad   = scaler.inverse_transform(Y_bad_s)

            Y_gen_s[bad_idx] = Y_bad_s
            Y_gen[bad_idx]   = Y_bad

        if n_bad > 0:
            print(f"    Warning: still {n_bad} invalid rows after {MAX_ROUNDS} rounds.")

        # save generated pool
        out = pd.DataFrame(Y_gen, columns=TARGET_COLS)
        out["seed"]     = seed
        out["run_idx"]  = run_idx
        out["run_seed"] = run_seed

        out_path = os.path.join(out_seed_dir, f"rnvp_generated_run{run_idx}.csv")
        out.to_csv(out_path, index=False)
        print(f"    -> saved {out_path}")

    # =========================================================================
    # PART 2 — Bayesian MVN  (one fit, fully determined by the training subset)
    # =========================================================================

    print(f"  [Bayesian MVN] seed={seed}")

    bayes = BayesianMVN().fit(Y_train_s, C_train)

    # posterior predictive mean and std (both in scaled space, then convert to raw)
    mu_s  = bayes.predict_mean(C_full)                 # (N, q) scaled
    std_s = bayes.predict_std(C_full)                  # (N, q) scaled

    mu  = scaler.inverse_transform(mu_s)               # raw units
    std = std_s * scaler.scale_                        # std in raw units (chain rule)

    # save mu
    mu_df = pd.DataFrame(mu, columns=[f"mu_{c}" for c in TARGET_COLS])
    mu_df["seed"] = seed
    mu_df.to_csv(os.path.join(out_seed_dir, "bayes_mu.csv"), index=False)

    # save std
    std_df = pd.DataFrame(std, columns=[f"std_{c}" for c in TARGET_COLS])
    std_df["seed"] = seed
    std_df.to_csv(os.path.join(out_seed_dir, "bayes_std.csv"), index=False)

    print(f"    -> saved bayes_mu.csv and bayes_std.csv")


print("\nAll done.")