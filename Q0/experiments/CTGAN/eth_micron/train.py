import os
import json
import random
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


# -----------------------
# CONFIG
# -----------------------
DATA_PATH = "/data/shared/fsibilla/clean_code/Q1/experiments/eth_micron/full.csv"
OUTPUT_DIR = "/data/shared/fsibilla/clean_code/Q0/experiments/CTGAN/eth_micron/results"

SEEDS = [1, 2, 3, 4, 5]

# CTGAN hyperparams
EPOCHS = 600
BATCH_SIZE = 500
VERBOSE = True
USE_CUDA = True

target_cols = [
    "va_ai", "fol_ai", "vb12_ai",
    "fe_ai", "zn_ai",
    "avg_adult_education", "log_exp"
]

# This is the only non-target column used in training,
# so CTGAN learns the joint distribution of targets + adm1.
adm1_col = "adm1name"


# -----------------------
# UTILITIES
# -----------------------
def assert_cuda_usable():
    """Fail fast if CUDA was requested but is not usable."""
    if not USE_CUDA:
        return True

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


def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -----------------------
# MAIN
# -----------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if USE_CUDA:
        assert_cuda_usable()

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    required_cols = list(set(target_cols + [adm1_col]))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only columns used by CTGAN:
    # targets + adm1 (categorical)
    train_cols = target_cols + [adm1_col]
    df = df.dropna(subset=train_cols).copy()

    # Type safety
    for c in target_cols:
        df[c] = df[c].astype(float)
    df[adm1_col] = df[adm1_col].astype(str)

    n_syn = df.shape[0]

    print(f"Rows after dropna on training cols: {df.shape[0]}")
    print(f"Unique {adm1_col}: {df[adm1_col].nunique()}")
    print(f"Synthetic rows to generate: {n_syn} (same as cleaned real dataset)")

    train_df = df[train_cols].reset_index(drop=True)

    # Save cleaned training data actually used
    cleaned_train_path = os.path.join(OUTPUT_DIR, "cleaned_training_data.csv")
    train_df.to_csv(cleaned_train_path, index=False)
    print(f"Saved cleaned training data to: {cleaned_train_path}")

    # Build SDV metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)

    # Force sdtypes explicitly for safety
    metadata.update_column(column_name=adm1_col, sdtype="categorical")
    for c in target_cols:
        metadata.update_column(column_name=c, sdtype="numerical")

    # Save metadata dict once
    save_json(
        os.path.join(OUTPUT_DIR, "metadata.json"),
        metadata.to_dict()
    )

    # Save experiment-level metadata once
    run_meta = {
        "data_path": DATA_PATH,
        "n_rows_train": int(train_df.shape[0]),
        "n_syn": int(n_syn),
        "generation_mode": "unconditional_sample_same_n_as_cleaned_real_data",
        "model": "CTGANSynthesizer",
        "conditioning_used_in_training": adm1_col,
        "conditioning_used_in_generation": None,
        "train_columns": train_cols,
        "target_cols": target_cols,
        "adm1_col": adm1_col,
        "ctgan_params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "cuda": USE_CUDA,
            "verbose": VERBOSE
        }
    }
    save_json(os.path.join(OUTPUT_DIR, "run_metadata.json"), run_meta)
    print(f"Saved run metadata to: {os.path.join(OUTPUT_DIR, 'run_metadata.json')}")

    for seed in SEEDS:
        print("\n==============================")
        print(f"Seed: {seed}")
        print("==============================")
        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        seed_everything(seed)

        # Train CTGAN on full cleaned dataset
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            cuda=USE_CUDA,
            verbose=VERBOSE
        )

        print("-> Fitting CTGAN on FULL dataset (targets + adm1)...")
        synthesizer.fit(train_df)

        # Save synthesizer if possible
        try:
            synthesizer.save(os.path.join(seed_dir, "ctgan_synthesizer.pkl"))
            print("-> Saved CTGAN synthesizer.")
        except Exception as e:
            print(f"-> Could not save synthesizer (non-fatal): {repr(e)}")

        # Generate WITHOUT conditioning at generation time
        print(f"-> Generating {n_syn} synthetic rows without conditioning...")
        synthetic_df = synthesizer.sample(num_rows=n_syn)

        # Reorder columns for consistency
        synthetic_df = synthetic_df[train_cols].copy()
        synthetic_df["seed"] = seed

        out_csv = os.path.join(seed_dir, f"synthetic_pool.csv")
        synthetic_df.to_csv(out_csv, index=False)
        print(f"-> Saved synthetic pool to: {out_csv}")

        # Save training data with seed for traceability
        real_out = train_df.copy()
        real_out["seed"] = seed
        real_out_path = os.path.join(seed_dir, f"full_eth_cleaned_used_for_training_seed{seed}.csv")
        real_out.to_csv(real_out_path, index=False)
        print(f"-> Saved cleaned real training data to: {real_out_path}")

        # Save per-seed config
        save_json(
            os.path.join(seed_dir, "seed_config.json"),
            {
                "seed": seed,
                "n_syn": int(n_syn),
                "generation_mode": "unconditional_sample_same_n_as_cleaned_real_data",
                "conditioning_used_in_training": adm1_col,
                "conditioning_used_in_generation": None
            }
        )

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()