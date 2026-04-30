# Graded conditioning ablation

## What this pipeline is for

The Science Advances reviewer wrote (point 2):

> "The mechanism claim ('collapses toward resampling without informative covariates') is asserted more than demonstrated. The NF-vs-cNF contrast is suggestive but not definitive — the NF baseline only encodes region-as-categorical, which is a very weak conditioning signal by design. A more convincing test would involve a graded series of conditioning richness: e.g., region only → region + one covariate → region + two → full. Right now the binary contrast leaves open whether the gain is monotonic in covariate informativeness, or whether there's a threshold effect, or whether some covariates carry essentially all the signal."

This is the experiment that answers them.

## What "intermediate conditioning levels" means

The current paper compares two extremes:

| Variant | Conditioning input | Where in repo |
|---|---|---|
| NF (region only) | one-hot of ADM1 | `Q1/experiments_minimal/` |
| cNF (full) | one-hot of ADM1 + sector + ALL continuous covariates | `Q1/experiments/` |

The reviewer asks: what happens **between** these two extremes? Is the cNF gain mostly from one informative covariate, or does each one help a little?

We add intermediate variants by only enabling a subset of the continuous covariates at a time:

| Variant | Conditioning | Directory |
|---|---|---|
| `NF + sector` | ADM1 + sector | `Q1/experiments_minimal_sector/` (already built) |
| `NF + sector + wealth` | ADM1 + sector + `rwi_2` | `wealth_lka_vam/`, `wealth_nga_micron/` |
| `NF + sector + wealth + accessibility` | ADM1 + sector + `rwi_2` + `r3q` | `wealth_access_nga_micron/` |
| `cNF (full)` | everything | `Q1/experiments/` |

For `lka_vam` (only 2 continuous covariates: `entropy_2` and `rwi_2`) you get **4 levels**:
1. region only (`experiments_minimal/lka_vam`)
2. region + sector (`experiments_minimal_sector/lka_vam`)
3. region + sector + wealth (`experiments_graded/wealth_lka_vam`) ← NEW
4. region + sector + wealth + meteo = full (`experiments/lka_vam`)

For `nga_micron` (many continuous covariates) you get **5 levels**:
1. region only (`experiments_minimal/nga_micron`)
2. region + sector (`experiments_minimal_sector/nga_micron`)
3. region + sector + wealth (`experiments_graded/wealth_nga_micron`) ← NEW
4. region + sector + wealth + accessibility (`experiments_graded/wealth_access_nga_micron`) ← NEW
5. region + sector + wealth + accessibility + meteo (etc) = full (`experiments/nga_micron`)

## What the headline figure would look like

X axis: conditioning level (1 = sparsest, 4–5 = full).
Y axis: mean EMD improvement over oversampling (averaged over seeds, ADM1s, target variables for each (dataset, level)).
One line per dataset.

If the line rises monotonically, the reviewer's objection is answered: every additional covariate group adds something. If it plateaus early, then a small subset carries most of the signal — also informative.

## Why only `lka_vam` and `nga_micron`

Pragmatic: these two are the simplest demonstration. `lka_vam` because we want at least one short pipeline (4 levels) for a clean illustration; `nga_micron` because it has the richest set of covariates so it's the dataset with the most levels and the most discriminating power. Adding more datasets is straightforward — just edit `LEVELS` in `tools/revision-patches/create_graded_pipelines.py` and re-run.

## What you need to do to use it

1. Run the new training pipelines (3 directories, each has the same structure as `experiments/<ds>/`):
   ```
   cd Q1/experiments_graded/wealth_lka_vam && python train.py
   cd Q1/experiments_graded/wealth_nga_micron && python train.py
   cd Q1/experiments_graded/wealth_access_nga_micron && python train.py
   ```
   Each run takes about as long as one `experiments/<ds>` run.

2. Run the corresponding `evaluate.ipynb` in each `results/` folder to produce the metrics CSV.

3. Aggregate across levels: write a small notebook that loads metrics from all the levels (`experiments_minimal/lka_vam`, `experiments_minimal_sector/lka_vam`, `experiments_graded/wealth_lka_vam`, `experiments/lka_vam`) and plots improvement-vs-level. I haven't written this aggregation notebook yet because the curve only makes sense once at least one dataset is fully run.

## Whether to do this at all

If the cNF-vs-NF contrast already convinces the reviewer (because the magnitude of the gap is so large), the graded ablation is icing. If they push back on the binary contrast, the graded ablation is the answer they're asking for. The decision is yours; the pipelines are here so they can be run quickly if needed.
