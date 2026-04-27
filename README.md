# Can Generative AI Recover What the Sample Missed?

Code and analysis for the paper:

> **Can generative AI recover what the sample missed? Evidence from sparse survey data**  
> Federica Sibilla, Vasiliki Voukelatou, Duccio Piovani, Kyriacos Koupparis, Daniela Paolotti, Kyriaki Kalimeri

This repository contains experiments on whether **conditional generative models** can improve sub-national estimates from sparse household survey data. The central question is whether a model trained on a nationally representative but locally sparse survey sample can recover missing sub-national heterogeneity when conditioned on informative contextual covariates.

The main model is a **conditional normalizing flow (cNF)** trained on scarce survey samples and conditioned on exogenous place-level covariates such as accessibility, wealth proxies, and meteoclimatic indicators. The generated household-level data are evaluated against held-out survey data and compared with oversampling, region-only conditioning, Bayesian regression, and tabular generative-model baselines.

---

## Main idea

Many household surveys are large enough for national-level summaries but too sparse to reliably describe local variation. This project tests whether generative models can help in that setting.

The paper's core finding is:

> Generative models are useful for inference only when they are given informative external signal.  
> With rich contextual covariates, cNF improves sub-national distributional estimates beyond oversampling.  
> Without such context, the model behaves much more like structured resampling.

The workflow simulates data scarcity by selecting a small number of primary sampling units (PSUs) per sub-national region, trains conditional generators on the sparse sample, generates synthetic household records for under-represented locations, and evaluates whether the generated data better recovers the full survey distribution.

---

## Repository structure

```text
.
├── data_descriptions/
├── Q0/
├── Q1/
├── pipeline_figure/
├── project_tree.txt
└── README.md
```

### `data_descriptions/`

Contains notebooks for producing dataset description tables.

```text
data_descriptions/
└── create_tables.ipynb
```

Use this section to document the source surveys, target variables, contextual covariates, and country/dataset metadata used in the experiments.

---

## `Q1/`: main scarcity and refinement experiments

`Q1` contains the main experimental pipeline corresponding to the paper's primary research question: whether context-conditioned generation improves sub-national inference under survey sparsity.

### Main experiment folders

```text
Q1/
├── experiments/
├── experiments_minimal/
├── biased/
├── bayesian/
├── feature_importance/
├── R2/
├── across_experiments_eval/
├── across_experiments_eval_minimal/
├── across_experiments_sensitivity/
└── decision_supp_model/
```

### `Q1/experiments/`

Main cNF experiments across datasets.

Each dataset folder contains a `train.py` script and a `results/` directory with generated pools, scaled data, train subsets, scalers, evaluation notebooks, and diagnostic figures.

Datasets visible in the project tree include:

```text
eth_micron
lka_micron
lka_vam
moz_vam
nga_mics
nga_micron
yem_mvam
zwe_mics
```

Typical structure:

```text
Q1/experiments/<dataset>/
├── full.csv
├── train.py
└── results/
    ├── evaluate.ipynb
    ├── train_<n>_scaled/
    │   └── seed_<k>_scaled/
    │       ├── train_subset_<n>_seed<k>.csv
    │       ├── train_subset_<n>_seed<k>_scaled.csv
    │       ├── generated_pool_<n>_seed<k>.csv
    │       ├── generated_pool_<n>_seed<k>_scaled.csv
    │       ├── full_<dataset>_scaled_train<n>_seed<k>.csv
    │       └── x_scaler_iqr_train<n>_seed<k>.json
    └── evaluation/
        ├── improvement_maps.pdf
        ├── err_comparison.pdf
        ├── distributions.pdf
        ├── variability.pdf
        └── eval_adm1_seed_metrics_train_<n>.csv
```

The `train_<n>_scaled` folders correspond to different training sample sizes, where `n` indicates the number of PSUs or sampling units per region used in the scarcity simulation.

### `Q1/experiments_minimal/`

A reduced version of the main experiment pipeline, likely intended for lighter runs or cleaner reproducibility.

Use this if you want a smaller reproducible pipeline before running the full experiment set.

### `Q1/biased/`

Experiments where the training sample is intentionally biased, for example toward urban households.

This supports the paper's robustness analysis: cNF can still outperform oversampling in relative terms under biased sampling, but absolute error can remain substantial when the training sample is structurally unrepresentative.

### `Q1/across_experiments_eval/`

Aggregated evaluation across datasets, variables, regions, and seeds.

Key files include:

```text
metrics_all_experiments.csv
metrics_all_experiments.parquet
metrics_all_experiments_aggregated_over_seeds.csv
metrics_adm1_avg_over_variables.csv
adm1_geometry_lookup.csv
adm1_geometry_lookup.parquet
```

This folder also contains figures used to summarize improvements across experiments, including violin plots, ADM1-level plots, Wasserstein/EMD diagnostics, and improvement figures.

### `Q1/across_experiments_eval_minimal/`

Minimal aggregated outputs for the reduced experiment pipeline.

### `Q1/across_experiments_sensitivity/`

Sensitivity analysis for training sample size.

The paper reports that improvements are largest under the strongest scarcity condition and decline toward zero as sample size increases. This folder contains merged sensitivity results and figures such as:

```text
merged_all_experiments.csv
sensitivity.ipynb
sensitivity_new.ipynb
sensitivity_imp_emd_grid.*
sensitivity_imp_abs_grid.*
sensitivity_emd_contributions_grid.*
```

### `Q1/feature_importance/`

Feature contribution analysis for contextual covariates.

This section supports the Shapley-style analysis of which contextual variables contribute to improvement over oversampling. The paper finds that contextual covariates usually help, with wealth- and accessibility-related signals often more consistently useful than meteoclimatic ones, though usefulness remains dataset-dependent.

### `Q1/R2/`

Analysis of the predictive power of contextual covariates.

This folder contains leave-one-out/random-forest style analyses linking the predictive informativeness of contextual covariates to cNF performance.

Representative outputs include:

```text
metrics_with_context.csv
r2_vs_absmeanerr.pdf
r2_vs_improvement.png
r2_vs_improvement_per_experiment.png
loo_scatter.pdf
regression_log_rf_loo.txt
```

### `Q1/bayesian/`

Bayesian regression baseline.

This section compares cNF with a Bayesian multivariate linear regression model fitted on contextual covariates. The paper reports that cNF performs slightly but significantly better on sub-national mean estimates while also generating full household-level distributions rather than only aggregate predictions.

Typical outputs include:

```text
bayes_mu.csv
bayes_std.csv
rnvp_generated_run*.csv
eval_adm1.csv
violin_bayesian_vs_generated.*
```

### `Q1/decision_supp_model/`

Decision-support model for predicting when cNF is likely to help.

This folder contains an interpretable model trained to predict whether generative refinement improves over a national baseline based on sample size and intrinsic sub-national variability.

Key files include:

```text
Q1/decision_supp_model/
├── full.csv
├── adm1_level_summary.csv
├── create_impr_dataset.ipynb
├── create_dataset_nhh.ipynb
├── eval_tree.ipynb
├── roc_regression.*
└── tree/
    ├── train.py
    ├── metadata.json
    ├── cls_fold_test_metrics.csv
    ├── reg_fold_test_metrics.csv
    ├── cls_feature_importances.csv
    ├── reg_feature_importances.csv
    └── models/
        └── *.joblib
```

The paper reports a ROC AUC around 0.63 for the decision-support model, suggesting that sample size and intrinsic variability provide useful but incomplete signal about when cNF refinement will be beneficial.

---

## `Q0/`: synthetic-data quality benchmarks

`Q0` appears to contain supplementary benchmark analyses focused on synthetic-data quality rather than the main small-area refinement question.

```text
Q0/
├── experiments/
├── fidelity/
├── recall/
└── overall_eval/
```

### `Q0/experiments/`

Benchmark generative-model runs. The tree includes model-specific folders such as `cNF`, and likely additional baseline model folders depending on the full repository state.

Typical outputs include:

```text
synthetic_pool.csv
synthetic_pool_scaled.csv
realnvp_model.pkl
cleaned_training_data.csv
seed_config.json
run_metadata.json
```

### `Q0/fidelity/`

Fidelity metrics comparing generated data with true/full data.

Per-dataset folders contain marginal, bivariate, and multivariate fidelity notebooks and outputs:

```text
marginal_fidelity.ipynb
bivariate_fidelity.ipynb
multivariate_fidelity.ipynb
fidelity_overall.ipynb
emd_gen_true.csv
emd_gen_true_normalized.csv
joint_emd_gen_true.csv
corrdiff_gen_true.csv
merged_marginal_bivariate_joint_normalized.csv
```

### `Q0/recall/`

Recall/originality-style evaluation of generated samples.

Typical files include:

```text
recall_radius_results.csv
recall_radius_summary.csv
compute_recall_cNF.ipynb
<dataset>/recall_complete.ipynb
<dataset>/recall_gen_true_by_adm1.csv
<dataset>/recall_gen_true_overall.csv
```

### `Q0/overall_eval/`

Aggregated synthetic-data quality plots.

Representative outputs include:

```text
originality_fidelity_profiles.*
fidelity_originality_scatter.*
originality_fidelity_subplots.*
spider_fidelity_originality.*
spider_plots_science_advances.*
```

---

## `pipeline_figure/`

Contains notebooks and graphics used to produce the paper's pipeline figure.

```text
pipeline_figure/
├── nga_figures.ipynb
├── maps_output.png
├── dia.svg
├── diagramma_fixed.svg
└── pipeline.pdf
```

---

## Data

The project uses household survey datasets from humanitarian and development contexts, including VAM, MICS, mVAM, MICRON, and related country-level survey sources.

The paper evaluates eight household survey datasets across six countries. Dataset identifiers visible in the repository include:

| Identifier | Likely dataset |
|---|---|
| `eth_micron` | Ethiopia MICRON |
| `lka_micron` | Sri Lanka MICRON |
| `lka_vam` | Sri Lanka VAM |
| `moz_vam` | Mozambique VAM |
| `nga_mics` | Nigeria MICS |
| `nga_micron` | Nigeria MICRON |
| `yem_mvam` | Yemen mVAM |
| `zwe_mics` | Zimbabwe MICS |

Large preprocessed data files may not be suitable for GitHub storage. If data are not included in this repository, place the required `full.csv` files in the corresponding dataset folders before running the scripts.

---

## Methodological overview

### 1. Simulate sparse survey data

The full survey is treated as the evaluation reference. Sparse training samples are created by selecting a limited number of PSUs per ADM1 region while preserving national-level representativeness as much as possible.

### 2. Train a conditional generator

The main model is a conditional normalizing flow trained on household-level target variables and conditioned on contextual covariates.

The model learns:

```text
p(Y | X, sector)
```

where:

- `Y` = household-level target variables;
- `X` = contextual covariates for the location;
- `sector` = rural/urban/estate sector where available.

### 3. Generate synthetic records

After training, the model generates household-level records for target regions by conditioning on known contextual covariates.

### 4. Evaluate generated data

Generated data are compared against held-out full survey data using:

- **Earth Mover's Distance (EMD)** for distributional fidelity;
- **Average Mean Error (AME)** for sub-national mean estimates;
- improvement over an **oversampling baseline**;
- comparison with Bayesian regression for aggregate estimates;
- fidelity/originality/recall metrics for synthetic-data quality.

Positive improvement means:

```text
improvement = baseline_error - model_error
```

So a positive value indicates that the model outperforms the baseline.

---

## Reproducing the analyses

The repository does not currently expose a single top-level pipeline script in the project tree. The workflow is organized by experiment folders and notebooks.

A typical reproduction sequence is:

### 1. Set up the environment

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install the required packages. If a `requirements.txt` or environment file is added later, use:

```bash
pip install -r requirements.txt
```

Based on the project files and methods, expected dependencies may include:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn geopandas shapely pyarrow joblib jupyter
```

Additional model-specific dependencies may be required for:

- normalizing flows / RealNVP / `probaforms`;
- Bayesian regression;
- SDV, CTGAN, or TVAE benchmarks;
- geospatial plotting.

### 2. Prepare data

Ensure that each dataset folder contains the expected `full.csv` file, for example:

```text
Q1/experiments/nga_mics/full.csv
Q1/experiments/moz_vam/full.csv
Q1/experiments/yem_mvam/full.csv
```

### 3. Run main cNF experiments

From a dataset folder:

```bash
cd Q1/experiments/<dataset>
python train.py
```

For example:

```bash
cd Q1/experiments/nga_mics
python train.py
```

Outputs are written under:

```text
Q1/experiments/<dataset>/results/
```

### 4. Evaluate a dataset

Open the evaluation notebook:

```bash
jupyter notebook Q1/experiments/<dataset>/results/evaluate.ipynb
```

This generates or updates dataset-level evaluation files such as:

```text
improvement_maps.pdf
err_comparison.pdf
distributions.pdf
variability.pdf
eval_adm1_seed_metrics_train_<n>.csv
```

### 5. Aggregate across experiments

Use notebooks in:

```text
Q1/across_experiments_eval/
Q1/across_experiments_sensitivity/
Q1/R2/
Q1/feature_importance/
```

These produce the cross-dataset tables and figures used in the paper.

### 6. Run decision-support model

```bash
cd Q1/decision_supp_model/tree
python train.py
```

Then inspect:

```text
cls_fold_test_metrics.csv
reg_fold_test_metrics.csv
cls_feature_importances.csv
reg_feature_importances.csv
roc_regression.*
```

### 7. Run synthetic-data quality benchmarks

For fidelity/originality/recall analyses, use notebooks in:

```text
Q0/fidelity/
Q0/recall/
Q0/overall_eval/
```

---

## Key outputs

| Output | Location |
|---|---|
| Main generated pools | `Q1/experiments/<dataset>/results/train_*_scaled/seed_*_scaled/generated_pool_*.csv` |
| Train subsets | `Q1/experiments/<dataset>/results/train_*_scaled/seed_*_scaled/train_subset_*.csv` |
| Dataset-level evaluation | `Q1/experiments/<dataset>/results/evaluation/` |
| Aggregated metrics | `Q1/across_experiments_eval/metrics_all_experiments.csv` |
| Sensitivity results | `Q1/across_experiments_sensitivity/` |
| Context predictiveness | `Q1/R2/metrics_with_context.csv` |
| Feature importance | `Q1/feature_importance/` |
| Bayesian comparison | `Q1/bayesian/` |
| Decision-support model | `Q1/decision_supp_model/` |
| Fidelity metrics | `Q0/fidelity/` |
| Recall metrics | `Q0/recall/` |
| Overall synthetic-data quality plots | `Q0/overall_eval/` |
| Pipeline figure | `pipeline_figure/` |

---

## Interpretation guide

The main comparison is between:

1. **Oversampling baseline**  
   Reuses the sparse observed sample to approximate local distributions.

2. **NF with region-only conditioning**  
   Uses categorical region labels but little continuous external context.

3. **cNF with contextual covariates**  
   Uses external place-level information to condition generation.

The expected pattern is:

- cNF should improve most when contextual covariates are informative;
- gains should be largest under strong data scarcity;
- gains should decline as the training sample becomes more representative;
- cNF is most valuable when sub-national distributions differ substantially from the national distribution;
- if the training sample is structurally biased, cNF may still improve over oversampling relatively, but it cannot fully repair missing information.

---

## Important limitations

This repository implements an inference-oriented use of generative AI, not a guarantee that synthetic data create new information.

Key assumptions and limitations:

- The training sample should remain nationally representative along relevant dimensions.
- cNF can exploit contextual signal but cannot correct structural blind spots in the original sample.
- The framework assumes that relationships between contextual covariates and target variables are sufficiently stable across space.
- Current analyses focus mainly on continuous target variables.
- Many metrics are computed marginally, so multivariate dependencies may require additional checks.
- Contextual covariates may contain their own measurement error.
- Some data files may be too large or restricted for public release.

---

## Suggested repository clean-up before public release

Before publishing, consider adding:

```text
requirements.txt or environment.yml
LICENSE
CITATION.cff
data/README.md
scripts/ or src/ package structure
a top-level reproduction script
clear instructions for restricted data access
```

Also consider removing generated artifacts from Git tracking if they are large and reproducible:

```text
*.pkl
*.joblib
large generated_pool_*.csv files
large full_*_scaled*.csv files
.git/ from exported project trees
```

A useful `.gitignore` pattern could include:

```gitignore
__pycache__/
.ipynb_checkpoints/
.venv/
*.pkl
*.joblib
*.parquet
results/
```

Adapt this carefully if some result files are intentionally versioned for the paper.

---

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{sibilla2025generative,
  title = {Can generative AI recover what the sample missed? Evidence from sparse survey data},
  author = {Sibilla, Federica and Voukelatou, Vasiliki and Piovani, Duccio and Koupparis, Kyriacos and Paolotti, Daniela and Kalimeri, Kyriaki},
  year = {2025},
  note = {Manuscript / preprint}
}
```

Update the citation once the final DOI, venue, or arXiv record is available.

---

## Contact

For questions about the methodology, data, or experiments, contact the paper authors or repository maintainer.

