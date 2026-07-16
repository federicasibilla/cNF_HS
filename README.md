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
├── Q0/
├── Q1/
└── README.md
```

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
├── train.py
└── results/
    ├──sensitivity.ipynb
    └──evaluate.ipynb
```

### `Q1/experiments_minimal/`

A reduced version of the main experiment pipeline, for running categorical region-label conditioned experiments (NF).

### `Q1/experiments_sectoronly/`

A reduced version of the main experiment pipeline, for running categorical region-label and sector label conditioned experiments (NF+sector).


### `Q1/biased/`

Experiments where the training sample is intentionally biased, for example toward urban households.

This supports the paper's robustness analysis: cNF can still outperform oversampling in relative terms under biased sampling, but absolute error can remain substantial when the training sample is structurally unrepresentative.

### `Q1/across_experiments_eval/`

Aggregated evaluation across datasets, variables, regions, and seeds.

Key files include:

```text
cross_eval.ipynb
improvement.ipynb
variability.ipynb
```

### `Q1/across_experiments_eval_minimal/`

Aggregated outputs for the reduced experiment pipeline.

### `Q1/across_experiments_eval_sectoronly/`

Aggregated outputs for the reduced experiment pipeline with only sector conditioning.

### `Q1/across_experiments_sensitivity/`

Sensitivity analysis for training sample size.

The paper reports that improvements are largest under the strongest scarcity condition and decline toward zero as sample size increases. This folder contains merged sensitivity results and figures such as:

```text
sensitivity.ipynb
sensitivity_new.ipynb
```

### `Q1/feature_importance/`

Feature contribution analysis for contextual covariates.

This section supports the Shapley-style analysis of which contextual variables contribute to improvement over oversampling. 

### `Q1/R2/`

Analysis of the predictive power of contextual covariates.

This folder contains leave-one-out/random-forest style analyses linking the predictive informativeness of contextual covariates to cNF performance.

### `Q1/bayesian/`

Bayesian regression baseline.

This section compares cNF with a Bayesian multivariate linear regression model fitted on contextual covariates. 


### `Q1/decision_supp_model/`

Decision-support model for predicting when cNF is likely to help.

This folder contains an interpretable model trained to predict whether generative refinement improves over a national baseline based on sample size and intrinsic sub-national variability.

---

## `Q0/`: synthetic-data quality benchmarks

`Q0` contains supplementary benchmark analyses focused on synthetic-data quality rather than the main small-area refinement question.

```text
Q0/
├── experiments/
├── fidelity/
├── recall/
└── overall_eval/
```

### `Q0/experiments/`

Benchmark generative-model runs. The tree includes model-specific folders such: `cNF`, `TVAE`, `CTGAN`


### `Q0/fidelity/`

Fidelity metrics comparing generated data with true/full data.


### `Q0/recall/`

Recall/originality-style evaluation of generated samples.


### `Q0/overall_eval/`

Aggregated synthetic-data quality plots.


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

Large preprocessed data files are not suitable for GitHub storage. The required `full.csv` files are available at ADD ONCE WE DECIDE.

---

[![DOI](https://zenodo.org/records/21391841)]
