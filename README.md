# MSW Calorific Value Prediction — Gaussian Sugeno Rule-Base Compression

## Student information

- **Student:** Hira Nur BAYTOK
- **Student ID:** 1306200001
- **University:** İstanbul University-Cerrahpaşa, Faculty of Engineering
- **Department:** Computer Engineering
- **Course:** BIMU4092 Bulanık Mantık
- **Academic project:** Fuzzy Logic term project, November 2026

A reproducible course project examining the accuracy–complexity trade-off of a Gaussian Takagi–Sugeno neuro-fuzzy model for predicting municipal solid waste (MSW) calorific value.

## Problem description

Municipal solid waste has variable composition and energy content. Laboratory calorimetry measures calorific value directly but requires equipment and sample preparation. This project investigates regression-based estimation from four proximate-analysis measurements: moisture content (MC), volatile content (VC), fixed carbon (FC), and ash content (A). The predicted output is calorific value in kJ/g.

## Research question

Can reducing the fuzzy rule base from **81 rules to 16 rules** substantially reduce training time without a large deterioration in prediction accuracy?

## Reference study and reproduction scope

S. Mondal and I. M. Rafizul, *Predicting Calorific Value through Proximate Analysis of Municipal Solid Waste Using Soft Computing System* (2024 preprint), DOI: [10.21203/rs.3.rs-5297925/v1](https://doi.org/10.21203/rs.3.rs-5297925/v1).

This implementation is an **architectural approximation of the fuzzy inference subsystem**, not an exact numerical reproduction of the paper's MATLAB ANFIS. The original trained parameters, exact split, and full training configuration were not available. In particular, the present code uses ridge-regularized least squares and L-BFGS-B optimization rather than the paper's reported 1000-iteration backpropagation training.

## Dataset

`data/table2_transcribed.csv` contains **37 visible observations** transcribed from Table 2 of the reference paper: 28 identified as measured and nine sourced from other studies. The original article also discusses 46 datasets / 230 values, which is not consistent with the 37 visible table rows. This discrepancy is documented rather than resolved by inventing data.

Inputs: moisture content (MC), volatile content (VC), fixed carbon (FC), and ash content (A). Target: calorific value in kJ/g. The data were checked against the visible Table 2; they should not be treated as a large or independently validated benchmark.

## Method

Both variants use four inputs, Gaussian membership functions, product-based rule firing, normalized firing strengths, first-order Takagi–Sugeno linear consequents, and weighted-sum output. Consequent coefficients are estimated using ridge-regularized least squares; Gaussian centers are optimized using L-BFGS-B while widths remain fixed.

| Variant | Membership functions per input | Number of rules |
|---|---:|---:|
| Baseline | 3 | 81 |
| Compact proposal | 2 | 16 |

The **proposed extension** reduces the rule count by **80.25%**. Both variants use the same data split, preprocessing, and evaluation protocol for each run.

## Fuzzy system architecture

The implemented first-order Takagi–Sugeno system follows these computational stages:

1. **Fuzzification:** calculate Gaussian membership degrees for each of the four normalized input variables.
2. **Rule firing:** calculate each rule's activation as the product of its four membership degrees.
3. **Normalization:** divide each activation by the sum of all rule activations.
4. **Consequent evaluation:** calculate a first-order linear function of the four inputs for each rule.
5. **Aggregation:** calculate the final crisp output as the weighted sum of rule consequents, then convert it to kJ/g.

With three Gaussian membership functions per input, the Cartesian rule base has 3^4 = 81 rules; with two functions per input it has 2^4 = 16 rules. There are 405 and 80 linear consequent coefficients, respectively (five per rule). This is an interpretable Sugeno-style architectural approximation, not the original paper's trained MATLAB model.

## Reproduction experiments

The reproduction baseline retains the reference paper's four inputs, Gaussian memberships, first-order Sugeno rule structure, and reported 81-rule count. It does **not** reproduce the original learned parameter values, subtractive-clustering configuration, or 1000-iteration backpropagation training. Instead, this Python implementation uses a documented shared optimization procedure for both the baseline and compact variant. The original paper's published performance metrics must not be confused with the measurements obtained here.

Both variants were evaluated on the same five seeded 80/20 splits. Each seed produces a pair of comparable RMSE, MAE, R², and training-time measurements. The test set is held out from fitting. Detailed outputs and the consolidated CSV are included in the repository.

## Requirements and setup

Experiments were conducted on Windows with **Python 3.11.9** and an **Intel Core i7-13620H** CPU. `requirements.txt` specifies project dependencies; `environment.txt` records the Python packages installed in the experiment environment.

```bash
git clone https://github.com/hirayren/MSW-ANFIS-Rule-Compression.git
cd MSW-ANFIS-Rule-Compression
python -m pip install -r requirements.txt
python -m pytest tests -q
```

## Running experiments

From the project root:

```bash
python src/run_experiments.py --seed 42
```

The script writes outputs to `results/` by default: `metrics.csv`, `test_predictions.csv`, `prediction_comparison.png`, and `run_metadata.json`. Running it again may overwrite those root-level outputs. The archived five-seed outputs are preserved separately under `results/results_seed7/`, `results/results_seed21/`, `results/results_seed42/`, `results/results_seed73/`, and `results/results_seed99/`.

The experiments used seeds **7, 21, 42, 73, and 99**, with an approximately 80/20 training/test split (29 training and 8 test observations per run). Each pair of models was evaluated on the same split. `all_experiment_results.csv` aggregates the individual runs.

## Experimental results

The table shows mean ± sample standard deviation across five runs. RMSE and MAE are in kJ/g; training time is in seconds.

| Metric | 81-rule baseline | 16-rule compact model |
|---|---:|---:|
| RMSE ↓ | 7.0611 ± 2.2294 | 6.9864 ± 2.4118 |
| MAE ↓ | 4.9642 ± 0.9762 | 5.2382 ± 1.3779 |
| R² ↑ | 0.1928 ± 0.1515 | 0.2213 ± 0.1500 |
| Training time ↓ | 4.9340 ± 1.5343 | 0.0918 ± 0.0533 |
| Rule count ↓ | 81 | 16 |

The compact model trained approximately **53.7× faster** on average. It had slightly lower mean RMSE but higher mean MAE. Both variants had low R² values and performance varied across data splits. These findings **do not establish a general accuracy advantage** for the compact model.

The seed-42 prediction comparison is available at `results/results_seed42/prediction_comparison.png`. One high-calorific-value test observation produced a large prediction error for both models.

## Repository contents

- `src/model.py` — Gaussian Sugeno model implementation
- `src/run_experiments.py` — training, evaluation, and output generation
- `data/table2_transcribed.csv` — transcribed source-table observations
- `tests/` — automated tests
- `experiments/EXPERIMENT_PLAN.txt` — experiment plan
- `results/results_seed*/` — archived metrics, predictions, plots, and metadata
- `all_experiment_results.csv` — consolidated results
- `report/` — Turkish technical report
- `presentation/` — presentation materials
- `requirements.txt`, `environment.txt` — dependencies and environment snapshot

## Limitations

The dataset is very small (37 records), with only eight test records per split. Source records are heterogeneous, and the reference paper's original split and learned model parameters are unavailable. Five holdout splits provide limited evidence; no independent external validation or source-stratified evaluation was conducted. Gaussian widths were fixed, and the current training approach differs from the source article. Timing results are hardware- and environment-dependent.

## References and further reading

- L. A. Zadeh, “Fuzzy sets,” *Information and Control*, 8(3), 338–353, 1965. DOI: 10.1016/S0019-9958(65)90241-X.
- J.-S. R. Jang, “ANFIS: Adaptive-network-based fuzzy inference system,” *IEEE Transactions on Systems, Man, and Cybernetics*, 23(3), 665–685, 1993. DOI: 10.1109/21.256541.

## Generative AI disclosure

OpenAI ChatGPT assisted with code, documentation, report, and presentation drafting. The experiments were executed locally, and the observed results are reported separately from the reference paper's claims. The project does not claim exact reproduction of the original ANFIS numerical results.
