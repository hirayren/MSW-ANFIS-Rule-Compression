# MSW calorific-value prediction: interpretable Sugeno fuzzy baseline and rule compression

Student: Hira (surname and student ID to be added by student).

## Reference
Mondal, S. and Rafizul, I. M., *Predicting Calorific Value through Proximate Analysis of Municipal Solid Waste Using Soft Computing System*, 2024 preprint, DOI: 10.21203/rs.3.rs-5297925/v1.

## Problem
Predict municipal solid waste calorific value (kJ/g) from moisture, volatile matter, fixed carbon and ash percentages.

## Fuzzy system
Four inputs, first-order Takagi–Sugeno rules, Gaussian membership functions, product rule activation, normalized firing strengths, linear consequents and weighted-sum output. A transparent approximation to the paper's ANFIS, NOT an exact MATLAB reproduction. Paper reports 81 rules; baseline uses 3 Gaussian MFs/input = 81 rules. Proposed compact variant uses 2 MFs/input = 16 rules. Both use identical ridge and optimization procedure. Gaussian widths are fixed; centers optimized with L-BFGS-B, consequents solved by ridge least squares. Not the paper's backpropagation 1000-epoch procedure.

## Dataset
`data/table2_transcribed.csv`: 37 visible rows from Table 2 of supplied paper, 28 labeled measured and 9 from other studies. Original paper also refers to 46 datasets / 230 data values, an inconsistency requiring disclosure. Manually transcribed from the PDF; user MUST check the values before citing experimental results. No fabricated rows.

## Install
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python -m pytest tests -q
python src/run_experiments.py --seed 42
```
Results created after execution: `results/metrics.csv`, `results/test_predictions.csv`, `results/prediction_comparison.png`, `results/run_metadata.json`.

## Reproduction / baseline
A faithful architectural *approximation* of the paper's four-input Gaussian Sugeno neuro-fuzzy sub-system. The exact trained MATLAB model and 81 learned rules were not provided. Do not present this as a numerical reproduction of the reported R².

## Proposed extension
Rule-base compression: 81 rules to 16 rules, same dataset, split, optimization and regularization. Hypothesis: smaller model reduces computational cost and overfitting; accuracy is an empirical question, not an established result.

## Experiments and results
Experiments are NOT run as part of the provided package. Run the script and insert measured results into the report/presentation. Do not copy numbers from the paper as your own results. See `experiments/EXPERIMENT_PLAN.txt`.

## Limitations
Small data; 37 visible records; unknown original train/test indices; incomplete hyperparameter details; manually transcribed values; fixed MF widths; single holdout; reported paper performance may not generalize; variables in Table 2 are not constrained to sum to 100% (basis may differ).

## Sources
Paper above; Jang (1993), ANFIS, IEEE TSMC 23(3):665–685, DOI:10.1109/21.256541; Zadeh (1965), Fuzzy sets, Information and Control 8(3):338–353, DOI:10.1016/S0019-9958(65)90241-X.

## Generative AI disclosure
ChatGPT assisted in drafting code, documentation, presentation text, and report. Student is responsible for checking, running, understanding and reporting actual experiments.
