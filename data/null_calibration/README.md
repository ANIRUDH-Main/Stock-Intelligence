# Null Calibration — Reproducibility Archive

This directory contains the reproducibility artefacts for the synthetic
no-signal experiment described in Section V and Appendix A of the research
paper.

## Files

- `gbm_paths.csv` — the generated synthetic OHLCV paths.
- `null_results.csv` — per-path test-set metrics for all five regressors,
  persistence, and the selected best-of-five model.
- `run_manifest.json` — simulation parameters, seeds, source of starting
  prices, and summary statistics.

## Re-running the experiment

From the repository root:

```bash
python experiments/null_calibration.py
```

The script reads audited starting prices from:

```text
models/model_metadata.json
```

and writes the three files in this directory.

## Protocol

- 200 synthetic paths
- 251 daily observations per path
- 201 usable rows after feature engineering
- 160 training rows / 41 test rows
- chronological 80/20 split
- annualized volatility sampled from `{0.20, 0.35, 0.50, 0.80}`
- annualized drift sampled from `N(0.08, 0.25^2)`
- simulation seed: `7`
- permutation seed reserved by the paper: `0`

The feature set and model hyperparameters follow Appendix A of the paper.

## Important provenance note

The synthetic series are **generated data**, not an external dataset. The
GitHub repository is therefore an archive of the generation procedure and
outputs, not the original source of the observations.

If the historical Section V numbers were produced by an earlier unarchived
run and cannot be reproduced exactly from the original code/seed, do not
present the newly generated values as the historical values. In that case,
rerun the experiment, replace Table III with the reproducible results, and
update the paper's numerical discussion accordingly.

## Citation

Cite the repository as the implementation/archive reference in the paper:

[25] A. Chaurasia, V. Kohli, P. Kaushik, and H. Bhatia, *Stock Intelligence
& Social Sentiment Dashboard*, GitHub repository, 2026. [Online]. Available:
https://github.com/ANIRUDH-Main/Stock-Intelligence
