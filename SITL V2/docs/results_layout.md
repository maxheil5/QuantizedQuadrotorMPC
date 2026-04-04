# Results Layout

All V2 outputs should live under `SITL V2/results`.

## Directory structure

```text
results/
  baselines/
    linear_mpc/<scenario>/<run_id>/
    nonlinear_mpc/<scenario>/<run_id>/
  learned/
    unquantized/<scenario>/<run_id>/
    quantized/wl_<word_length>/N_<realizations>/<scenario>/<run_id>/
  summary/
```

## Per-run files

Each run directory should contain:

- `config.json`
- `metrics.json`
- `trajectory.csv`
- `control.csv`
- `timing.csv`
- `environment.json`
- `notes.txt`
- `figures/`

## Summary files

`results/summary` should contain:

- `run_index.csv`
- `metrics_long.csv`
- `metrics_wide.csv`
- `experiment_manifest.json`

## Naming rules

- Use `word length`, not `bit width`, everywhere.
- Include the controller family in the path instead of overloading run names.
- Include `N_<realizations>` in quantized experiment paths so scaling stages are explicit.

## Intended use

- Raw CSV/JSON files are the source of truth.
- Figures should always be generated from saved files.
- Summary CSVs are for quick plotting and paper tables.

