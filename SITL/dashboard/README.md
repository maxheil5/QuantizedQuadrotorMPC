# SITL Dashboard

This folder contains a minimal Streamlit dashboard for reviewing SITL paper figures from `results/sitl/<run>/runtime_log.csv`.

## Launch

From `SITL/`:

```bash
bash ./scripts/run_dashboard.sh
```

The app will:

- discover available run folders under `results/sitl/`
- show them in a dropdown
- load the selected run's `runtime_log.csv`
- optionally read `run_metadata.json` if present
- render an interactive 3D trajectory plus MATLAB-style error, state, control, and timing plots

## Expected Inputs

Each selected run folder should contain:

- `runtime_log.csv`
- optionally `run_metadata.json`

The dashboard is read-only and does not modify SITL results.
