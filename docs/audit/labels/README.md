# Audit label sheets

Each rater's export from the project site's audit page (`audit.html`): `labels_<name>.csv`, one row per item ×
criterion (`item_id`, `criterion`, `label`, `comment`). Commit each sheet here as exported. `hcai reproduce`
computes Cohen's κ once two sheets are here (`data/runs/core-v1/analysis/audit/agreement*.csv`).

Rate on your own, and do not open `data/runs/core-v1/analysis/audit/audit_key.csv`: it holds the judge's verdicts.
