# News Diffusion Code

Clean source-code repository for the Twitter15/16 news-diffusion thesis
experiments.

## Thesis Details

- Student: Κωστής Γεώργιος
- Student ID: Π2020030
- Supervisor: Κανάβος Ανδρέας
- Academic year: 2025–2026
- Thesis title: Διάχυση Ειδησεογραφικής Πληροφορίας στα Κοινωνικά Δίκτυα

This GitHub repository intentionally contains only code and minimal execution
metadata. Datasets, generated CSV/JSON artifacts, reports, thesis text, figures,
PDFs and raw data are not committed.

## Contents

- `workflow/`: Python workflow scripts
- `run_all.sh`: convenience runner for the research pipeline
- `requirements.txt`: Python runtime dependency list

## Run

The full local pipeline expects the thesis data folders to exist locally
(`data_clean/`, and optionally `data_raw/` for raw cleaning). Those folders are
ignored by git.

```bash
python3 -m pip install -r requirements.txt
./run_all.sh
```

## Workflow Order

`run_all.sh` executes these scripts in order:

1. `workflow/build_research_assets.py`
2. `workflow/run_advanced_research.py`
3. `workflow/generate_descriptive_tables.py`
4. `workflow/run_stat_tests.py`
5. `workflow/integrate_supplementary_models.py`
6. `workflow/generate_thesis_figures.py`
