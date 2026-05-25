# News Diffusion Code

Clean source-code repository for the Twitter15/16 news-diffusion thesis
experiments.

## Thesis Details

- Student: Κωστής Γεώργιος
- Student ID: Π2020030
- Supervisor: Κανάβος Ανδρέας
- Academic year: 2025–2026
- Thesis title: Διάχυση Ειδησεογραφικής Πληροφορίας στα Κοινωνικά Δίκτυα

This GitHub repository contains the workflow code plus the cleaned input data
needed to run the thesis pipeline locally. Generated CSV/JSON artifacts,
reports, thesis text, figures, PDFs and raw data are not committed.

## Data Availability

The cleaned input data are uploaded to GitHub. The two large cleaned CSV files
are stored with Git LFS:

data_clean/nodes_clean.csv
data_clean/edges_clean.csv

The smaller cleaned files are regular git files:

```text
data_clean/events_clean.csv
data_clean/cleaning_summary.json
```

The repository also includes the precomputed supplementary logreg/random-forest
metrics under:

```text
research/supplementary_models/ml/
```

Raw Twitter15/16 files are **not** uploaded. They are only needed if you want to
rerun raw-data cleaning:

```text
data_raw/Twitter15_16_dataset/
```

## Contents

- `workflow/`: Python workflow scripts
- `run_all.sh`: convenience runner for the research pipeline
- `requirements.txt`: Python runtime dependency list

## Setup On Another Computer

Clone the repository:

```bash
git lfs install
git clone https://github.com/kwsthssss/ptyxiaki.git
cd ptyxiaki
git lfs pull
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

After cloning, the required local structure should already be present:

```text
ptyxiaki/
  data_clean/
    events_clean.csv
    nodes_clean.csv
    edges_clean.csv
  workflow/
  run_all.sh
  requirements.txt
```

## Run The Full Pipeline

```bash
./run_all.sh
```

This regenerates:

```text
research/
docs/
thesis_lualatex_project/figures/
```

## Workflow Order

`run_all.sh` executes these scripts in order:

1. `workflow/build_research_assets.py`
2. `workflow/run_advanced_research.py`
3. `workflow/generate_descriptive_tables.py`
4. `workflow/run_stat_tests.py`
5. `workflow/integrate_supplementary_models.py`
6. `workflow/generate_thesis_figures.py`
