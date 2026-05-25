# Windows Run Guide

This guide explains how to clone and run the project on a Windows computer.

## 1. Install Required Software

Install these first:

- Python 3.10 or newer: https://www.python.org/downloads/
- Git for Windows: https://git-scm.com/download/win
- Git LFS: https://git-lfs.com/

During Python installation, enable:

```text
Add python.exe to PATH
```

After installing Git LFS, open PowerShell and run:

```powershell
git lfs install
```

## 2. Clone The Repository

Open PowerShell and run:

```powershell
git clone https://github.com/kwsthssss/ptyxiaki.git
cd ptyxiaki
git lfs pull
```

Check that the large data files were downloaded correctly:

```powershell
dir data_clean
```

Expected important files:

```text
events_clean.csv
nodes_clean.csv
edges_clean.csv
cleaning_summary.json
```

If `nodes_clean.csv` or `edges_clean.csv` are only a few KB, Git LFS did not
download the real files. Run again:

```powershell
git lfs pull
```

## 3. Create Python Environment

In the repository folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.venv\Scripts\Activate.ps1
```

## 4. Run The Pipeline

Recommended Windows method: run each workflow script with Python:

```powershell
python workflow\build_research_assets.py
python workflow\run_advanced_research.py
python workflow\generate_descriptive_tables.py
python workflow\run_stat_tests.py
python workflow\integrate_supplementary_models.py
python workflow\generate_thesis_figures.py
```

Alternative method: use Git Bash and run:

```bash
./run_all.sh
```

## 5. Expected Outputs

After a successful run, these folders/files are regenerated:

```text
research/
docs/
thesis_lualatex_project/figures/
```

The expected best results are:

```text
Task A: knn_11 + combined, accuracy 0.482759, macro-F1 0.469967
Task B: knn_11 + structural, accuracy 0.847262, macro-F1 0.791033
```

## 6. Common Problems

### Git LFS files did not download

Run:

```powershell
git lfs install
git lfs pull
```

### Python is not recognized

Reinstall Python and enable `Add python.exe to PATH`, or try:

```powershell
py -m venv .venv
```

### Missing Pillow

Run:

```powershell
python -m pip install -r requirements.txt
```

### `./run_all.sh` does not work in PowerShell

That is normal. Either run the Python scripts one by one in PowerShell, or open
Git Bash and run:

```bash
./run_all.sh
```
