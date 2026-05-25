# Cleaned Data

This folder contains the cleaned input data required by the reproducibility
pipeline.

The two large cleaned tables are stored with Git LFS:

- `nodes_clean.csv`
- `edges_clean.csv`

After cloning, run `git lfs pull` if those files appear as small pointer files.
Then run the pipeline from the repository root:

```bash
python3 -m pip install -r requirements.txt
./run_all.sh
```
