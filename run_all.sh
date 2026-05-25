#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python3}"

if ! "$PYTHON_BIN" - <<'PY'
from PIL import Image
print("Pillow OK")
PY
then
  echo "Missing Python dependency. Run: $PYTHON_BIN -m pip install -r requirements.txt" >&2
  exit 1
fi

scripts=(
  "workflow/build_research_assets.py"
  "workflow/run_advanced_research.py"
  "workflow/generate_descriptive_tables.py"
  "workflow/run_stat_tests.py"
  "workflow/integrate_supplementary_models.py"
  "workflow/generate_thesis_figures.py"
)

for script in "${scripts[@]}"; do
  echo
  echo "==> $script"
  "$PYTHON_BIN" "$script"
done

echo
echo "Done. Regenerated research artifacts, reports, supplementary summaries, and thesis figures."
