"""Integrate the supplementary logreg / random-forest results from the
official zip into our project as a comparable analysis layer.

The zip contains pre-computed metrics under ``research/supplementary_models/ml``
(taskA_metrics.csv, taskB_metrics.csv, classification reports, confusion
matrices). We do not retrain those models here (sklearn is not available in
the offline build environment); we just consolidate the metrics into a single
``supplementary_summary.json`` and a Markdown report so the rest of the
pipeline and the LaTeX text can cite consistent numbers.

Run:
    python3 workflow/integrate_supplementary_models.py
"""
import csv
import json
from datetime import datetime
from pathlib import Path


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def load_metrics_csv(path):
    """Load a supplementary ``task{A,B}_metrics.csv`` file as typed dicts.

    Args:
        path: Path to the CSV exported with the official zip.

    Returns:
        list of dicts with ``task``, ``model``, ``feature_set``, ``accuracy``
        (float), ``macro_f1`` (float), ``n_train`` (int) and ``n_test`` (int).
    """
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "task": r["task"],
                    "model": r["model"],
                    "feature_set": r["feature_set"],
                    "accuracy": to_float(r["accuracy"]),
                    "macro_f1": to_float(r["macro_f1"]),
                    "n_train": int(r["n_train"]),
                    "n_test": int(r["n_test"]),
                }
            )
    return rows


def parse_classification_report(path):
    """Parse a sklearn classification_report text file into per-class dicts.

    Expected layout::

                precision   recall   f1-score   support
        <label> <p>         <r>      <f1>       <s>
        ...
        accuracy                     <a>        <total>
        macro avg <p>       <r>      <f1>       <total>
        weighted avg <p>    <r>      <f1>       <total>
    """
    per_class = []
    overall = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line.strip() or line.lstrip().startswith("precision"):
                continue
            tokens = line.split()
            if not tokens:
                continue
            # detect overall rows
            if tokens[0] == "accuracy":
                overall["accuracy"] = to_float(tokens[1])
            elif tokens[0] == "macro" and len(tokens) >= 5 and tokens[1] == "avg":
                overall["macro_avg"] = {
                    "precision": to_float(tokens[2]),
                    "recall": to_float(tokens[3]),
                    "f1": to_float(tokens[4]),
                    "support": int(tokens[5]) if len(tokens) > 5 else 0,
                }
            elif tokens[0] == "weighted" and len(tokens) >= 5 and tokens[1] == "avg":
                overall["weighted_avg"] = {
                    "precision": to_float(tokens[2]),
                    "recall": to_float(tokens[3]),
                    "f1": to_float(tokens[4]),
                    "support": int(tokens[5]) if len(tokens) > 5 else 0,
                }
            else:
                # per-class row: label may be multi-token but in this dataset
                # it is a single token (false / true / unverified / non-rumor /
                # rumor). For safety we treat all but last 4 tokens as label.
                if len(tokens) < 5:
                    continue
                label = " ".join(tokens[:-4])
                per_class.append(
                    {
                        "label": label,
                        "precision": to_float(tokens[-4]),
                        "recall": to_float(tokens[-3]),
                        "f1": to_float(tokens[-2]),
                        "support": int(tokens[-1]),
                    }
                )
    return {"per_class": per_class, "overall": overall}


def best_by_macro_f1(rows):
    """Pick the row with the highest macro-F1, breaking ties by accuracy.

    Args:
        rows: Sequence of metric dicts as returned by :func:`load_metrics_csv`.

    Returns:
        The selected row, or ``None`` if ``rows`` is empty.
    """
    if not rows:
        return None
    return max(rows, key=lambda r: (r["macro_f1"], r["accuracy"]))


def build():
    """Consolidate supplementary metrics into a JSON summary and Markdown report.

    Reads the pre-computed logreg / random-forest results from
    ``research/supplementary_models/ml/`` and writes
    ``research/supplementary_models/supplementary_summary.json`` plus
    ``docs/supplementary_models_report.md``.

    Raises:
        SystemExit: If the supplementary results directory is missing.
    """
    project_root = Path(__file__).resolve().parents[1]
    supp_root = project_root / "research" / "supplementary_models" / "ml"
    docs_root = project_root / "docs"
    docs_root.mkdir(parents=True, exist_ok=True)

    if not supp_root.exists():
        raise SystemExit(
            f"Supplementary results not found at {supp_root}. "
            "Copy them from the official zip first."
        )

    task_a = load_metrics_csv(supp_root / "taskA_metrics.csv")
    task_b = load_metrics_csv(supp_root / "taskB_metrics.csv")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Pre-computed sklearn logreg / random-forest results from the "
                  "official zip (split 1846/462). Reproduced verbatim — not "
                  "retrained in this run because sklearn is unavailable in the "
                  "offline environment.",
        "splits_used": {"n_train": task_a[0]["n_train"], "n_test": task_a[0]["n_test"]},
        "taskA_metrics": task_a,
        "taskB_metrics": task_b,
        "best_taskA": best_by_macro_f1(task_a),
        "best_taskB": best_by_macro_f1(task_b),
        "per_class_best": {},
    }

    # Resolve per-class metrics for the best models in each task.
    for task_name, best in (("taskA", summary["best_taskA"]), ("taskB", summary["best_taskB"])):
        report_path = supp_root / (
            f"{best['task']}__{best['model']}__{best['feature_set']}__classification_report.txt"
        )
        if report_path.exists():
            summary["per_class_best"][task_name] = parse_classification_report(report_path)

    out_json = supp_root.parent / "supplementary_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Build a Markdown digest for documentation.
    lines = []
    lines.append("# Supplementary Models Report (logreg, RandomForest)")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Source")
    lines.append("- Pre-computed metrics from the official zip "
                 "(`results_research/ml/`).")
    lines.append(f"- Train / test split: {summary['splits_used']['n_train']} / "
                 f"{summary['splits_used']['n_test']}.")
    lines.append("")
    lines.append("## Task A (4-class)")
    for r in task_a:
        lines.append(f"- {r['model']:>6s} + {r['feature_set']:<17s} | acc={r['accuracy']:.4f} | macro-F1={r['macro_f1']:.4f}")
    best_a = summary["best_taskA"]
    lines.append("")
    lines.append(f"**Best Task A**: {best_a['model']} + {best_a['feature_set']} "
                 f"(acc={best_a['accuracy']:.4f}, macro-F1={best_a['macro_f1']:.4f})")
    lines.append("")
    lines.append("## Task B (binary)")
    for r in task_b:
        lines.append(f"- {r['model']:>6s} + {r['feature_set']:<17s} | acc={r['accuracy']:.4f} | macro-F1={r['macro_f1']:.4f}")
    best_b = summary["best_taskB"]
    lines.append("")
    lines.append(f"**Best Task B**: {best_b['model']} + {best_b['feature_set']} "
                 f"(acc={best_b['accuracy']:.4f}, macro-F1={best_b['macro_f1']:.4f})")
    lines.append("")
    lines.append("## Comparison with the primary KNN/Naive-Bayes pipeline")
    lines.append("- The primary pipeline uses event-level stratified splits with "
                 "1614 / 346 / 348 (Task A) and 1615 / 346 / 347 (Task B) and "
                 "five models (majority, nearest centroid, gaussian NB, KNN-5, KNN-11).")
    lines.append("- The supplementary pipeline reported here uses an "
                 "alternative 1846 / 462 split and two models (logreg, RF) on a "
                 "smaller feature taxonomy (`diffusion_struct`, "
                 "`diffusion_early`, `combined`).")
    lines.append("- Cross-method readings should therefore be interpreted as "
                 "*directional* corroboration rather than strict comparison: a "
                 "different split changes both training size and the test "
                 "marginal class distribution.")
    lines.append("")

    out_md = docs_root / "supplementary_models_report.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    build()
