"""Generate descriptive-statistics tables for the thesis dataset.

Reads the modelling-ready dataset and emits per-feature summaries
(``n``, mean, std, median, p90, p95) globally, by 4-class label, and by
(split, label).

Inputs:
    - ``research/final_thesis_dataset.csv``

Outputs:
    - ``research/descriptive/by_label_feature_summary.csv``
    - ``research/descriptive/by_split_label_feature_summary.csv``
    - ``research/descriptive/global_feature_summary.csv``
    - ``docs/descriptive_report.md``

Run:
    python3 workflow/generate_descriptive_tables.py
"""
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values):
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def quantile(values, q):
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def summarize(group_values):
    """Compute the standard summary stat bundle used in the descriptive tables.

    Args:
        group_values: Sequence of numeric values for one group.

    Returns:
        dict with ``n``, ``mean``, ``std``, ``p50``, ``p90`` and ``p95``.
    """
    return {
        "n": len(group_values),
        "mean": mean(group_values),
        "std": std(group_values),
        "p50": quantile(group_values, 0.5),
        "p90": quantile(group_values, 0.9),
        "p95": quantile(group_values, 0.95),
    }


def build():
    """Build all descriptive-statistics CSV tables and the Markdown digest."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "research" / "final_thesis_dataset.csv"
    out_root = project_root / "research" / "descriptive"
    out_root.mkdir(parents=True, exist_ok=True)

    by_label_path = out_root / "by_label_feature_summary.csv"
    by_split_label_path = out_root / "by_split_label_feature_summary.csv"
    global_path = out_root / "global_feature_summary.csv"
    report_path = project_root / "docs" / "descriptive_report.md"

    feature_cols = [
        "n_nodes",
        "n_edges",
        "n_users",
        "max_depth",
        "max_breadth",
        "duration_min_clean",
        "delay_mean",
        "delay_median",
        "delay_p90",
        "delay_p95",
        "nodes_le_10m_ratio",
        "nodes_le_30m_ratio",
        "nodes_le_60m_ratio",
        "nodes_le_120m_ratio",
        "source_text_len",
        "source_word_count",
        "source_upper_ratio",
        "avg_out_degree",
        "avg_internal_out_degree",
        "leaf_ratio",
        "density_directed",
    ]

    rows = []
    with dataset_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_label = defaultdict(lambda: defaultdict(list))
    by_split_label = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    global_values = defaultdict(list)

    for r in rows:
        label = r["label_4class"]
        split = r["split"]
        for col in feature_cols:
            v = to_float(r[col], 0.0)
            by_label[label][col].append(v)
            by_split_label[split][label][col].append(v)
            global_values[col].append(v)

    by_label_rows = []
    for label in sorted(by_label.keys()):
        for col in feature_cols:
            s = summarize(by_label[label][col])
            by_label_rows.append(
                {
                    "label_4class": label,
                    "feature": col,
                    "n": s["n"],
                    "mean": f"{s['mean']:.6f}",
                    "std": f"{s['std']:.6f}",
                    "p50": f"{s['p50']:.6f}",
                    "p90": f"{s['p90']:.6f}",
                    "p95": f"{s['p95']:.6f}",
                }
            )

    with by_label_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["label_4class", "feature", "n", "mean", "std", "p50", "p90", "p95"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_label_rows)

    by_split_rows = []
    for split in sorted(by_split_label.keys()):
        for label in sorted(by_split_label[split].keys()):
            for col in feature_cols:
                s = summarize(by_split_label[split][label][col])
                by_split_rows.append(
                    {
                        "split": split,
                        "label_4class": label,
                        "feature": col,
                        "n": s["n"],
                        "mean": f"{s['mean']:.6f}",
                        "std": f"{s['std']:.6f}",
                        "p50": f"{s['p50']:.6f}",
                        "p90": f"{s['p90']:.6f}",
                        "p95": f"{s['p95']:.6f}",
                    }
                )

    with by_split_label_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["split", "label_4class", "feature", "n", "mean", "std", "p50", "p90", "p95"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_split_rows)

    global_rows = []
    for col in feature_cols:
        s = summarize(global_values[col])
        global_rows.append(
            {
                "feature": col,
                "n": s["n"],
                "mean": f"{s['mean']:.6f}",
                "std": f"{s['std']:.6f}",
                "p50": f"{s['p50']:.6f}",
                "p90": f"{s['p90']:.6f}",
                "p95": f"{s['p95']:.6f}",
            }
        )

    with global_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["feature", "n", "mean", "std", "p50", "p90", "p95"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(global_rows)

    lines = []
    lines.append("# Descriptive Statistics Report")
    lines.append("")
    lines.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    lines.append("")
    lines.append("## Input")
    lines.append("- " + str(dataset_path))
    lines.append("")
    lines.append("## Outputs")
    lines.append("- " + str(by_label_path))
    lines.append("- " + str(by_split_label_path))
    lines.append("- " + str(global_path))
    lines.append("")
    lines.append("## Features summarized")
    for col in feature_cols:
        lines.append("- " + col)
    lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {by_label_path}")
    print(f"Wrote {by_split_label_path}")
    print(f"Wrote {global_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    build()
