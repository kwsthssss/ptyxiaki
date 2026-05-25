"""Permutation-based hypothesis tests on rumor vs non-rumor groups.

For each event-level feature in the modelling dataset, computes mean and
median group differences, two-sided permutation p-values and Cliff's
delta effect size with a magnitude bin label.

Inputs:
    - ``research/final_thesis_dataset.csv``

Outputs:
    - ``research/stats/binary_group_stat_tests.csv``
    - ``docs/stat_tests_report.md``

Run:
    python3 workflow/run_stat_tests.py
"""
import bisect
import csv
import math
import random
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


def median(values):
    if not values:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    mid = n // 2
    if n % 2 == 1:
        return arr[mid]
    return 0.5 * (arr[mid - 1] + arr[mid])


def cliffs_delta(a, b):
    """Cliff's delta effect size, computed in O((n1+n2) log(n1+n2)) via
    binary search instead of the naive O(n1*n2) double loop.

    delta = (gt - lt) / (n1 * n2) where gt = #{(i,j): a_i > b_j},
    lt = #{(i,j): a_i < b_j}.
    """
    n1 = len(a)
    n2 = len(b)
    if n1 == 0 or n2 == 0:
        return 0.0
    b_sorted = sorted(b)
    gt = 0
    lt = 0
    for x in a:
        # bisect_left  → count of b_j < x  (these contribute to gt)
        # bisect_right → count of b_j <= x; n2 - bisect_right → count of b_j > x (contribute to lt)
        lo = bisect.bisect_left(b_sorted, x)
        hi = bisect.bisect_right(b_sorted, x)
        gt += lo
        lt += n2 - hi
    return (gt - lt) / (n1 * n2)


def permutation_pvalue_stat(a, b, n_perm=3000, seed=123, stat="mean"):
    """Two-sided permutation test for a difference in mean or median.

    Returns the standard add-one estimate (hits + 1) / (n_perm + 1). Note that
    the smallest reportable p-value is therefore 1 / (n_perm + 1); any value
    equal to that floor should be interpreted as p < 1/(n_perm+1) rather
    than as an exact estimate.
    """
    rng = random.Random(seed)
    if not a or not b:
        return 1.0
    if stat == "mean":
        obs = abs(mean(a) - mean(b))
    else:
        obs = abs(median(a) - median(b))
    combined = list(a + b)
    n1 = len(a)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        g1 = combined[:n1]
        g2 = combined[n1:]
        if stat == "mean":
            d = abs(mean(g1) - mean(g2))
        else:
            d = abs(median(g1) - median(g2))
        if d >= obs:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def delta_magnitude(d):
    """Map a Cliff's delta value to its conventional magnitude bin.

    Args:
        d: Cliff's delta in ``[-1, 1]``.

    Returns:
        ``"negligible"``, ``"small"``, ``"medium"`` or ``"large"`` using
        Romano et al. thresholds (0.147 / 0.33 / 0.474).
    """
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def build():
    """Run permutation tests for every feature and write CSV + Markdown.

    Loads the modelling dataset, splits rows by ``label_binary``, and for
    each configured feature reports mean/median, group differences,
    Cliff's delta and permutation p-values for the mean and median
    statistics.
    """
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "research" / "final_thesis_dataset.csv"
    out_root = project_root / "research" / "stats"
    out_root.mkdir(parents=True, exist_ok=True)

    out_csv = out_root / "binary_group_stat_tests.csv"
    report_md = project_root / "docs" / "stat_tests_report.md"

    feature_cols = [
        "n_nodes",
        "n_edges",
        "n_users",
        "max_depth",
        "max_breadth",
        "duration_min_clean",
        "nodes_le_10m_ratio",
        "nodes_le_30m_ratio",
        "nodes_le_60m_ratio",
        "nodes_le_120m_ratio",
        "t_to_10th_node_min",
        "t_to_20th_node_min",
        "t_to_50th_node_min",
        "avg_out_degree",
        "avg_internal_out_degree",
        "leaf_ratio",
        "density_directed",
        "source_text_len",
        "source_word_count",
        "source_upper_ratio",
    ]

    rumor = {f: [] for f in feature_cols}
    non_rumor = {f: [] for f in feature_cols}

    with dataset_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            target = rumor if r["label_binary"] == "rumor" else non_rumor
            for col in feature_cols:
                target[col].append(to_float(r[col], 0.0))

    rows_out = []
    for i, col in enumerate(feature_cols):
        a = rumor[col]
        b = non_rumor[col]
        p_mean = permutation_pvalue_stat(a, b, n_perm=3000, seed=2000 + i, stat="mean")
        p_median = permutation_pvalue_stat(a, b, n_perm=3000, seed=4000 + i, stat="median")
        d = cliffs_delta(a, b)
        rows_out.append(
            {
                "feature": col,
                "n_rumor": len(a),
                "n_non_rumor": len(b),
                "mean_rumor": f"{mean(a):.6f}",
                "mean_non_rumor": f"{mean(b):.6f}",
                "median_rumor": f"{median(a):.6f}",
                "median_non_rumor": f"{median(b):.6f}",
                "mean_diff_rumor_minus_non_rumor": f"{(mean(a) - mean(b)):.6f}",
                "median_diff_rumor_minus_non_rumor": f"{(median(a) - median(b)):.6f}",
                "cliffs_delta": f"{d:.6f}",
                "cliffs_magnitude": delta_magnitude(d),
                "permutation_pvalue_mean_diff": f"{p_mean:.6f}",
                "permutation_pvalue_median_diff": f"{p_median:.6f}",
                "significant_mean_0_05": "True" if p_mean < 0.05 else "False",
                "significant_median_0_05": "True" if p_median < 0.05 else "False",
            }
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "feature",
            "n_rumor",
            "n_non_rumor",
            "mean_rumor",
            "mean_non_rumor",
            "median_rumor",
            "median_non_rumor",
            "mean_diff_rumor_minus_non_rumor",
            "median_diff_rumor_minus_non_rumor",
            "cliffs_delta",
            "cliffs_magnitude",
            "permutation_pvalue_mean_diff",
            "permutation_pvalue_median_diff",
            "significant_mean_0_05",
            "significant_median_0_05",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    top_effect = sorted(rows_out, key=lambda r: abs(to_float(r["cliffs_delta"], 0.0)), reverse=True)[:8]
    top_sig_mean = sorted(rows_out, key=lambda r: to_float(r["permutation_pvalue_mean_diff"], 1.0))[:8]
    top_sig_median = sorted(rows_out, key=lambda r: to_float(r["permutation_pvalue_median_diff"], 1.0))[:8]

    lines = []
    lines.append("# Statistical Tests Report")
    lines.append("")
    lines.append("Generated: " + datetime.now().isoformat(timespec="seconds"))
    lines.append("")
    lines.append("## Input")
    lines.append("- " + str(dataset_path))
    lines.append("")
    lines.append("## Output")
    lines.append("- " + str(out_csv))
    lines.append("")
    lines.append("## Group definition")
    lines.append("- Group A: rumor")
    lines.append("- Group B: non-rumor")
    lines.append("")
    lines.append("## Method")
    lines.append("- Effect size: Cliff's delta.")
    lines.append("- Significance: two-sided permutation tests on mean and median differences.")
    lines.append("- Permutations per feature: 3000.")
    lines.append(
        "- p-values use the (hits+1)/(n_perm+1) estimator, so the smallest"
        " reportable value is 1/3001 ≈ 3.3e-04. A printed p-value equal to"
        " that floor should be read as p < 1/3001."
    )
    lines.append("")
    lines.append("## Top effect sizes")
    for r in top_effect:
        lines.append(
            "- "
            + r["feature"]
            + " | cliffs_delta="
            + r["cliffs_delta"]
            + " ("
            + r["cliffs_magnitude"]
            + ") | p_mean="
            + r["permutation_pvalue_mean_diff"]
            + " | p_median="
            + r["permutation_pvalue_median_diff"]
        )
    lines.append("")
    lines.append("## Smallest p-values (mean difference)")
    for r in top_sig_mean:
        lines.append(
            "- "
            + r["feature"]
            + " | p_mean="
            + r["permutation_pvalue_mean_diff"]
            + " | p_median="
            + r["permutation_pvalue_median_diff"]
            + " | cliffs_delta="
            + r["cliffs_delta"]
        )
    lines.append("")
    lines.append("## Smallest p-values (median difference)")
    for r in top_sig_median:
        lines.append(
            "- "
            + r["feature"]
            + " | p_median="
            + r["permutation_pvalue_median_diff"]
            + " | p_mean="
            + r["permutation_pvalue_mean_diff"]
            + " | cliffs_delta="
            + r["cliffs_delta"]
        )
    lines.append("")

    with report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {out_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    build()
