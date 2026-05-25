"""Render all PNG figures used in the thesis from the research artifacts.

Each ``save_*`` function reads one or more CSV/JSON inputs from
``research/`` (baselines, advanced metrics, robustness sweeps, statistical
tests) and writes a self-contained PNG to
``thesis_lualatex_project/figures/`` using PIL's ImageDraw — no external
plotting dependency.

Inputs:
    - ``research/baseline_results.csv``, ``research/research_summary.json``
    - ``research/advanced/advanced_metrics.csv``,
      ``advanced_per_class.csv``, ``advanced_robustness.csv``,
      ``advanced_summary.json``, ``confusion/*.csv``
    - ``research/stats/binary_group_stat_tests.csv``

Outputs:
    - ``thesis_lualatex_project/figures/*.png``

Run:
    python3 workflow/generate_thesis_figures.py
"""
import csv
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_12 = load_font(12)
FONT_14 = load_font(14)
FONT_16 = load_font(16)
FONT_20 = load_font(20)
FONT_24 = load_font(24)


def text_w(draw, txt, font):
    box = draw.textbbox((0, 0), txt, font=font)
    return box[2] - box[0]


def text_h(draw, txt, font):
    box = draw.textbbox((0, 0), txt, font=font)
    return box[3] - box[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_model_comparison(out_path, baseline_path, advanced_summary_path):
    """Render the per-task macro-F1 bar chart comparing baselines and best.

    Args:
        out_path: PNG file to write.
        baseline_path: ``research/baseline_results.csv``.
        advanced_summary_path: ``research/advanced/advanced_summary.json``.
    """
    baseline = read_csv(baseline_path)
    with open(advanced_summary_path, encoding="utf-8") as f:
        import json
        adv = json.load(f)

    a_major = next(r for r in baseline if r["task"] == "taskA_4class" and r["model"] == "majority")
    a_near = next(r for r in baseline if r["task"] == "taskA_4class" and r["model"] == "nearest_centroid")
    b_major = next(r for r in baseline if r["task"] == "taskB_binary" and r["model"] == "majority")
    b_near = next(r for r in baseline if r["task"] == "taskB_binary" and r["model"] == "nearest_centroid")

    a_best = adv["best_models"]["taskA_4class"]
    b_best = adv["best_models"]["taskB_binary"]

    labels = ["Majority", "Nearest Centroid", "Best Advanced"]
    task_a = [float(a_major["macro_f1_test"]), float(a_near["macro_f1_test"]), float(a_best["macro_f1_test"])]
    task_b = [float(b_major["macro_f1_test"]), float(b_near["macro_f1_test"]), float(b_best["macro_f1_test"])]

    w, h = 1400, 900
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, h), fill=(255, 255, 255))
    d.text((40, 25), "Model Comparison by Task (Macro-F1 Test)", fill=(20, 20, 20), font=FONT_24)
    d.text((40, 60), "Task A: 4-class    Task B: binary", fill=(70, 70, 70), font=FONT_14)

    left, top, right, bottom = 120, 120, w - 80, h - 120
    d.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    d.line((left, top, left, bottom), fill=(0, 0, 0), width=2)

    for i in range(0, 11):
        yv = i / 10
        y = bottom - (bottom - top) * yv
        d.line((left - 6, y, left, y), fill=(0, 0, 0), width=1)
        if i % 2 == 0:
            d.line((left, y, right, y), fill=(230, 230, 230), width=1)
        t = f"{yv:.1f}"
        d.text((left - 48, y - 8), t, fill=(40, 40, 40), font=FONT_12)

    n = len(labels)
    group_w = (right - left) / n
    bar_w = group_w * 0.25
    for i in range(n):
        gx = left + group_w * i + group_w * 0.15
        vals = [task_a[i], task_b[i]]
        colors = [(72, 128, 228), (228, 110, 72)]
        names = ["Task A", "Task B"]
        for j, v in enumerate(vals):
            x0 = gx + j * (bar_w + 14)
            x1 = x0 + bar_w
            y1 = bottom
            y0 = bottom - (bottom - top) * v
            d.rectangle((x0, y0, x1, y1), fill=colors[j], outline=(40, 40, 40), width=1)
            vt = f"{v:.3f}"
            d.text((x0 + 6, y0 - 20), vt, fill=(20, 20, 20), font=FONT_12)
        lw = text_w(d, labels[i], FONT_14)
        d.text((gx + bar_w - lw / 2 + 8, bottom + 20), labels[i], fill=(20, 20, 20), font=FONT_14)

    lx, ly = right - 220, top + 10
    d.rectangle((lx, ly, lx + 16, ly + 16), fill=(72, 128, 228), outline=(0, 0, 0))
    d.text((lx + 24, ly - 1), "Task A", fill=(20, 20, 20), font=FONT_14)
    d.rectangle((lx, ly + 30, lx + 16, ly + 46), fill=(228, 110, 72), outline=(0, 0, 0))
    d.text((lx + 24, ly + 29), "Task B", fill=(20, 20, 20), font=FONT_14)
    img.save(out_path)


def heat_color(v, vmax):
    if vmax <= 0:
        return (245, 245, 245)
    r = v / vmax
    r = max(0.0, min(1.0, r))
    b = int(255 - 165 * r)
    g = int(255 - 210 * r)
    rr = int(255 - 30 * r)
    return (rr, g, b)


def save_confusion(out_path, matrix, labels, title):
    """Render a confusion matrix heatmap PNG.

    Args:
        out_path: PNG file to write.
        matrix: list-of-lists confusion matrix.
        labels: Label order for both axes.
        title: Figure title.
    """
    w, h = 1100, 860
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), title, fill=(20, 20, 20), font=FONT_24)
    left, top = 220, 120
    cell = 130 if len(labels) == 4 else 220
    vmax = max(max(r) for r in matrix)
    n = len(labels)

    for i in range(n):
        for j in range(n):
            x0 = left + j * cell
            y0 = top + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            c = heat_color(matrix[i][j], vmax)
            d.rectangle((x0, y0, x1, y1), fill=c, outline=(70, 70, 70), width=2)
            t = str(matrix[i][j])
            tw = text_w(d, t, FONT_20)
            th = text_h(d, t, FONT_20)
            d.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2), t, fill=(10, 10, 10), font=FONT_20)

    for i, lab in enumerate(labels):
        tw = text_w(d, lab, FONT_16)
        d.text((left + i * cell + (cell - tw) / 2, top - 42), lab, fill=(20, 20, 20), font=FONT_16)
        d.text((65, top + i * cell + (cell - text_h(d, lab, FONT_16)) / 2), lab, fill=(20, 20, 20), font=FONT_16)

    d.text((left + cell * n / 2 - 60, top - 80), "Predicted label", fill=(30, 30, 30), font=FONT_16)
    d.text((20, top + cell * n / 2 - 10), "True label", fill=(30, 30, 30), font=FONT_16)
    img.save(out_path)


def quantiles(arr):
    a = sorted(arr)
    n = len(a)
    if n == 0:
        return 0, 0, 0, 0, 0
    def q(v):
        if n == 1:
            return a[0]
        p = (n - 1) * v
        lo = int(math.floor(p))
        hi = int(math.ceil(p))
        if lo == hi:
            return a[lo]
        return a[lo] + (a[hi] - a[lo]) * (p - lo)
    return a[0], q(0.25), q(0.5), q(0.75), a[-1]


def save_robustness(out_path, robustness_path):
    """Render box plots of accuracy and macro-F1 across robustness seeds.

    Args:
        out_path: PNG file to write.
        robustness_path: ``research/advanced/advanced_robustness.csv``.
    """
    rows = read_csv(robustness_path)
    a_acc = [float(r["accuracy_test"]) for r in rows if r["task"] == "taskA_4class"]
    a_f1 = [float(r["macro_f1_test"]) for r in rows if r["task"] == "taskA_4class"]
    b_acc = [float(r["accuracy_test"]) for r in rows if r["task"] == "taskB_binary"]
    b_f1 = [float(r["macro_f1_test"]) for r in rows if r["task"] == "taskB_binary"]
    groups = [("A-Accuracy", a_acc), ("A-MacroF1", a_f1), ("B-Accuracy", b_acc), ("B-MacroF1", b_f1)]

    w, h = 1400, 900
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 25), "Robustness Across 10 Stratified Seeds", fill=(20, 20, 20), font=FONT_24)
    left, top, right, bottom = 120, 120, w - 80, h - 110
    d.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    d.line((left, top, left, bottom), fill=(0, 0, 0), width=2)

    y_min, y_max = 0.35, 0.90
    for i in range(0, 12):
        yv = y_min + (y_max - y_min) * i / 11
        y = bottom - (bottom - top) * (yv - y_min) / (y_max - y_min)
        d.line((left - 6, y, left, y), fill=(0, 0, 0), width=1)
        if i % 2 == 0:
            d.line((left, y, right, y), fill=(235, 235, 235), width=1)
        d.text((left - 62, y - 8), f"{yv:.2f}", fill=(40, 40, 40), font=FONT_12)

    step = (right - left) / len(groups)
    box_w = step * 0.42
    for i, (name, vals) in enumerate(groups):
        mn, q1, med, q3, mx = quantiles(vals)
        x = left + step * i + step * 0.28
        def yy(v):
            return bottom - (bottom - top) * (v - y_min) / (y_max - y_min)
        y_mn, y_q1, y_med, y_q3, y_mx = yy(mn), yy(q1), yy(med), yy(q3), yy(mx)
        d.line((x + box_w / 2, y_mn, x + box_w / 2, y_q1), fill=(60, 60, 60), width=2)
        d.line((x + box_w / 2, y_q3, x + box_w / 2, y_mx), fill=(60, 60, 60), width=2)
        d.rectangle((x, y_q3, x + box_w, y_q1), fill=(144, 183, 235), outline=(40, 40, 40), width=2)
        d.line((x, y_med, x + box_w, y_med), fill=(180, 30, 30), width=3)
        d.line((x + box_w * 0.2, y_mn, x + box_w * 0.8, y_mn), fill=(60, 60, 60), width=2)
        d.line((x + box_w * 0.2, y_mx, x + box_w * 0.8, y_mx), fill=(60, 60, 60), width=2)
        tw = text_w(d, name, FONT_14)
        d.text((x + box_w / 2 - tw / 2, bottom + 16), name, fill=(20, 20, 20), font=FONT_14)

    img.save(out_path)


def save_effect_sizes(out_path, stat_path):
    """Render a Cliff's delta bar chart for the chosen temporal/structural features.

    Args:
        out_path: PNG file to write.
        stat_path: ``research/stats/binary_group_stat_tests.csv``.
    """
    rows = read_csv(stat_path)
    feats = ["n_nodes", "n_edges", "max_breadth", "density_directed", "t_to_10th_node_min", "t_to_20th_node_min", "t_to_50th_node_min"]
    vals = {}
    for r in rows:
        vals[r["feature"]] = float(r["cliffs_delta"])
    data = [(f, vals.get(f, 0.0)) for f in feats]

    w, h = 1400, 900
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Effect Sizes (Cliff's Delta): Rumor vs Non-rumor", fill=(20, 20, 20), font=FONT_24)
    left, top, right, bottom = 160, 110, w - 80, h - 90
    x0 = (left + right) // 2
    d.line((x0, top, x0, bottom), fill=(0, 0, 0), width=2)
    d.line((left, top, right, top), fill=(210, 210, 210), width=1)
    d.line((left, bottom, right, bottom), fill=(210, 210, 210), width=1)

    mn, mx = -0.75, 0.75
    for v in [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
        x = left + (right - left) * (v - mn) / (mx - mn)
        d.line((x, top, x, bottom), fill=(235, 235, 235), width=1)
        d.text((x - 14, bottom + 12), f"{v:.1f}", fill=(40, 40, 40), font=FONT_12)

    n = len(data)
    row_h = (bottom - top) / n
    for i, (f, v) in enumerate(data):
        cy = top + row_h * (i + 0.5)
        xv = left + (right - left) * (v - mn) / (mx - mn)
        color = (80, 130, 240) if v >= 0 else (240, 120, 80)
        d.rectangle((min(x0, xv), cy - 22, max(x0, xv), cy + 22), fill=color, outline=(70, 70, 70))
        d.text((28, cy - 9), f, fill=(20, 20, 20), font=FONT_14)
        d.text((xv + 6 if v >= 0 else xv - 62, cy - 9), f"{v:.3f}", fill=(20, 20, 20), font=FONT_12)

    img.save(out_path)


def scale_color(v, vmin, vmax):
    if vmax <= vmin:
        return (240, 240, 240)
    r = (v - vmin) / (vmax - vmin)
    r = max(0.0, min(1.0, r))
    b = int(255 - 170 * r)
    g = int(240 - 160 * r)
    rr = int(255 - 50 * r)
    return (rr, g, b)


def pretty_model(name):
    mapping = {
        "majority": "majority",
        "nearest_centroid": "nearest_centroid",
        "gaussian_nb": "gaussian_nb",
        "knn_5": "knn_5",
        "knn_11": "knn_11",
    }
    return mapping.get(name, name)


def pretty_feature(name):
    mapping = {
        "structural": "structural",
        "early_diffusion": "early_diffusion",
        "source_meta": "source_meta",
        "quality_flags": "quality_flags",
        "combined": "combined",
    }
    return mapping.get(name, name)


def save_metric_heatmap(out_path, metrics_path, task, metric, title):
    """Render a model x feature-set heatmap for a single metric and task.

    Args:
        out_path: PNG file to write.
        metrics_path: ``research/advanced/advanced_metrics.csv``.
        task: Task name to filter on (e.g. ``"taskA_4class"``).
        metric: Metric column to color (e.g. ``"macro_f1_test"``).
        title: Figure title.
    """
    rows = read_csv(metrics_path)
    rows = [r for r in rows if r["task"] == task]
    models = ["majority", "nearest_centroid", "gaussian_nb", "knn_5", "knn_11"]
    feature_sets = ["structural", "early_diffusion", "source_meta", "quality_flags", "combined"]
    lookup = {}
    for r in rows:
        lookup[(r["model"], r["feature_set"])] = float(r[metric])
    vals = []
    for m in models:
        for f in feature_sets:
            vals.append(lookup.get((m, f), 0.0))
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0

    w, h = 1500, 980
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), title, fill=(20, 20, 20), font=FONT_24)
    d.text((40, 58), f"task={task} | metric={metric}", fill=(60, 60, 60), font=FONT_14)

    left, top = 300, 140
    cell_w, cell_h = 220, 130
    for i, m in enumerate(models):
        y = top + i * cell_h
        d.text((70, y + cell_h // 2 - 8), pretty_model(m), fill=(20, 20, 20), font=FONT_16)
        for j, f in enumerate(feature_sets):
            x = left + j * cell_w
            v = lookup.get((m, f), 0.0)
            c = scale_color(v, vmin, vmax)
            d.rectangle((x, y, x + cell_w, y + cell_h), fill=c, outline=(80, 80, 80), width=2)
            t = f"{v:.3f}"
            tw = text_w(d, t, FONT_20)
            d.text((x + (cell_w - tw) / 2, y + cell_h / 2 - 12), t, fill=(10, 10, 10), font=FONT_20)

    for j, f in enumerate(feature_sets):
        x = left + j * cell_w
        tw = text_w(d, pretty_feature(f), FONT_14)
        d.text((x + (cell_w - tw) / 2, top - 36), pretty_feature(f), fill=(20, 20, 20), font=FONT_14)

    lx, ly, lw, lh = left, h - 120, cell_w * 5, 24
    for k in range(lw):
        r = k / max(1, lw - 1)
        v = vmin + (vmax - vmin) * r
        c = scale_color(v, vmin, vmax)
        d.line((lx + k, ly, lx + k, ly + lh), fill=c, width=1)
    d.rectangle((lx, ly, lx + lw, ly + lh), outline=(60, 60, 60), width=2)
    d.text((lx, ly + lh + 8), f"{vmin:.3f}", fill=(30, 30, 30), font=FONT_12)
    tmax = f"{vmax:.3f}"
    d.text((lx + lw - text_w(d, tmax, FONT_12), ly + lh + 8), tmax, fill=(30, 30, 30), font=FONT_12)
    img.save(out_path)


def save_feature_set_best_by_task(out_path, metrics_path):
    """Render the best macro-F1 (over models) per feature set, per task.

    Args:
        out_path: PNG file to write.
        metrics_path: ``research/advanced/advanced_metrics.csv``.
    """
    rows = read_csv(metrics_path)
    feature_sets = ["structural", "early_diffusion", "source_meta", "quality_flags", "combined"]
    tasks = ["taskA_4class", "taskB_binary"]
    best = {t: {f: 0.0 for f in feature_sets} for t in tasks}
    for r in rows:
        t = r["task"]
        f = r["feature_set"]
        v = float(r["macro_f1_test"])
        if t in best and f in best[t] and v > best[t][f]:
            best[t][f] = v

    w, h = 1500, 920
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Best Macro-F1 by Feature Set and Task", fill=(20, 20, 20), font=FONT_24)
    d.text((40, 58), "max over models per feature set", fill=(60, 60, 60), font=FONT_14)
    left, top, right, bottom = 120, 130, w - 80, h - 130
    d.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    d.line((left, top, left, bottom), fill=(0, 0, 0), width=2)
    for i in range(0, 11):
        yv = i / 10
        y = bottom - (bottom - top) * yv
        d.line((left - 6, y, left, y), fill=(0, 0, 0), width=1)
        if i % 2 == 0:
            d.line((left, y, right, y), fill=(235, 235, 235), width=1)
        d.text((left - 44, y - 8), f"{yv:.1f}", fill=(40, 40, 40), font=FONT_12)

    group_w = (right - left) / len(feature_sets)
    bar_w = group_w * 0.30
    colors = [(72, 128, 228), (228, 110, 72)]
    for i, f in enumerate(feature_sets):
        gx = left + group_w * i + group_w * 0.18
        vals = [best["taskA_4class"][f], best["taskB_binary"][f]]
        for j, v in enumerate(vals):
            x0 = gx + j * (bar_w + 18)
            x1 = x0 + bar_w
            y0 = bottom - (bottom - top) * v
            d.rectangle((x0, y0, x1, bottom), fill=colors[j], outline=(40, 40, 40), width=1)
            d.text((x0 + 4, y0 - 18), f"{v:.3f}", fill=(20, 20, 20), font=FONT_12)
        label = pretty_feature(f)
        d.text((gx - 6, bottom + 20), label, fill=(20, 20, 20), font=FONT_14)

    lx, ly = right - 240, top + 14
    d.rectangle((lx, ly, lx + 16, ly + 16), fill=colors[0], outline=(0, 0, 0))
    d.text((lx + 24, ly - 1), "Task A", fill=(20, 20, 20), font=FONT_14)
    d.rectangle((lx, ly + 30, lx + 16, ly + 46), fill=colors[1], outline=(0, 0, 0))
    d.text((lx + 24, ly + 29), "Task B", fill=(20, 20, 20), font=FONT_14)
    img.save(out_path)


def save_per_class_f1_best_models(out_path, per_class_path, summary_path):
    """Render per-class F1 bars for the best model of each task.

    Args:
        out_path: PNG file to write.
        per_class_path: ``research/advanced/advanced_per_class.csv``.
        summary_path: ``research/advanced/advanced_summary.json`` (used to
            resolve which (model, feature_set) combination is best per task).
    """
    rows = read_csv(per_class_path)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    a = summary["best_models"]["taskA_4class"]
    b = summary["best_models"]["taskB_binary"]
    a_rows = [r for r in rows if r["task"] == "taskA_4class" and r["model"] == a["model"] and r["feature_set"] == a["feature_set"] and r["split"] == "test"]
    b_rows = [r for r in rows if r["task"] == "taskB_binary" and r["model"] == b["model"] and r["feature_set"] == b["feature_set"] and r["split"] == "test"]
    a_order = ["false", "non-rumor", "true", "unverified"]
    b_order = ["non-rumor", "rumor"]
    a_map = {r["label"]: float(r["f1"]) for r in a_rows}
    b_map = {r["label"]: float(r["f1"]) for r in b_rows}
    a_vals = [a_map.get(x, 0.0) for x in a_order]
    b_vals = [b_map.get(x, 0.0) for x in b_order]

    w, h = 1600, 950
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Per-class F1 for Best Models", fill=(20, 20, 20), font=FONT_24)
    d.text((40, 58), "Task A: knn_11 + combined | Task B: knn_11 + structural", fill=(60, 60, 60), font=FONT_14)

    panels = [
        ("Task A (4-class)", a_order, a_vals, (100, 120, 760, 860), (72, 128, 228)),
        ("Task B (binary)", b_order, b_vals, (860, 120, 1520, 860), (228, 110, 72)),
    ]
    for title, labels, vals, box, color in panels:
        left, top, right, bottom = box
        d.rectangle((left, top, right, bottom), outline=(180, 180, 180), width=1)
        d.text((left + 12, top + 8), title, fill=(20, 20, 20), font=FONT_16)
        pl, pt, pr, pb = left + 70, top + 50, right - 20, bottom - 60
        d.line((pl, pb, pr, pb), fill=(0, 0, 0), width=2)
        d.line((pl, pt, pl, pb), fill=(0, 0, 0), width=2)
        for i in range(0, 11):
            yv = i / 10
            y = pb - (pb - pt) * yv
            if i % 2 == 0:
                d.line((pl, y, pr, y), fill=(235, 235, 235), width=1)
            d.line((pl - 4, y, pl, y), fill=(0, 0, 0), width=1)
            if i % 2 == 0:
                d.text((pl - 38, y - 8), f"{yv:.1f}", fill=(50, 50, 50), font=FONT_12)
        step = (pr - pl) / len(labels)
        bw = step * 0.55
        for i, (lab, v) in enumerate(zip(labels, vals)):
            x0 = pl + step * i + step * 0.2
            x1 = x0 + bw
            y0 = pb - (pb - pt) * v
            d.rectangle((x0, y0, x1, pb), fill=color, outline=(40, 40, 40), width=1)
            d.text((x0 + 5, y0 - 18), f"{v:.3f}", fill=(20, 20, 20), font=FONT_12)
            d.text((x0, pb + 14), lab, fill=(20, 20, 20), font=FONT_12)
    img.save(out_path)


def save_seed_trajectories(out_path, robustness_path):
    """Render line plots of accuracy / macro-F1 across robustness seeds.

    Args:
        out_path: PNG file to write.
        robustness_path: ``research/advanced/advanced_robustness.csv``.
    """
    rows = read_csv(robustness_path)
    rows = sorted(rows, key=lambda r: int(r["seed"]))
    a = [r for r in rows if r["task"] == "taskA_4class"]
    b = [r for r in rows if r["task"] == "taskB_binary"]
    x_vals = [int(r["seed"]) for r in a]
    a_acc = [float(r["accuracy_test"]) for r in a]
    a_f1 = [float(r["macro_f1_test"]) for r in a]
    b_acc = [float(r["accuracy_test"]) for r in b]
    b_f1 = [float(r["macro_f1_test"]) for r in b]
    series = [
        ("A-Accuracy", x_vals, a_acc, (72, 128, 228)),
        ("A-MacroF1", x_vals, a_f1, (125, 86, 208)),
        ("B-Accuracy", x_vals, b_acc, (228, 110, 72)),
        ("B-MacroF1", x_vals, b_f1, (58, 160, 98)),
    ]

    w, h = 1500, 950
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Robustness Trajectories by Seed", fill=(20, 20, 20), font=FONT_24)
    left, top, right, bottom = 120, 120, w - 80, h - 130
    d.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    d.line((left, top, left, bottom), fill=(0, 0, 0), width=2)

    y_min, y_max = 0.40, 0.90
    for i in range(0, 11):
        yv = y_min + (y_max - y_min) * i / 10
        y = bottom - (bottom - top) * (yv - y_min) / (y_max - y_min)
        d.line((left - 6, y, left, y), fill=(0, 0, 0), width=1)
        if i % 2 == 0:
            d.line((left, y, right, y), fill=(235, 235, 235), width=1)
        d.text((left - 54, y - 8), f"{yv:.2f}", fill=(40, 40, 40), font=FONT_12)

    x_min, x_max = 0, 9
    def sx(x):
        return left + (right - left) * (x - x_min) / (x_max - x_min)
    def sy(v):
        return bottom - (bottom - top) * (v - y_min) / (y_max - y_min)

    for seed in range(x_min, x_max + 1):
        x = sx(seed)
        d.line((x, bottom, x, bottom + 6), fill=(0, 0, 0), width=1)
        d.text((x - 5, bottom + 12), str(seed), fill=(40, 40, 40), font=FONT_12)

    for name, xs, ys, color in series:
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=3)
        for x, y in pts:
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(30, 30, 30))

    lx, ly = right - 250, top + 12
    for i, (name, _, _, color) in enumerate(series):
        yy = ly + i * 28
        d.rectangle((lx, yy, lx + 16, yy + 16), fill=color, outline=(0, 0, 0))
        d.text((lx + 24, yy - 1), name, fill=(20, 20, 20), font=FONT_14)
    img.save(out_path)


def save_label_distribution(out_path, summary_path):
    """Render side-by-side label-count bar charts for Task A and Task B.

    Args:
        out_path: PNG file to write.
        summary_path: ``research/research_summary.json``.
    """
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    c = summary["counts"]
    labels4 = ["false", "non-rumor", "true", "unverified"]
    vals4 = [int(c["labels_4class"][k]) for k in labels4]
    labels2 = ["non-rumor", "rumor"]
    vals2 = [int(c["labels_binary"][k]) for k in labels2]

    w, h = 1450, 900
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Label Distribution for Task A and Task B", fill=(20, 20, 20), font=FONT_24)
    panels = [
        ("Task A (4-class)", labels4, vals4, (80, 110, 760, 830), (84, 140, 233)),
        ("Task B (binary)", labels2, vals2, (820, 110, 1370, 830), (230, 124, 77)),
    ]
    for title, labs, vals, box, color in panels:
        left, top, right, bottom = box
        d.rectangle((left, top, right, bottom), outline=(180, 180, 180), width=1)
        d.text((left + 12, top + 8), title, fill=(20, 20, 20), font=FONT_16)
        pl, pt, pr, pb = left + 70, top + 50, right - 20, bottom - 70
        d.line((pl, pb, pr, pb), fill=(0, 0, 0), width=2)
        d.line((pl, pt, pl, pb), fill=(0, 0, 0), width=2)
        vmax = max(vals) * 1.1
        for i in range(0, 6):
            yv = vmax * i / 5
            y = pb - (pb - pt) * yv / vmax
            if i % 1 == 0:
                d.line((pl, y, pr, y), fill=(240, 240, 240), width=1)
                d.text((pl - 48, y - 8), str(int(yv)), fill=(50, 50, 50), font=FONT_12)
        step = (pr - pl) / len(labs)
        bw = step * 0.55
        for i, (lab, v) in enumerate(zip(labs, vals)):
            x0 = pl + step * i + step * 0.2
            x1 = x0 + bw
            y0 = pb - (pb - pt) * v / vmax
            d.rectangle((x0, y0, x1, pb), fill=color, outline=(40, 40, 40), width=1)
            d.text((x0 + 5, y0 - 18), str(v), fill=(20, 20, 20), font=FONT_12)
            d.text((x0, pb + 14), lab, fill=(20, 20, 20), font=FONT_12)
    img.save(out_path)


def save_temporal_asymmetry(out_path, stat_path):
    """Render -log10(p-value) bars contrasting mean vs median permutation tests.

    Focused on the time-to-N-th-node and duration features so the
    asymmetry between mean and median significance is visible.

    Args:
        out_path: PNG file to write.
        stat_path: ``research/stats/binary_group_stat_tests.csv``.
    """
    rows = read_csv(stat_path)
    target = ["t_to_10th_node_min", "t_to_20th_node_min", "t_to_50th_node_min", "duration_min_clean"]
    data = []
    for f in target:
        r = next((x for x in rows if x["feature"] == f), None)
        if r is not None:
            p_mean = float(r["permutation_pvalue_mean_diff"])
            p_med = float(r["permutation_pvalue_median_diff"])
            data.append((f, p_mean, p_med))
    vals = []
    for _, a, b in data:
        vals.append(-math.log10(max(a, 1e-9)))
        vals.append(-math.log10(max(b, 1e-9)))
    vmax = max(vals) * 1.1 if vals else 4.0

    w, h = 1500, 920
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.text((40, 24), "Temporal Features: Mean vs Median Permutation Significance", fill=(20, 20, 20), font=FONT_24)
    d.text((40, 58), "bars show -log10(p-value)", fill=(60, 60, 60), font=FONT_14)
    left, top, right, bottom = 140, 130, w - 80, h - 130
    d.line((left, bottom, right, bottom), fill=(0, 0, 0), width=2)
    d.line((left, top, left, bottom), fill=(0, 0, 0), width=2)
    for i in range(0, 9):
        yv = vmax * i / 8
        y = bottom - (bottom - top) * yv / vmax
        d.line((left - 6, y, left, y), fill=(0, 0, 0), width=1)
        if i % 1 == 0:
            d.line((left, y, right, y), fill=(235, 235, 235), width=1)
        d.text((left - 62, y - 8), f"{yv:.1f}", fill=(40, 40, 40), font=FONT_12)
    group_w = (right - left) / max(1, len(data))
    bar_w = group_w * 0.30
    for i, (name, p_mean, p_med) in enumerate(data):
        xg = left + group_w * i + group_w * 0.2
        vm = -math.log10(max(p_mean, 1e-9))
        vd = -math.log10(max(p_med, 1e-9))
        y0 = bottom - (bottom - top) * vm / vmax
        y1 = bottom - (bottom - top) * vd / vmax
        d.rectangle((xg, y0, xg + bar_w, bottom), fill=(104, 150, 233), outline=(40, 40, 40), width=1)
        d.rectangle((xg + bar_w + 16, y1, xg + bar_w * 2 + 16, bottom), fill=(233, 134, 82), outline=(40, 40, 40), width=1)
        d.text((xg, y0 - 18), f"{p_mean:.3f}", fill=(20, 20, 20), font=FONT_12)
        d.text((xg + bar_w + 16, y1 - 18), f"{p_med:.3f}", fill=(20, 20, 20), font=FONT_12)
        d.text((xg - 10, bottom + 18), name, fill=(20, 20, 20), font=FONT_12)
    lx, ly = right - 220, top + 14
    d.rectangle((lx, ly, lx + 16, ly + 16), fill=(104, 150, 233), outline=(0, 0, 0))
    d.text((lx + 24, ly - 1), "mean p-value", fill=(20, 20, 20), font=FONT_14)
    d.rectangle((lx, ly + 30, lx + 16, ly + 46), fill=(233, 134, 82), outline=(0, 0, 0))
    d.text((lx + 24, ly + 29), "median p-value", fill=(20, 20, 20), font=FONT_14)
    img.save(out_path)


def load_confusion_csv(path, labels):
    """Read a confusion CSV produced by run_advanced_research and return a
    list-of-lists matrix aligned with the given label order."""
    rows = read_csv(path)
    idx = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for r in rows:
        true_lab = r["true_label"]
        if true_lab not in idx:
            continue
        for col, val in r.items():
            if col == "true_label" or col not in idx:
                continue
            try:
                matrix[idx[true_lab]][idx[col]] = int(val)
            except (TypeError, ValueError):
                matrix[idx[true_lab]][idx[col]] = 0
    return matrix


def _resolve_best(adv_summary_path, task_name):
    with open(adv_summary_path, encoding="utf-8") as f:
        adv = json.load(f)
    return adv["best_models"][task_name]


def build():
    """Render every thesis figure to ``thesis_lualatex_project/figures/``."""
    root = Path(__file__).resolve().parents[1]
    research = root / "research"
    adv = research / "advanced"
    stats = research / "stats"
    out = root / "thesis_lualatex_project" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    save_model_comparison(
        out / "model_comparison.png",
        research / "baseline_results.csv",
        adv / "advanced_summary.json",
    )

    # Build confusion-matrix figures from the actual CSVs of the current best
    # models, instead of hard-coded matrices that would drift if the pipeline
    # is re-run with different splits or features.
    labels_a = ["false", "non-rumor", "true", "unverified"]
    labels_b = ["non-rumor", "rumor"]
    best_a = _resolve_best(adv / "advanced_summary.json", "taskA_4class")
    best_b = _resolve_best(adv / "advanced_summary.json", "taskB_binary")

    conf_a_path = adv / "confusion" / f"taskA_4class__{best_a['model']}__{best_a['feature_set']}__confusion.csv"
    conf_b_path = adv / "confusion" / f"taskB_binary__{best_b['model']}__{best_b['feature_set']}__confusion.csv"

    save_confusion(
        out / "confusion_taskA_best.png",
        load_confusion_csv(conf_a_path, labels_a),
        labels_a,
        "Confusion Matrix - Task A Best Model",
    )
    save_confusion(
        out / "confusion_taskB_best.png",
        load_confusion_csv(conf_b_path, labels_b),
        labels_b,
        "Confusion Matrix - Task B Best Model",
    )
    save_robustness(
        out / "robustness_boxplots.png",
        adv / "advanced_robustness.csv",
    )
    save_effect_sizes(
        out / "effect_sizes_cliffs_delta.png",
        stats / "binary_group_stat_tests.csv",
    )
    save_metric_heatmap(
        out / "advanced_macrof1_heatmap_taskA.png",
        adv / "advanced_metrics.csv",
        "taskA_4class",
        "macro_f1_test",
        "Advanced Macro-F1 Heatmap (Task A)",
    )
    save_metric_heatmap(
        out / "advanced_macrof1_heatmap_taskB.png",
        adv / "advanced_metrics.csv",
        "taskB_binary",
        "macro_f1_test",
        "Advanced Macro-F1 Heatmap (Task B)",
    )
    save_feature_set_best_by_task(
        out / "feature_set_best_by_task.png",
        adv / "advanced_metrics.csv",
    )
    save_per_class_f1_best_models(
        out / "per_class_f1_best_models.png",
        adv / "advanced_per_class.csv",
        adv / "advanced_summary.json",
    )
    save_seed_trajectories(
        out / "robustness_seed_trajectories.png",
        adv / "advanced_robustness.csv",
    )
    save_label_distribution(
        out / "label_distribution_tasks.png",
        research / "research_summary.json",
    )
    save_temporal_asymmetry(
        out / "temporal_pvalue_asymmetry.png",
        stats / "binary_group_stat_tests.csv",
    )
    print("Wrote figures to", out)


if __name__ == "__main__":
    build()
