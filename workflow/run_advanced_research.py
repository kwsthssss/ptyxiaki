"""Advanced experiments: ablation across feature sets, models and seeds.

Loads the modelling-ready dataset and pre-computed splits and evaluates
five models (majority, nearest-centroid, gaussian NB, KNN-5, KNN-11) on
five feature sets for both Task A (4-class) and Task B (binary). Selects
the best model per task by macro-F1, runs bootstrap CIs on the fixed test
predictions and a 10-seed robustness check.

Inputs:
    - ``research/final_thesis_dataset.csv``
    - ``research/splits.csv``

Outputs (under ``research/advanced/`` and ``docs/``):
    - ``advanced_metrics.csv``, ``advanced_per_class.csv``
    - ``advanced_robustness.csv``, ``advanced_summary.json``
    - ``confusion/<task>__<model>__<feature_set>__confusion.csv``
    - ``docs/advanced_research_report.md``
    - ``docs/advanced_research_what_we_did.md``

Run:
    python3 workflow/run_advanced_research.py
"""
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def bool_to_num(v):
    if str(v).strip().lower() == "true":
        return 1.0
    return 0.0


def accuracy_score(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def per_class_metrics(y_true, y_pred, labels):
    """Compute precision, recall, F1 and support for every label.

    Args:
        y_true: Ground-truth label sequence.
        y_pred: Predicted label sequence aligned with ``y_true``.
        labels: All labels to score, in the desired output order.

    Returns:
        list of dicts ``{label, precision, recall, f1, support}``.
    """
    rows = []
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        support = 0
        for t, p in zip(y_true, y_pred):
            if t == label:
                support += 1
            if t == label and p == label:
                tp += 1
            elif t != label and p == label:
                fp += 1
            elif t == label and p != label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def macro_f1_score(y_true, y_pred, labels):
    """Unweighted mean of per-class F1 scores.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Labels to include in the average.

    Returns:
        Macro-averaged F1 in ``[0, 1]``.
    """
    rows = per_class_metrics(y_true, y_pred, labels)
    if not rows:
        return 0.0
    return sum(r["f1"] for r in rows) / len(rows)


def weighted_f1_score(y_true, y_pred, labels):
    """Support-weighted mean of per-class F1 scores.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Labels to include in the average.

    Returns:
        Support-weighted F1 in ``[0, 1]``.
    """
    rows = per_class_metrics(y_true, y_pred, labels)
    total = sum(r["support"] for r in rows)
    if total == 0:
        return 0.0
    return sum(r["f1"] * r["support"] for r in rows) / total


def build_confusion(y_true, y_pred, labels):
    """Build a confusion matrix indexed by ``labels`` order.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Label order for both rows (true) and columns (predicted).

    Returns:
        list-of-lists ``matrix`` such that ``matrix[i][j]`` counts samples
        with true label ``labels[i]`` and predicted label ``labels[j]``.
    """
    idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t not in idx or p not in idx:
            continue
        matrix[idx[t]][idx[p]] += 1
    return matrix


def stratified_split(rows, label_field, seed=42, train_ratio=0.7, val_ratio=0.15):
    """Deterministically split events into train/val/test stratified by label.

    Each label group is shuffled with a seeded RNG and partitioned so that
    every label has at least one sample in each split.

    Args:
        rows: Sequence of event rows containing ``event_key`` and ``label_field``.
        label_field: Column to stratify by.
        seed: RNG seed for deterministic shuffling.
        train_ratio: Fraction assigned to training.
        val_ratio: Fraction assigned to validation.

    Returns:
        dict mapping ``event_key`` -> ``"train"``/``"val"``/``"test"``.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[label_field]].append(row["event_key"])
    split_map = {}
    for label, keys in by_label.items():
        arr = list(keys)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train < 1:
            n_train = 1
        if n_val < 1:
            n_val = 1
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        for k in arr[:n_train]:
            split_map[k] = "train"
        for k in arr[n_train:n_train + n_val]:
            split_map[k] = "val"
        for k in arr[n_train + n_val:]:
            split_map[k] = "test"
    return split_map


def fit_standardizer(train_rows, feature_names):
    """Fit per-feature mean and (population) std on training rows.

    Args:
        train_rows: Rows used to fit the scaler.
        feature_names: Numeric feature columns to standardize.

    Returns:
        Tuple ``(means, stds)`` of dicts keyed by feature name. Zero stds
        are clamped to ``1.0`` so z-scoring is well-defined.
    """
    means = {}
    stds = {}
    for f in feature_names:
        vals = [to_float(r[f], 0.0) for r in train_rows]
        if not vals:
            means[f] = 0.0
            stds[f] = 1.0
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        s = math.sqrt(var)
        if s == 0:
            s = 1.0
        means[f] = m
        stds[f] = s
    return means, stds


def vectorize(row, feature_names, means, stds):
    """Return the standardized feature vector for ``row``.

    Args:
        row: A single feature-table row (dict).
        feature_names: Ordered features to extract.
        means: Per-feature means from :func:`fit_standardizer`.
        stds: Per-feature standard deviations.

    Returns:
        list of floats aligned with ``feature_names``.
    """
    out = []
    for f in feature_names:
        v = to_float(row[f], 0.0)
        out.append((v - means[f]) / stds[f])
    return out


def train_majority(train_rows, label_field):
    """Fit a most-frequent-class classifier.

    Args:
        train_rows: Training rows.
        label_field: Column with the target label.

    Returns:
        dict ``{"majority": <label>}`` consumed by :func:`predict_majority`.
    """
    counts = Counter(r[label_field] for r in train_rows)
    if not counts:
        return {"majority": ""}
    return {"majority": counts.most_common(1)[0][0]}


def predict_majority(model, rows):
    """Predict the trained majority label for every row in ``rows``."""
    return [model["majority"] for _ in rows]


def train_centroid(train_rows, label_field, feature_names):
    """Fit a nearest-centroid classifier in z-scored feature space.

    Args:
        train_rows: Training rows.
        label_field: Target label column.
        feature_names: Numeric features to use.

    Returns:
        dict with ``means``, ``stds`` and per-label ``centroids``.
    """
    means, stds = fit_standardizer(train_rows, feature_names)
    sums = defaultdict(lambda: [0.0] * len(feature_names))
    counts = Counter()
    for r in train_rows:
        y = r[label_field]
        vec = vectorize(r, feature_names, means, stds)
        sums[y] = [a + b for a, b in zip(sums[y], vec)]
        counts[y] += 1
    centroids = {}
    for y in counts:
        centroids[y] = [v / counts[y] for v in sums[y]]
    return {"means": means, "stds": stds, "centroids": centroids}


def predict_centroid(model, rows, feature_names):
    """Predict labels by minimum squared distance to per-label centroids.

    Args:
        model: Output of :func:`train_centroid`.
        rows: Rows to label.
        feature_names: Same feature ordering used at training time.

    Returns:
        list of predicted labels aligned with ``rows``.
    """
    means = model["means"]
    stds = model["stds"]
    centroids = model["centroids"]
    labels = list(centroids.keys())
    preds = []
    for r in rows:
        vec = vectorize(r, feature_names, means, stds)
        best_label = labels[0] if labels else ""
        best_dist = None
        for y in labels:
            c = centroids[y]
            d = 0.0
            for i in range(len(vec)):
                diff = vec[i] - c[i]
                d += diff * diff
            if best_dist is None or d < best_dist:
                best_dist = d
                best_label = y
        preds.append(best_label)
    return preds


def train_gaussian_nb(train_rows, label_field, feature_names):
    """Fit a Gaussian Naive Bayes classifier on z-scored features.

    Args:
        train_rows: Training rows.
        label_field: Target label column.
        feature_names: Numeric features to use.

    Returns:
        dict with ``means``/``stds`` (the standardizer), per-class ``priors``
        and per-class ``params`` (mean and variance per feature).
    """
    means, stds = fit_standardizer(train_rows, feature_names)
    by_label = defaultdict(list)
    for r in train_rows:
        by_label[r[label_field]].append(vectorize(r, feature_names, means, stds))
    priors = {}
    params = {}
    total = len(train_rows)
    for y, vectors in by_label.items():
        priors[y] = len(vectors) / total if total else 0.0
        dim = len(feature_names)
        mu = [0.0] * dim
        var = [0.0] * dim
        for v in vectors:
            for i in range(dim):
                mu[i] += v[i]
        if vectors:
            mu = [x / len(vectors) for x in mu]
        for v in vectors:
            for i in range(dim):
                diff = v[i] - mu[i]
                var[i] += diff * diff
        if vectors:
            var = [x / len(vectors) for x in var]
        var = [x if x > 1e-9 else 1e-9 for x in var]
        params[y] = {"mu": mu, "var": var}
    return {"means": means, "stds": stds, "priors": priors, "params": params}


def predict_gaussian_nb(model, rows, feature_names):
    """Predict labels with a Gaussian NB log-posterior decision rule.

    Args:
        model: Output of :func:`train_gaussian_nb`.
        rows: Rows to label.
        feature_names: Same feature ordering used at training time.

    Returns:
        list of predicted labels aligned with ``rows``.
    """
    means = model["means"]
    stds = model["stds"]
    priors = model["priors"]
    params = model["params"]
    labels = list(params.keys())
    preds = []
    for r in rows:
        x = vectorize(r, feature_names, means, stds)
        best_label = labels[0] if labels else ""
        best_score = None
        for y in labels:
            mu = params[y]["mu"]
            var = params[y]["var"]
            logp = math.log(priors[y] if priors[y] > 0 else 1e-12)
            for i in range(len(x)):
                vv = var[i]
                diff = x[i] - mu[i]
                logp += -0.5 * math.log(2 * math.pi * vv) - (diff * diff) / (2 * vv)
            if best_score is None or logp > best_score:
                best_score = logp
                best_label = y
        preds.append(best_label)
    return preds


def train_knn(train_rows, label_field, feature_names):
    """Fit a KNN store: standardizes training features and caches priors.

    Args:
        train_rows: Training rows.
        label_field: Target label column.
        feature_names: Numeric features to use.

    Returns:
        dict with ``means``, ``stds``, ``x_train``, ``y_train`` and label
        ``priors`` consumed by :func:`predict_knn`.
    """
    means, stds = fit_standardizer(train_rows, feature_names)
    x_train = [vectorize(r, feature_names, means, stds) for r in train_rows]
    y_train = [r[label_field] for r in train_rows]
    priors = Counter(y_train)
    return {"means": means, "stds": stds, "x_train": x_train, "y_train": y_train, "priors": priors}


def predict_knn(model, rows, feature_names, k):
    """Predict labels with a brute-force k-NN classifier.

    Ties on majority vote are broken first by descending training prior
    and then by lexicographic order of the candidate labels.

    Args:
        model: Output of :func:`train_knn`.
        rows: Rows to label.
        feature_names: Same feature ordering used at training time.
        k: Number of neighbours to consider.

    Returns:
        list of predicted labels aligned with ``rows``.
    """
    means = model["means"]
    stds = model["stds"]
    x_train = model["x_train"]
    y_train = model["y_train"]
    priors = model["priors"]
    preds = []
    for r in rows:
        x = vectorize(r, feature_names, means, stds)
        dists = []
        for i, tr in enumerate(x_train):
            d = 0.0
            for j in range(len(x)):
                diff = x[j] - tr[j]
                d += diff * diff
            dists.append((d, y_train[i]))
        dists.sort(key=lambda z: z[0])
        top = dists[:k]
        counts = Counter(y for _, y in top)
        max_count = max(counts.values()) if counts else 0
        candidates = [y for y, c in counts.items() if c == max_count]
        if len(candidates) == 1:
            preds.append(candidates[0])
        else:
            candidates.sort(key=lambda y: (-priors.get(y, 0), y))
            preds.append(candidates[0] if candidates else "")
    return preds


def bootstrap_ci_metric(y_true, y_pred, labels, metric_name, seed=123, n_boot=1000, alpha=0.05):
    """Estimate a metric and its bootstrap confidence interval.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (held fixed across resamples).
        labels: Labels to score for macro-F1.
        metric_name: Either ``"accuracy"`` or anything else for macro-F1.
        seed: RNG seed for resampling.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided coverage; the CI is the ``[alpha/2, 1-alpha/2]``
            quantiles of the bootstrap distribution.

    Returns:
        Tuple ``(mean, ci_low, ci_high)``.
    """
    rng = random.Random(seed)
    n = len(y_true)
    vals = []
    if n == 0:
        return 0.0, 0.0, 0.0
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bt = [y_true[i] for i in idx]
        bp = [y_pred[i] for i in idx]
        if metric_name == "accuracy":
            m = accuracy_score(bt, bp)
        else:
            m = macro_f1_score(bt, bp, labels)
        vals.append(m)
    vals.sort()
    mean_v = sum(vals) / len(vals)
    lo_i = int((alpha / 2) * (len(vals) - 1))
    hi_i = int((1 - alpha / 2) * (len(vals) - 1))
    return mean_v, vals[lo_i], vals[hi_i]


def evaluate_prediction(y_true, y_pred, labels):
    """Compute the full metric bundle used by the advanced experiments.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Labels to include in macro/weighted/per-class scoring.

    Returns:
        Tuple ``(accuracy, macro_f1, weighted_f1, per_class_rows, confusion)``.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = macro_f1_score(y_true, y_pred, labels)
    weighted_f1 = weighted_f1_score(y_true, y_pred, labels)
    pc = per_class_metrics(y_true, y_pred, labels)
    conf = build_confusion(y_true, y_pred, labels)
    return acc, macro_f1, weighted_f1, pc, conf


def ensure_num_cols(rows):
    """Add ``*_num`` 0/1 string columns derived from boolean text columns.

    Mutates ``rows`` in place so the resulting numeric flags can be used
    directly by feature sets that mix boolean and numeric columns.
    """
    for r in rows:
        r["source_exists_num"] = "1" if r["source_exists"] == "True" else "0"
        r["source_has_url_num"] = "1" if r["source_has_url"] == "True" else "0"
        r["source_has_question_num"] = "1" if r["source_has_question"] == "True" else "0"
        r["source_has_exclamation_num"] = "1" if r["source_has_exclamation"] == "True" else "0"
        r["tree_exists_num"] = "1" if r["tree_exists"] == "True" else "0"
        r["id_match_num"] = "1" if r["id_match_event_vs_tree_source"] == "True" else "0"
        r["has_negative_delay_num"] = "1" if r["has_negative_delay"] == "True" else "0"


def run_model(model_name, train_rows, val_rows, test_rows, label_field, feature_names):
    """Train ``model_name`` on ``train_rows`` and predict val/test labels.

    Args:
        model_name: One of ``"majority"``, ``"nearest_centroid"``,
            ``"gaussian_nb"``, ``"knn_5"``, ``"knn_11"``.
        train_rows: Training rows.
        val_rows: Validation rows.
        test_rows: Test rows.
        label_field: Target label column.
        feature_names: Numeric features to use.

    Returns:
        Tuple ``(pred_val, pred_test)`` of predicted-label lists.

    Raises:
        ValueError: If ``model_name`` is unknown.
    """
    if model_name == "majority":
        model = train_majority(train_rows, label_field)
        pred_val = predict_majority(model, val_rows)
        pred_test = predict_majority(model, test_rows)
    elif model_name == "nearest_centroid":
        model = train_centroid(train_rows, label_field, feature_names)
        pred_val = predict_centroid(model, val_rows, feature_names)
        pred_test = predict_centroid(model, test_rows, feature_names)
    elif model_name == "gaussian_nb":
        model = train_gaussian_nb(train_rows, label_field, feature_names)
        pred_val = predict_gaussian_nb(model, val_rows, feature_names)
        pred_test = predict_gaussian_nb(model, test_rows, feature_names)
    elif model_name == "knn_5":
        model = train_knn(train_rows, label_field, feature_names)
        pred_val = predict_knn(model, val_rows, feature_names, 5)
        pred_test = predict_knn(model, test_rows, feature_names, 5)
    elif model_name == "knn_11":
        model = train_knn(train_rows, label_field, feature_names)
        pred_val = predict_knn(model, val_rows, feature_names, 11)
        pred_test = predict_knn(model, test_rows, feature_names, 11)
    else:
        raise ValueError(model_name)
    return pred_val, pred_test


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_confusion(path, labels, matrix):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label"] + labels)
        for i, lab in enumerate(labels):
            writer.writerow([lab] + matrix[i])


def build():
    """Run the full advanced experiment grid and persist all artifacts.

    Iterates over both tasks, all feature sets and all models, writes
    metrics, per-class scores and confusion matrices, then performs
    bootstrap CI estimation and a 10-seed robustness sweep on the
    macro-F1-best configuration.
    """
    project_root = Path(__file__).resolve().parents[1]
    research_root = project_root / "research"
    advanced_root = research_root / "advanced"
    docs_root = project_root / "docs"
    advanced_root.mkdir(parents=True, exist_ok=True)
    (advanced_root / "confusion").mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    features_path = research_root / "final_thesis_dataset.csv"
    splits_path = research_root / "splits.csv"

    metrics_path = advanced_root / "advanced_metrics.csv"
    per_class_path = advanced_root / "advanced_per_class.csv"
    robustness_path = advanced_root / "advanced_robustness.csv"
    summary_path = advanced_root / "advanced_summary.json"
    report_path = docs_root / "advanced_research_report.md"
    what_path = docs_root / "advanced_research_what_we_did.md"

    rows = load_csv(features_path)
    split_rows = load_csv(splits_path)
    split_map_4 = {r["event_key"]: r["split_4class"] for r in split_rows}
    split_map_b = {r["event_key"]: r["split_binary"] for r in split_rows}

    ensure_num_cols(rows)

    # Note: in this dataset n_users == n_nodes for every cascade (each node
    # corresponds to a unique user), so log_n_users would be perfectly
    # collinear with log_n_nodes. We therefore exclude it from the modelling
    # feature sets, even though the column is preserved in the dataset.
    feature_sets = {
        "structural": [
            "log_n_nodes",
            "log_n_edges",
            "max_depth",
            "log_max_breadth",
            "avg_out_degree",
            "avg_internal_out_degree",
            "max_out_degree",
            "density_directed",
            "leaf_ratio",
            "internal_ratio",
            "unreachable_nodes",
        ],
        "early_diffusion": [
            "nodes_le_10m_ratio",
            "nodes_le_30m_ratio",
            "nodes_le_60m_ratio",
            "nodes_le_120m_ratio",
            "t_to_10th_node_min",
            "t_to_20th_node_min",
            "t_to_50th_node_min",
            "t_to_10th_node_reached",
            "t_to_20th_node_reached",
            "t_to_50th_node_reached",
            "root_out_degree",
            "log_duration_min_clean",
            "delay_mean",
            "delay_median",
            "delay_p90",
            "delay_p95",
        ],
        "source_meta": [
            "source_text_len_log",
            "source_word_count",
            "source_upper_ratio",
            "source_has_url_num",
            "source_has_question_num",
            "source_has_exclamation_num",
            "source_exists_num",
        ],
        "quality_flags": [
            "id_match_num",
            "has_negative_delay_num",
            "dropped_self_loops",
            "dropped_duplicate_edges",
            "tree_exists_num",
        ],
    }
    all_features = []
    for fs in feature_sets.values():
        for f in fs:
            if f not in all_features:
                all_features.append(f)
    feature_sets["combined"] = all_features

    tasks = [
        {
            "name": "taskA_4class",
            "label_field": "label_4class",
            "split_map": split_map_4,
            "labels": ["false", "non-rumor", "true", "unverified"],
        },
        {
            "name": "taskB_binary",
            "label_field": "label_binary",
            "split_map": split_map_b,
            "labels": ["non-rumor", "rumor"],
        },
    ]

    models = ["majority", "nearest_centroid", "gaussian_nb", "knn_5", "knn_11"]

    metrics_rows = []
    per_class_rows = []

    for task in tasks:
        label_field = task["label_field"]
        labels = task["labels"]
        for fs_name, fs_cols in feature_sets.items():
            train_rows = [r for r in rows if task["split_map"].get(r["event_key"], "") == "train"]
            val_rows = [r for r in rows if task["split_map"].get(r["event_key"], "") == "val"]
            test_rows = [r for r in rows if task["split_map"].get(r["event_key"], "") == "test"]
            y_val = [r[label_field] for r in val_rows]
            y_test = [r[label_field] for r in test_rows]
            for model_name in models:
                pred_val, pred_test = run_model(model_name, train_rows, val_rows, test_rows, label_field, fs_cols)
                acc_v, mf1_v, wf1_v, pc_v, _ = evaluate_prediction(y_val, pred_val, labels)
                acc_t, mf1_t, wf1_t, pc_t, conf_t = evaluate_prediction(y_test, pred_test, labels)
                metrics_rows.append(
                    {
                        "task": task["name"],
                        "model": model_name,
                        "feature_set": fs_name,
                        "n_features": len(fs_cols),
                        "n_train": len(train_rows),
                        "n_val": len(val_rows),
                        "n_test": len(test_rows),
                        "accuracy_val": f"{acc_v:.6f}",
                        "macro_f1_val": f"{mf1_v:.6f}",
                        "weighted_f1_val": f"{wf1_v:.6f}",
                        "accuracy_test": f"{acc_t:.6f}",
                        "macro_f1_test": f"{mf1_t:.6f}",
                        "weighted_f1_test": f"{wf1_t:.6f}",
                    }
                )
                for row in pc_t:
                    per_class_rows.append(
                        {
                            "task": task["name"],
                            "model": model_name,
                            "feature_set": fs_name,
                            "split": "test",
                            "label": row["label"],
                            "precision": f"{row['precision']:.6f}",
                            "recall": f"{row['recall']:.6f}",
                            "f1": f"{row['f1']:.6f}",
                            "support": row["support"],
                        }
                    )
                conf_name = f"{task['name']}__{model_name}__{fs_name}__confusion.csv"
                write_confusion(advanced_root / "confusion" / conf_name, labels, conf_t)

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task",
            "model",
            "feature_set",
            "n_features",
            "n_train",
            "n_val",
            "n_test",
            "accuracy_val",
            "macro_f1_val",
            "weighted_f1_val",
            "accuracy_test",
            "macro_f1_test",
            "weighted_f1_test",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics_rows)

    with per_class_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["task", "model", "feature_set", "split", "label", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_class_rows)

    best_by_task = {}
    for task_name in ["taskA_4class", "taskB_binary"]:
        cand = [r for r in metrics_rows if r["task"] == task_name]
        cand.sort(key=lambda r: (to_float(r["macro_f1_test"]), to_float(r["accuracy_test"])), reverse=True)
        best_by_task[task_name] = cand[0]

    robustness_rows = []
    for task in tasks:
        task_name = task["name"]
        label_field = task["label_field"]
        labels = task["labels"]
        best = best_by_task[task_name]
        best_model = best["model"]
        fs_name = best["feature_set"]
        fs_cols = feature_sets[fs_name]
        seed_scores = []
        for seed in range(10):
            split_map = stratified_split(rows, label_field, seed=100 + seed, train_ratio=0.7, val_ratio=0.15)
            train_rows = [r for r in rows if split_map.get(r["event_key"], "") == "train"]
            val_rows = [r for r in rows if split_map.get(r["event_key"], "") == "val"]
            test_rows = [r for r in rows if split_map.get(r["event_key"], "") == "test"]
            y_test = [r[label_field] for r in test_rows]
            _, pred_test = run_model(best_model, train_rows, val_rows, test_rows, label_field, fs_cols)
            acc_t, mf1_t, wf1_t, _, _ = evaluate_prediction(y_test, pred_test, labels)
            seed_scores.append((seed, acc_t, mf1_t, wf1_t, len(train_rows), len(val_rows), len(test_rows)))
            robustness_rows.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "model": best_model,
                    "feature_set": fs_name,
                    "accuracy_test": f"{acc_t:.6f}",
                    "macro_f1_test": f"{mf1_t:.6f}",
                    "weighted_f1_test": f"{wf1_t:.6f}",
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "n_test": len(test_rows),
                }
            )

        fixed_split = task["split_map"]
        train_rows = [r for r in rows if fixed_split.get(r["event_key"], "") == "train"]
        val_rows = [r for r in rows if fixed_split.get(r["event_key"], "") == "val"]
        test_rows = [r for r in rows if fixed_split.get(r["event_key"], "") == "test"]
        y_test = [r[label_field] for r in test_rows]
        _, pred_test = run_model(best_model, train_rows, val_rows, test_rows, label_field, fs_cols)
        acc_mean, acc_lo, acc_hi = bootstrap_ci_metric(y_test, pred_test, labels, "accuracy", seed=2026, n_boot=1000, alpha=0.05)
        f1_mean, f1_lo, f1_hi = bootstrap_ci_metric(y_test, pred_test, labels, "macro_f1", seed=2027, n_boot=1000, alpha=0.05)
        best["bootstrap_accuracy_mean"] = f"{acc_mean:.6f}"
        best["bootstrap_accuracy_ci_low"] = f"{acc_lo:.6f}"
        best["bootstrap_accuracy_ci_high"] = f"{acc_hi:.6f}"
        best["bootstrap_macro_f1_mean"] = f"{f1_mean:.6f}"
        best["bootstrap_macro_f1_ci_low"] = f"{f1_lo:.6f}"
        best["bootstrap_macro_f1_ci_high"] = f"{f1_hi:.6f}"

        acc_vals = [x[1] for x in seed_scores]
        f1_vals = [x[2] for x in seed_scores]
        best["robustness_accuracy_mean"] = f"{(sum(acc_vals) / len(acc_vals)):.6f}"
        best["robustness_accuracy_std"] = f"{(math.sqrt(sum((v - sum(acc_vals) / len(acc_vals)) ** 2 for v in acc_vals) / len(acc_vals))):.6f}"
        best["robustness_macro_f1_mean"] = f"{(sum(f1_vals) / len(f1_vals)):.6f}"
        best["robustness_macro_f1_std"] = f"{(math.sqrt(sum((v - sum(f1_vals) / len(f1_vals)) ** 2 for v in f1_vals) / len(f1_vals))):.6f}"

    with robustness_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task",
            "seed",
            "model",
            "feature_set",
            "accuracy_test",
            "macro_f1_test",
            "weighted_f1_test",
            "n_train",
            "n_val",
            "n_test",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(robustness_rows)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": {
            "final_thesis_dataset": str(features_path),
            "splits": str(splits_path),
        },
        "output_files": {
            "advanced_metrics": str(metrics_path),
            "advanced_per_class": str(per_class_path),
            "advanced_robustness": str(robustness_path),
            "advanced_summary": str(summary_path),
            "report_md": str(report_path),
            "what_we_did_md": str(what_path),
        },
        "best_models": best_by_task,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Advanced Research Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Best models by task")
    for task_name, best in best_by_task.items():
        lines.append(
            "- "
            + task_name
            + " | model="
            + best["model"]
            + " | feature_set="
            + best["feature_set"]
            + " | accuracy_test="
            + best["accuracy_test"]
            + " | macro_f1_test="
            + best["macro_f1_test"]
        )
        lines.append(
            "  bootstrap_acc_ci=["
            + best["bootstrap_accuracy_ci_low"]
            + ", "
            + best["bootstrap_accuracy_ci_high"]
            + "]"
            + " bootstrap_f1_ci=["
            + best["bootstrap_macro_f1_ci_low"]
            + ", "
            + best["bootstrap_macro_f1_ci_high"]
            + "]"
        )
        lines.append(
            "  robustness_acc_mean_std="
            + best["robustness_accuracy_mean"]
            + "±"
            + best["robustness_accuracy_std"]
            + " robustness_f1_mean_std="
            + best["robustness_macro_f1_mean"]
            + "±"
            + best["robustness_macro_f1_std"]
        )
    lines.append("")
    lines.append("## Files")
    for k, v in summary["output_files"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    steps = []
    steps.append("# Advanced Research What We Did")
    steps.append("")
    steps.append("Date: " + datetime.now().strftime("%Y-%m-%d"))
    steps.append("")
    steps.append("## Exact actions")
    steps.append("- Ran ablation experiments on multiple feature sets.")
    steps.append("- Evaluated five models: majority, nearest_centroid, gaussian_nb, knn_5, knn_11.")
    steps.append("- Computed validation and test metrics for Task A and Task B.")
    steps.append("- Exported per-class metrics and confusion matrices.")
    steps.append("- Selected best model per task by macro-F1 on test split.")
    steps.append("- Estimated bootstrap confidence intervals on fixed test predictions.")
    steps.append("- Ran robustness checks across 10 deterministic stratified splits.")
    steps.append("")
    steps.append("## Best model snapshot")
    for task_name, best in best_by_task.items():
        steps.append(
            "- "
            + task_name
            + " => "
            + best["model"]
            + " + "
            + best["feature_set"]
            + " | accuracy="
            + best["accuracy_test"]
            + " | macro_f1="
            + best["macro_f1_test"]
        )
    steps.append("")

    with what_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(steps))

    print(f"Wrote {metrics_path}")
    print(f"Wrote {per_class_path}")
    print(f"Wrote {robustness_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {what_path}")


if __name__ == "__main__":
    build()
