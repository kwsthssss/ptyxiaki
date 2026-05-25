"""Build event-level features, splits and baseline results for the thesis.

Consumes the cleaned tables under ``data_clean/`` produced by
``clean_raw_dataset.py`` and emits the modelling-ready dataset, the
deterministic stratified splits for the 4-class and binary tasks, and a
small set of baseline metrics (majority and nearest-centroid).

Inputs:
    - ``data_clean/events_clean.csv``
    - ``data_clean/nodes_clean.csv``
    - ``data_clean/edges_clean.csv``

Outputs (all under ``research/`` and ``docs/``):
    - ``research/event_features.csv``
    - ``research/final_thesis_dataset.csv``
    - ``research/splits.csv``
    - ``research/baseline_results.csv``
    - ``research/research_summary.json``
    - ``docs/research_report.md``
    - ``docs/research_what_we_did.md``

Run:
    python3 workflow/build_research_assets.py
"""
import csv
import json
import math
import random
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def log1p(v):
    if v < 0:
        return 0.0
    return math.log1p(v)


def group_rows_by_event(path, key_field):
    """Yield (event_key, rows) groups in the order they appear in the file.

    The cleaning step writes events_clean / nodes_clean / edges_clean in the
    same per-event order, so consumers of this iterator only need to consume
    groups sequentially in lock-step with events_clean. We deliberately do not
    rely on lexicographic comparison of event keys, since tweet IDs may not be
    of constant width.
    """
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        current_key = None
        bucket = []
        for row in reader:
            key = row[key_field]
            if current_key is None:
                current_key = key
                bucket = [row]
                continue
            if key == current_key:
                bucket.append(row)
                continue
            yield current_key, bucket
            current_key = key
            bucket = [row]
        if current_key is not None:
            yield current_key, bucket


def compute_graph_features(nodes_rows, edges_rows):
    """Compute structural and temporal cascade features for one event.

    Args:
        nodes_rows: Iterable of node-CSV rows belonging to a single event.
        edges_rows: Iterable of edge-CSV rows belonging to the same event.

    Returns:
        dict with size, depth/breadth, degree, density, time-to-N-th-node
        milestones and delay-distribution features used by all downstream
        models.
    """
    node_ids = [r["node_id"] for r in nodes_rows]
    node_set = set(node_ids)
    delays = [to_float(r["delay_min_clean"], 0.0) for r in nodes_rows]
    source_candidates = [r["node_id"] for r in nodes_rows if r["is_source_node"] == "True"]
    source_node_id = source_candidates[0] if source_candidates else ""

    out_neighbors = {n: [] for n in node_ids}
    in_degree = {n: 0 for n in node_ids}
    out_degree = {n: 0 for n in node_ids}

    for e in edges_rows:
        p = e["parent_node_id"]
        c = e["child_node_id"]
        if p not in node_set or c not in node_set:
            continue
        out_neighbors[p].append(c)
        out_degree[p] += 1
        in_degree[c] += 1

    n_nodes = len(node_ids)
    n_edges = len(edges_rows)

    leaves = sum(1 for n in node_ids if out_degree[n] == 0)
    internal = sum(1 for n in node_ids if out_degree[n] > 0)
    max_out_degree = max(out_degree.values()) if out_degree else 0
    avg_out_degree = safe_div(sum(out_degree.values()), n_nodes)
    avg_internal_out_degree = safe_div(sum(out_degree[n] for n in node_ids if out_degree[n] > 0), internal)
    root_out_degree = out_degree.get(source_node_id, 0)

    delays_sorted = sorted(delays)
    nodes_le_10m = sum(1 for d in delays if d <= 10.0)
    nodes_le_30m = sum(1 for d in delays if d <= 30.0)
    nodes_le_60m = sum(1 for d in delays if d <= 60.0)
    nodes_le_120m = sum(1 for d in delays if d <= 120.0)

    # Time-to-N-th-node milestones. If the cascade never reaches N nodes,
    # we mark the milestone as not reached and use the maximum observed
    # delay as a censored sentinel value (instead of silently using 0.0,
    # which would falsely indicate instant diffusion).
    max_observed_delay = delays_sorted[-1] if delays_sorted else 0.0

    if len(delays_sorted) >= 10:
        t_to_10 = delays_sorted[9]
        t_to_10_reached = 1
    else:
        t_to_10 = max_observed_delay
        t_to_10_reached = 0

    if len(delays_sorted) >= 20:
        t_to_20 = delays_sorted[19]
        t_to_20_reached = 1
    else:
        t_to_20 = max_observed_delay
        t_to_20_reached = 0

    if len(delays_sorted) >= 50:
        t_to_50 = delays_sorted[49]
        t_to_50_reached = 1
    else:
        t_to_50 = max_observed_delay
        t_to_50_reached = 0

    # BFS from source for depth/breadth metrics. Uses deque for O(1) popleft.
    depth_map = {}
    depth_counts = Counter()
    visited = set()

    if source_node_id and source_node_id in node_set:
        queue = deque([source_node_id])
        depth_map[source_node_id] = 0
        visited.add(source_node_id)
        while queue:
            parent = queue.popleft()
            parent_depth = depth_map[parent]
            depth_counts[parent_depth] += 1
            for child in out_neighbors[parent]:
                if child in visited:
                    continue
                visited.add(child)
                depth_map[child] = parent_depth + 1
                queue.append(child)

    unreachable_nodes = n_nodes - len(visited) if source_node_id else n_nodes
    max_depth = max(depth_map.values()) if depth_map else 0
    max_breadth = max(depth_counts.values()) if depth_counts else 0

    density_directed = safe_div(n_edges, n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    leaf_ratio = safe_div(leaves, n_nodes)
    internal_ratio = safe_div(internal, n_nodes)
    unique_delay_count = len(set(delays))

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_users": len(set(r["user_id"] for r in nodes_rows)),
        "max_depth": max_depth,
        "max_breadth": max_breadth,
        "root_out_degree": root_out_degree,
        "leaves": leaves,
        "leaf_ratio": leaf_ratio,
        "internal_nodes": internal,
        "internal_ratio": internal_ratio,
        "avg_out_degree": avg_out_degree,
        "avg_internal_out_degree": avg_internal_out_degree,
        "max_out_degree": max_out_degree,
        "density_directed": density_directed,
        "unreachable_nodes": unreachable_nodes,
        "nodes_le_10m": nodes_le_10m,
        "nodes_le_30m": nodes_le_30m,
        "nodes_le_60m": nodes_le_60m,
        "nodes_le_120m": nodes_le_120m,
        "nodes_le_10m_ratio": safe_div(nodes_le_10m, n_nodes),
        "nodes_le_30m_ratio": safe_div(nodes_le_30m, n_nodes),
        "nodes_le_60m_ratio": safe_div(nodes_le_60m, n_nodes),
        "nodes_le_120m_ratio": safe_div(nodes_le_120m, n_nodes),
        "t_to_10th_node_min": t_to_10,
        "t_to_20th_node_min": t_to_20,
        "t_to_50th_node_min": t_to_50,
        "t_to_10th_node_reached": t_to_10_reached,
        "t_to_20th_node_reached": t_to_20_reached,
        "t_to_50th_node_reached": t_to_50_reached,
        "delay_mean": safe_div(sum(delays), n_nodes),
        "delay_median": percentile(delays_sorted, 0.5),
        "delay_p90": percentile(delays_sorted, 0.9),
        "delay_p95": percentile(delays_sorted, 0.95),
        "delay_max": max(delays) if delays else 0.0,
        "delay_min": min(delays) if delays else 0.0,
        "unique_delay_count": unique_delay_count,
    }


def label_binary_from_4class(label):
    """Collapse the 4-class label space into a binary rumor/non-rumor label.

    Args:
        label: One of ``{"non-rumor", "false", "true", "unverified"}``.

    Returns:
        ``"non-rumor"`` for the negative class, otherwise ``"rumor"``.
    """
    if label == "non-rumor":
        return "non-rumor"
    return "rumor"


def stratified_split(rows, label_field, seed=42, train_ratio=0.7, val_ratio=0.15):
    """Deterministically split events into train/val/test stratified by label.

    Each label group is shuffled with a seeded RNG and partitioned so that
    every label has at least one sample in each split.

    Args:
        rows: Sequence of event rows containing ``event_key`` and ``label_field``.
        label_field: Column name to stratify by (e.g. ``"label_binary"``).
        seed: RNG seed for deterministic shuffling.
        train_ratio: Fraction of each label assigned to the training split.
        val_ratio: Fraction of each label assigned to the validation split.

    Returns:
        dict mapping ``event_key`` -> one of ``{"train", "val", "test"}``.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row[label_field]].append(row["event_key"])

    split_map = {}
    for label, keys in by_label.items():
        keys_copy = list(keys)
        rng.shuffle(keys_copy)
        n = len(keys_copy)
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
        train_keys = keys_copy[:n_train]
        val_keys = keys_copy[n_train:n_train + n_val]
        test_keys = keys_copy[n_train + n_val:]
        for k in train_keys:
            split_map[k] = "train"
        for k in val_keys:
            split_map[k] = "val"
        for k in test_keys:
            split_map[k] = "test"
    return split_map


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def macro_f1(y_true, y_pred, labels):
    f1_values = []
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(y_true, y_pred):
            if p == label and t == label:
                tp += 1
            elif p == label and t != label:
                fp += 1
            elif p != label and t == label:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_values.append(f1)
    if not f1_values:
        return 0.0
    return sum(f1_values) / len(f1_values)


def majority_class(train_rows, label_field):
    counts = Counter(r[label_field] for r in train_rows)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def build_standardization_stats(train_rows, feature_names):
    """Fit per-feature mean and (population) standard deviation on train rows.

    Args:
        train_rows: Sequence of feature-table rows used to fit the scaler.
        feature_names: Names of numeric feature columns to standardize.

    Returns:
        Tuple ``(means, stds)`` of dicts keyed by feature name. Zero stds
        are replaced by ``1.0`` to make z-scoring well-defined.
    """
    means = {}
    stds = {}
    for f in feature_names:
        vals = [to_float(r[f], 0.0) for r in train_rows]
        if not vals:
            means[f] = 0.0
            stds[f] = 1.0
            continue
        mean_v = sum(vals) / len(vals)
        var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
        std_v = math.sqrt(var_v)
        if std_v == 0:
            std_v = 1.0
        means[f] = mean_v
        stds[f] = std_v
    return means, stds


def z_features(row, feature_names, means, stds):
    """Return the z-score vector for ``row`` using the provided fit stats.

    Args:
        row: A single feature-table row (dict).
        feature_names: Ordered names of the features to vectorize.
        means: Dict of per-feature means from :func:`build_standardization_stats`.
        stds: Dict of per-feature standard deviations.

    Returns:
        list of floats, one per ``feature_names`` entry.
    """
    return [(to_float(row[f], 0.0) - means[f]) / stds[f] for f in feature_names]


def train_centroids(train_rows, label_field, feature_names):
    """Fit a nearest-centroid classifier in standardized feature space.

    Args:
        train_rows: Training-split rows.
        label_field: Column with the target label.
        feature_names: Ordered list of features to use.

    Returns:
        Tuple ``(centroids, means, stds)`` where ``centroids`` maps each
        observed label to its mean z-vector.
    """
    means, stds = build_standardization_stats(train_rows, feature_names)
    vectors_by_label = defaultdict(list)
    for r in train_rows:
        vectors_by_label[r[label_field]].append(z_features(r, feature_names, means, stds))
    centroids = {}
    for label, vectors in vectors_by_label.items():
        if not vectors:
            continue
        dim = len(vectors[0])
        centroid = []
        for i in range(dim):
            centroid.append(sum(v[i] for v in vectors) / len(vectors))
        centroids[label] = centroid
    return centroids, means, stds


def predict_centroid(rows, feature_names, centroids, means, stds):
    """Predict labels for ``rows`` by minimum squared distance to centroids.

    Args:
        rows: Sequence of feature rows to label.
        feature_names: Features to use, in the same order as during training.
        centroids: Mapping from label to mean z-vector.
        means: Per-feature means used to standardize ``rows``.
        stds: Per-feature standard deviations used to standardize ``rows``.

    Returns:
        list of predicted labels aligned with ``rows``.
    """
    labels = list(centroids.keys())
    preds = []
    for r in rows:
        v = z_features(r, feature_names, means, stds)
        best_label = ""
        best_dist = None
        for label in labels:
            c = centroids[label]
            d = 0.0
            for i in range(len(v)):
                diff = v[i] - c[i]
                d += diff * diff
            if best_dist is None or d < best_dist:
                best_dist = d
                best_label = label
        preds.append(best_label)
    return preds


def evaluate_task(rows, split_map, label_field, feature_names):
    """Run majority and nearest-centroid baselines for one classification task.

    Args:
        rows: Full feature table (all splits combined).
        split_map: Mapping ``event_key`` -> ``"train"``/``"val"``/``"test"``.
        label_field: Column with the target label for this task.
        feature_names: Numeric features used by nearest-centroid.

    Returns:
        Tuple ``(majority_metrics, centroid_metrics)`` dicts containing
        accuracy, macro-F1 and split sizes evaluated on the test split.
    """
    train_rows = [r for r in rows if split_map.get(r["event_key"], "") == "train"]
    val_rows = [r for r in rows if split_map.get(r["event_key"], "") == "val"]
    test_rows = [r for r in rows if split_map.get(r["event_key"], "") == "test"]

    labels_sorted = sorted(set(r[label_field] for r in rows))

    majority = majority_class(train_rows, label_field)
    y_test = [r[label_field] for r in test_rows]
    y_pred_majority = [majority for _ in test_rows]

    majority_metrics = {
        "model": "majority",
        "task_label": label_field,
        "accuracy_test": accuracy(y_test, y_pred_majority),
        "macro_f1_test": macro_f1(y_test, y_pred_majority, labels_sorted),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
    }

    centroids, means, stds = train_centroids(train_rows, label_field, feature_names)
    y_pred_centroid = predict_centroid(test_rows, feature_names, centroids, means, stds)
    centroid_metrics = {
        "model": "nearest_centroid",
        "task_label": label_field,
        "accuracy_test": accuracy(y_test, y_pred_centroid),
        "macro_f1_test": macro_f1(y_test, y_pred_centroid, labels_sorted),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
    }

    return majority_metrics, centroid_metrics


def build():
    """Run the full research-assets pipeline end-to-end.

    Loads the cleaned tables, computes per-event features, derives binary
    labels, builds deterministic stratified splits for both tasks, runs
    the majority and nearest-centroid baselines and writes all artifacts
    under ``research/`` and ``docs/``.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_clean_dir = project_root / "data_clean"
    docs_dir = project_root / "docs"
    research_dir = project_root / "research"
    docs_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    events_path = data_clean_dir / "events_clean.csv"
    nodes_path = data_clean_dir / "nodes_clean.csv"
    edges_path = data_clean_dir / "edges_clean.csv"

    features_path = research_dir / "event_features.csv"
    final_dataset_path = research_dir / "final_thesis_dataset.csv"
    splits_path = research_dir / "splits.csv"
    baselines_path = research_dir / "baseline_results.csv"
    summary_json_path = research_dir / "research_summary.json"
    report_md_path = docs_dir / "research_report.md"
    what_md_path = docs_dir / "research_what_we_did.md"

    node_groups_iter = iter(group_rows_by_event(nodes_path, "event_key"))
    edge_groups_iter = iter(group_rows_by_event(edges_path, "event_key"))
    current_node_group = next(node_groups_iter, (None, []))
    current_edge_group = next(edge_groups_iter, (None, []))

    feature_rows = []

    with events_path.open(encoding="utf-8", newline="") as f_events:
        events_reader = csv.DictReader(f_events)
        for event in events_reader:
            event_key = event["event_key"]

            # Nodes/edges files are written in the same per-event order as
            # events_clean, so we just align by exact key equality. No
            # lexicographic fast-forward (would be unsafe for ids of mixed
            # widths) — events without nodes/edges (e.g. missing tree files)
            # naturally fall through to the empty list branch.
            nodes_rows = current_node_group[1] if current_node_group[0] == event_key else []
            edges_rows = current_edge_group[1] if current_edge_group[0] == event_key else []

            label_4class = event["label"]
            label_binary = label_binary_from_4class(label_4class)

            graph = compute_graph_features(nodes_rows, edges_rows)
            duration_clean = to_float(event["duration_min_clean"], 0.0)
            source_text = event["source_text"]
            source_word_count = len([w for w in source_text.split(" ") if w.strip()])
            source_has_question = "?" in source_text
            source_has_exclamation = "!" in source_text
            source_upper_ratio = safe_div(sum(1 for ch in source_text if ch.isupper()), max(len(source_text), 1))

            row = {
                "event_key": event_key,
                "split": event["split"],
                "event_id": event["event_id"],
                "label_4class": label_4class,
                "label_binary": label_binary,
                "source_exists": event["source_exists"],
                "source_has_url": event["source_has_url"],
                "source_text_len": event["source_text_len"],
                "source_word_count": source_word_count,
                "source_has_question": "True" if source_has_question else "False",
                "source_has_exclamation": "True" if source_has_exclamation else "False",
                "source_upper_ratio": f"{source_upper_ratio:.6f}",
                "tree_exists": event["tree_exists"],
                "id_match_event_vs_tree_source": event["id_match_event_vs_tree_source"],
                "has_negative_delay": event["has_negative_delay"],
                "dropped_self_loops": event["dropped_self_loops"],
                "dropped_duplicate_edges": event["dropped_duplicate_edges"],
                "duration_min_clean": f"{duration_clean:.2f}",
                "log_duration_min_clean": f"{log1p(duration_clean):.6f}",
            }

            for k, v in graph.items():
                if isinstance(v, float):
                    row[k] = f"{v:.6f}"
                else:
                    row[k] = str(v)

            row["log_n_nodes"] = f"{log1p(to_float(row['n_nodes'], 0.0)):.6f}"
            row["log_n_edges"] = f"{log1p(to_float(row['n_edges'], 0.0)):.6f}"
            row["log_n_users"] = f"{log1p(to_float(row['n_users'], 0.0)):.6f}"
            row["log_max_breadth"] = f"{log1p(to_float(row['max_breadth'], 0.0)):.6f}"
            row["source_text_len_log"] = f"{log1p(to_float(row['source_text_len'], 0.0)):.6f}"
            feature_rows.append(row)

            if current_node_group[0] == event_key:
                current_node_group = next(node_groups_iter, (None, []))
            if current_edge_group[0] == event_key:
                current_edge_group = next(edge_groups_iter, (None, []))

    fieldnames = list(feature_rows[0].keys()) if feature_rows else []
    with features_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    with final_dataset_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)

    split_map_4class = stratified_split(feature_rows, "label_4class", seed=42, train_ratio=0.7, val_ratio=0.15)
    split_map_binary = stratified_split(feature_rows, "label_binary", seed=42, train_ratio=0.7, val_ratio=0.15)

    with splits_path.open("w", encoding="utf-8", newline="") as f:
        split_fields = ["event_key", "split_4class", "split_binary"]
        writer = csv.DictWriter(f, fieldnames=split_fields)
        writer.writeheader()
        for row in feature_rows:
            ek = row["event_key"]
            writer.writerow(
                {
                    "event_key": ek,
                    "split_4class": split_map_4class.get(ek, ""),
                    "split_binary": split_map_binary.get(ek, ""),
                }
            )

    # Baseline feature set. log_n_users is intentionally excluded because, in
    # this dataset, every cascade has n_users == n_nodes (each node maps to a
    # unique user), making it perfectly collinear with log_n_nodes.
    feature_set = [
        "log_n_nodes",
        "log_n_edges",
        "log_duration_min_clean",
        "max_depth",
        "log_max_breadth",
        "avg_out_degree",
        "avg_internal_out_degree",
        "leaf_ratio",
        "nodes_le_10m_ratio",
        "nodes_le_30m_ratio",
        "nodes_le_60m_ratio",
        "t_to_10th_node_min",
        "t_to_20th_node_min",
        "t_to_10th_node_reached",
        "t_to_20th_node_reached",
        "source_text_len_log",
    ]

    task_a_majority, task_a_centroid = evaluate_task(feature_rows, split_map_4class, "label_4class", feature_set)
    task_b_majority, task_b_centroid = evaluate_task(feature_rows, split_map_binary, "label_binary", feature_set)

    baseline_rows = []
    for task_name, metric in [
        ("taskA_4class", task_a_majority),
        ("taskA_4class", task_a_centroid),
        ("taskB_binary", task_b_majority),
        ("taskB_binary", task_b_centroid),
    ]:
        baseline_rows.append(
            {
                "task": task_name,
                "model": metric["model"],
                "label_space": metric["task_label"],
                "accuracy_test": f"{metric['accuracy_test']:.6f}",
                "macro_f1_test": f"{metric['macro_f1_test']:.6f}",
                "n_train": metric["n_train"],
                "n_val": metric["n_val"],
                "n_test": metric["n_test"],
            }
        )

    with baselines_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["task", "model", "label_space", "accuracy_test", "macro_f1_test", "n_train", "n_val", "n_test"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(baseline_rows)

    split_counts_4class = Counter(split_map_4class.values())
    split_counts_binary = Counter(split_map_binary.values())
    label_counts_4class = Counter(r["label_4class"] for r in feature_rows)
    label_counts_binary = Counter(r["label_binary"] for r in feature_rows)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_files": {
            "events_clean": str(events_path),
            "nodes_clean": str(nodes_path),
            "edges_clean": str(edges_path),
        },
        "output_files": {
            "event_features": str(features_path),
            "final_thesis_dataset": str(final_dataset_path),
            "splits": str(splits_path),
            "baseline_results": str(baselines_path),
            "summary_json": str(summary_json_path),
            "report_md": str(report_md_path),
            "what_we_did_md": str(what_md_path),
        },
        "counts": {
            "events_total": len(feature_rows),
            "labels_4class": dict(label_counts_4class),
            "labels_binary": dict(label_counts_binary),
            "split_4class": dict(split_counts_4class),
            "split_binary": dict(split_counts_binary),
        },
        "baselines": baseline_rows,
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Research Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- {events_path}")
    lines.append(f"- {nodes_path}")
    lines.append(f"- {edges_path}")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- {features_path}")
    lines.append(f"- {final_dataset_path}")
    lines.append(f"- {splits_path}")
    lines.append(f"- {baselines_path}")
    lines.append(f"- {summary_json_path}")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- events_total: {len(feature_rows)}")
    lines.append(f"- split_4class: {dict(split_counts_4class)}")
    lines.append(f"- split_binary: {dict(split_counts_binary)}")
    lines.append(f"- labels_4class: {dict(label_counts_4class)}")
    lines.append(f"- labels_binary: {dict(label_counts_binary)}")
    lines.append("")
    lines.append("## Baseline results")
    for row in baseline_rows:
        lines.append(
            f"- {row['task']} | {row['model']} | accuracy_test={row['accuracy_test']} | macro_f1_test={row['macro_f1_test']}"
        )
    lines.append("")

    with report_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    steps = []
    steps.append("# Research What We Did")
    steps.append("")
    steps.append("Date: " + datetime.now().strftime("%Y-%m-%d"))
    steps.append("")
    steps.append("## Exact actions")
    steps.append("- Loaded cleaned events, nodes, and edges tables.")
    steps.append("- Built event-level structural and temporal features from cascades.")
    steps.append("- Added binary label mapping: non-rumor vs rumor.")
    steps.append("- Added deterministic stratified splits for 4-class and binary tasks.")
    steps.append("- Ran baseline evaluation with majority and nearest-centroid models.")
    steps.append("- Saved all research artifacts in the research directory.")
    steps.append("")
    steps.append("## Generated files")
    steps.append("- " + str(features_path))
    steps.append("- " + str(final_dataset_path))
    steps.append("- " + str(splits_path))
    steps.append("- " + str(baselines_path))
    steps.append("- " + str(summary_json_path))
    steps.append("- " + str(report_md_path))
    steps.append("")
    steps.append("## Baseline snapshot")
    for row in baseline_rows:
        steps.append(
            "- "
            + row["task"]
            + " | "
            + row["model"]
            + " | accuracy="
            + row["accuracy_test"]
            + " | macro_f1="
            + row["macro_f1_test"]
        )
    steps.append("")

    with what_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(steps))

    print(f"Wrote {features_path}")
    print(f"Wrote {final_dataset_path}")
    print(f"Wrote {splits_path}")
    print(f"Wrote {baselines_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {report_md_path}")
    print(f"Wrote {what_md_path}")


if __name__ == "__main__":
    build()
