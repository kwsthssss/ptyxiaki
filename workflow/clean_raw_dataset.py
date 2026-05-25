"""Clean and normalize the raw Twitter15/16 rumor-detection dataset.

Reads label files, source-tweet files, and per-event tree files from
``data_raw/Twitter15_16_dataset/{twitter15,twitter16}`` and produces a
normalized event/node/edge schema under ``data_clean/`` together with a
JSON cleaning summary and a Markdown report under ``docs/``.

Outputs:
    - ``data_clean/events_clean.csv``
    - ``data_clean/nodes_clean.csv``
    - ``data_clean/edges_clean.csv``
    - ``data_clean/cleaning_summary.json``
    - ``docs/cleaning_report.md``

Run:
    python3 workflow/clean_raw_dataset.py
"""
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import ast
import csv
import json


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def bool_str(value):
    return "True" if value else "False"


def sort_event_ids(ids):
    return sorted(ids, key=lambda x: int(x))


def read_labels(path):
    """Read a Twitter15/16 ``label.txt`` file into a dict.

    Args:
        path: Path to a ``label.txt`` file containing ``label:event_id`` lines.

    Returns:
        dict mapping ``event_id`` (str) to its 4-class label (str).
    """
    labels = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            label, event_id = line.split(":", 1)
            labels[event_id.strip()] = label.strip()
    return labels


def read_source_tweets(path):
    """Read a ``source_tweets.txt`` file mapping event ids to source text.

    Args:
        path: Path to a tab-separated ``event_id<TAB>text`` file.

    Returns:
        dict mapping ``event_id`` (str) to the raw source-tweet text (str).
    """
    source = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                event_id, text = line.split("\t", 1)
            else:
                event_id, text = line, ""
            source[event_id.strip()] = text
    return source


def normalize_node(node):
    """Normalize a single (user_id, tweet_id, delay) tree-node tuple.

    Args:
        node: A tuple/list with at least three elements as parsed by ``ast``.

    Returns:
        A dict with normalized ``user_id``, ``tweet_id``, ``delay_raw``,
        ``delay_value`` and non-negative ``delay_clean`` fields, or ``None``
        when the input is malformed.
    """
    if not isinstance(node, (list, tuple)):
        return None
    if len(node) < 3:
        return None
    user_id = str(node[0]).strip()
    tweet_id = str(node[1]).strip()
    delay_raw = str(node[2]).strip()
    if not user_id or not tweet_id:
        return None
    delay_value = to_float(delay_raw)
    delay_clean = None if delay_value is None else max(delay_value, 0.0)
    return {
        "user_id": user_id,
        "tweet_id": tweet_id,
        "delay_raw": delay_raw,
        "delay_value": delay_value,
        "delay_clean": delay_clean,
    }


def parse_tree(path):
    """Parse a Twitter15/16 cascade tree file into nodes, edges and stats.

    Strictly decodes each ``parent -> child`` line, drops malformed lines,
    self-loops and duplicate edges, and reports counts for each filter so
    the cleaning step is auditable.

    Args:
        path: Path to a per-event ``tree/<event_id>.txt`` file.

    Returns:
        dict containing the node table, edge list, source-node metadata,
        delay statistics and per-event quality counters used downstream.
    """
    nodes = {}
    node_order = []
    edge_pairs = []
    edge_pair_set = set()
    malformed_lines = 0
    invalid_delay_values = 0
    dropped_self_loops = 0
    dropped_duplicate_edges = 0
    source_node_key = None
    source_delay_raw = ""
    source_delay_clean = ""

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "->" not in line:
                malformed_lines += 1
                continue
            left_raw, right_raw = line.split("->", 1)
            try:
                left_obj = ast.literal_eval(left_raw)
                right_obj = ast.literal_eval(right_raw)
            except Exception:
                malformed_lines += 1
                continue

            left_node = normalize_node(left_obj)
            right_node = normalize_node(right_obj)
            if left_node is None or right_node is None:
                malformed_lines += 1
                continue

            left_is_root = left_node["user_id"] == "ROOT" and left_node["tweet_id"] == "ROOT"
            right_key = (right_node["user_id"], right_node["tweet_id"], right_node["delay_raw"])

            if right_key not in nodes:
                nodes[right_key] = right_node
                node_order.append(right_key)
                if right_node["delay_value"] is None:
                    invalid_delay_values += 1

            if left_is_root:
                if source_node_key is None:
                    source_node_key = right_key
                    source_delay_raw = right_node["delay_raw"]
                    source_delay_clean = "" if right_node["delay_clean"] is None else f"{right_node['delay_clean']:.2f}"
                continue

            left_key = (left_node["user_id"], left_node["tweet_id"], left_node["delay_raw"])
            if left_key not in nodes:
                nodes[left_key] = left_node
                node_order.append(left_key)
                if left_node["delay_value"] is None:
                    invalid_delay_values += 1

            if left_key == right_key:
                dropped_self_loops += 1
                continue

            edge_key = (left_key, right_key)
            if edge_key in edge_pair_set:
                dropped_duplicate_edges += 1
                continue

            edge_pair_set.add(edge_key)
            edge_pairs.append(edge_key)

    raw_delays = []
    clean_delays = []
    user_ids = set()
    for key in node_order:
        node = nodes[key]
        user_ids.add(node["user_id"])
        if node["delay_value"] is not None:
            raw_delays.append(node["delay_value"])
            clean_delays.append(node["delay_clean"])

    min_delay_raw = min(raw_delays) if raw_delays else None
    max_delay_raw = max(raw_delays) if raw_delays else None
    min_delay_clean = min(clean_delays) if clean_delays else None
    max_delay_clean = max(clean_delays) if clean_delays else None
    duration_clean = None if min_delay_clean is None or max_delay_clean is None else max_delay_clean - min_delay_clean
    has_negative_delay = any(d < 0 for d in raw_delays)

    source_user_id = ""
    source_tweet_id = ""
    if source_node_key is not None:
        source_user_id = source_node_key[0]
        source_tweet_id = source_node_key[1]

    return {
        "nodes": nodes,
        "node_order": node_order,
        "edges": edge_pairs,
        "source_node_key": source_node_key,
        "source_user_id": source_user_id,
        "source_tweet_id": source_tweet_id,
        "source_delay_raw": source_delay_raw,
        "source_delay_clean": source_delay_clean,
        "malformed_lines": malformed_lines,
        "invalid_delay_values": invalid_delay_values,
        "dropped_self_loops": dropped_self_loops,
        "dropped_duplicate_edges": dropped_duplicate_edges,
        "n_nodes": len(node_order),
        "n_edges": len(edge_pairs),
        "n_users": len(user_ids),
        "min_delay_raw": min_delay_raw,
        "max_delay_raw": max_delay_raw,
        "min_delay_clean": min_delay_clean,
        "max_delay_clean": max_delay_clean,
        "duration_clean": duration_clean,
        "has_negative_delay": has_negative_delay,
    }


def format_float(value):
    if value is None:
        return ""
    return f"{value:.2f}"


def build():
    """Run the full cleaning pipeline end-to-end.

    Walks both ``twitter15`` and ``twitter16`` raw folders, parses labels,
    source tweets and tree files, and writes the normalized
    ``events_clean``/``nodes_clean``/``edges_clean`` CSVs together with a
    JSON summary and a Markdown report. Side-effecting only.
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data_raw" / "Twitter15_16_dataset"
    clean_root = project_root / "data_clean"
    docs_root = project_root / "docs"
    clean_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    events_path = clean_root / "events_clean.csv"
    nodes_path = clean_root / "nodes_clean.csv"
    edges_path = clean_root / "edges_clean.csv"
    summary_path = clean_root / "cleaning_summary.json"
    report_path = docs_root / "cleaning_report.md"

    def rel(path):
        return str(path.relative_to(project_root))

    events_fields = [
        "event_key",
        "split",
        "event_id",
        "label",
        "source_exists",
        "source_text_len",
        "source_has_url",
        "tree_exists",
        "source_user_id",
        "source_tweet_id_from_tree",
        "id_match_event_vs_tree_source",
        "source_delay_min_raw",
        "source_delay_min_clean",
        "n_nodes",
        "n_edges",
        "n_users",
        "min_delay_raw",
        "max_delay_raw",
        "min_delay_clean",
        "max_delay_clean",
        "duration_min_clean",
        "has_negative_delay",
        "malformed_lines",
        "invalid_delay_values",
        "dropped_self_loops",
        "dropped_duplicate_edges",
        "tree_file",
        "source_text",
    ]

    nodes_fields = [
        "event_key",
        "node_id",
        "split",
        "event_id",
        "label",
        "user_id",
        "tweet_id",
        "delay_min_raw",
        "delay_min_clean",
        "is_source_node",
    ]

    edges_fields = [
        "event_key",
        "edge_id",
        "split",
        "event_id",
        "label",
        "parent_node_id",
        "child_node_id",
        "parent_user_id",
        "parent_tweet_id",
        "parent_delay_min_raw",
        "parent_delay_min_clean",
        "child_user_id",
        "child_tweet_id",
        "child_delay_min_raw",
        "child_delay_min_clean",
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": rel(raw_root),
        "output_files": {
            "events_clean": rel(events_path),
            "nodes_clean": rel(nodes_path),
            "edges_clean": rel(edges_path),
            "summary_json": rel(summary_path),
            "report_md": rel(report_path),
        },
        "totals": {
            "events": 0,
            "nodes": 0,
            "edges": 0,
            "missing_source_rows": 0,
            "missing_tree_files": 0,
            "event_tree_id_mismatches": 0,
            "events_with_negative_delay": 0,
            "malformed_lines": 0,
            "invalid_delay_values": 0,
            "dropped_self_loops": 0,
            "dropped_duplicate_edges": 0,
        },
        "split_counts": {},
        "label_counts": {},
        "label_counts_by_split": {},
    }

    split_counts = Counter()
    label_counts = Counter()
    label_counts_by_split = defaultdict(Counter)

    with events_path.open("w", newline="", encoding="utf-8") as f_events, nodes_path.open("w", newline="", encoding="utf-8") as f_nodes, edges_path.open("w", newline="", encoding="utf-8") as f_edges:
        events_writer = csv.DictWriter(f_events, fieldnames=events_fields)
        nodes_writer = csv.DictWriter(f_nodes, fieldnames=nodes_fields)
        edges_writer = csv.DictWriter(f_edges, fieldnames=edges_fields)
        events_writer.writeheader()
        nodes_writer.writeheader()
        edges_writer.writeheader()

        for split in ["twitter15", "twitter16"]:
            split_dir = raw_root / split
            labels = read_labels(split_dir / "label.txt")
            source_map = read_source_tweets(split_dir / "source_tweets.txt")

            for event_id in sort_event_ids(labels.keys()):
                label = labels[event_id]
                event_key = f"{split}__{event_id}"
                tree_path = split_dir / "tree" / f"{event_id}.txt"
                tree_exists = tree_path.exists()

                source_text = source_map.get(event_id, "")
                source_exists = event_id in source_map
                if not source_exists:
                    summary["totals"]["missing_source_rows"] += 1

                if tree_exists:
                    parsed = parse_tree(tree_path)
                else:
                    summary["totals"]["missing_tree_files"] += 1
                    parsed = {
                        "nodes": {},
                        "node_order": [],
                        "edges": [],
                        "source_node_key": None,
                        "source_user_id": "",
                        "source_tweet_id": "",
                        "source_delay_raw": "",
                        "source_delay_clean": "",
                        "malformed_lines": 0,
                        "invalid_delay_values": 0,
                        "dropped_self_loops": 0,
                        "dropped_duplicate_edges": 0,
                        "n_nodes": 0,
                        "n_edges": 0,
                        "n_users": 0,
                        "min_delay_raw": None,
                        "max_delay_raw": None,
                        "min_delay_clean": None,
                        "max_delay_clean": None,
                        "duration_clean": None,
                        "has_negative_delay": False,
                    }

                id_match = event_id == parsed["source_tweet_id"]
                if not id_match:
                    summary["totals"]["event_tree_id_mismatches"] += 1
                if parsed["has_negative_delay"]:
                    summary["totals"]["events_with_negative_delay"] += 1

                summary["totals"]["events"] += 1
                summary["totals"]["nodes"] += parsed["n_nodes"]
                summary["totals"]["edges"] += parsed["n_edges"]
                summary["totals"]["malformed_lines"] += parsed["malformed_lines"]
                summary["totals"]["invalid_delay_values"] += parsed["invalid_delay_values"]
                summary["totals"]["dropped_self_loops"] += parsed["dropped_self_loops"]
                summary["totals"]["dropped_duplicate_edges"] += parsed["dropped_duplicate_edges"]

                split_counts[split] += 1
                label_counts[label] += 1
                label_counts_by_split[split][label] += 1

                # Strict URL detection: count an http(s) scheme or a stand-alone
                # "url" token, but not arbitrary substrings like "curl" or
                # "URLs" embedded inside other words.
                source_text_lower = source_text.lower()
                source_has_url = (
                    "http://" in source_text_lower
                    or "https://" in source_text_lower
                    or " url " in f" {source_text_lower} "
                )

                events_writer.writerow(
                    {
                        "event_key": event_key,
                        "split": split,
                        "event_id": event_id,
                        "label": label,
                        "source_exists": bool_str(source_exists),
                        "source_text_len": len(source_text),
                        "source_has_url": bool_str(source_has_url),
                        "tree_exists": bool_str(tree_exists),
                        "source_user_id": parsed["source_user_id"],
                        "source_tweet_id_from_tree": parsed["source_tweet_id"],
                        "id_match_event_vs_tree_source": bool_str(id_match),
                        "source_delay_min_raw": parsed["source_delay_raw"],
                        "source_delay_min_clean": parsed["source_delay_clean"],
                        "n_nodes": parsed["n_nodes"],
                        "n_edges": parsed["n_edges"],
                        "n_users": parsed["n_users"],
                        "min_delay_raw": format_float(parsed["min_delay_raw"]),
                        "max_delay_raw": format_float(parsed["max_delay_raw"]),
                        "min_delay_clean": format_float(parsed["min_delay_clean"]),
                        "max_delay_clean": format_float(parsed["max_delay_clean"]),
                        "duration_min_clean": format_float(parsed["duration_clean"]),
                        "has_negative_delay": bool_str(parsed["has_negative_delay"]),
                        "malformed_lines": parsed["malformed_lines"],
                        "invalid_delay_values": parsed["invalid_delay_values"],
                        "dropped_self_loops": parsed["dropped_self_loops"],
                        "dropped_duplicate_edges": parsed["dropped_duplicate_edges"],
                        "tree_file": rel(tree_path),
                        "source_text": source_text,
                    }
                )

                node_ids = {}
                for idx, node_key in enumerate(parsed["node_order"], start=1):
                    node = parsed["nodes"][node_key]
                    node_id = f"{event_key}__n{idx}"
                    node_ids[node_key] = node_id
                    nodes_writer.writerow(
                        {
                            "event_key": event_key,
                            "node_id": node_id,
                            "split": split,
                            "event_id": event_id,
                            "label": label,
                            "user_id": node["user_id"],
                            "tweet_id": node["tweet_id"],
                            "delay_min_raw": node["delay_raw"],
                            "delay_min_clean": format_float(node["delay_clean"]),
                            "is_source_node": bool_str(node_key == parsed["source_node_key"]),
                        }
                    )

                for idx, edge in enumerate(parsed["edges"], start=1):
                    parent_key, child_key = edge
                    parent_node = parsed["nodes"][parent_key]
                    child_node = parsed["nodes"][child_key]
                    edges_writer.writerow(
                        {
                            "event_key": event_key,
                            "edge_id": f"{event_key}__e{idx}",
                            "split": split,
                            "event_id": event_id,
                            "label": label,
                            "parent_node_id": node_ids[parent_key],
                            "child_node_id": node_ids[child_key],
                            "parent_user_id": parent_node["user_id"],
                            "parent_tweet_id": parent_node["tweet_id"],
                            "parent_delay_min_raw": parent_node["delay_raw"],
                            "parent_delay_min_clean": format_float(parent_node["delay_clean"]),
                            "child_user_id": child_node["user_id"],
                            "child_tweet_id": child_node["tweet_id"],
                            "child_delay_min_raw": child_node["delay_raw"],
                            "child_delay_min_clean": format_float(child_node["delay_clean"]),
                        }
                    )

    summary["split_counts"] = dict(split_counts)
    summary["label_counts"] = dict(label_counts)
    summary["label_counts_by_split"] = {split: dict(counter) for split, counter in label_counts_by_split.items()}

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Data Cleaning Report")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- {summary['input_root']}")
    lines.append("")
    lines.append("## Cleaning actions")
    lines.append("- Parsed label and source files for twitter15/twitter16")
    lines.append("- Parsed all tree edge lines with strict tuple decoding")
    lines.append("- Removed malformed lines from graph construction and counted them")
    lines.append("- Removed self-loop edges and duplicate edges per event")
    lines.append("- Kept raw delays and added cleaned delay as max(delay, 0)")
    lines.append("- Kept source text and event labels in final clean events table")
    lines.append("- Built normalized outputs: events_clean, nodes_clean, edges_clean")
    lines.append("")
    lines.append("## Totals")
    for key, value in summary["totals"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Split counts")
    for key, value in summary["split_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Label counts")
    for key, value in summary["label_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Label counts by split")
    for split in sorted(summary["label_counts_by_split"].keys()):
        lines.append(f"- {split}:")
        for label, value in sorted(summary["label_counts_by_split"][split].items()):
            lines.append(f"  - {label}: {value}")
    lines.append("")
    lines.append("## Output files")
    for key, value in summary["output_files"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {events_path}")
    print(f"Wrote {nodes_path}")
    print(f"Wrote {edges_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    build()
