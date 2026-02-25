from collections import defaultdict
import csv

def _parse_semicolon_indices(s: str):
    """
    induced_errors stored as: "i;j" or "i;j;k"
    Returns: [i, j] / [i, j, k]
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [int(x) for x in s.split(";") if x != ""]

def compute_position_error_numbers_from_ior(ior_csv_path: str, out_prefix: str):
    """
    Reads your per-test report CSV (with corrected header):
      node, origami_data, decoded_stream, induced_errors, success, decoding_time

    Writes:
      1) <out_prefix>_position_counts_by_node.csv
      2) <out_prefix>_position_counts_global.csv

    Numbers only (no rates):
      - total_occurrences (how many tested combos include that bit)
      - fail_occurrences  (how many of those combos failed)
    """
    per_node = defaultdict(lambda: {"total": 0, "fail": 0})
    global_counts = defaultdict(lambda: {"total": 0, "fail": 0})

    with open(ior_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = int(float(row["node"]))
            combo = _parse_semicolon_indices(row.get("induced_errors", ""))

            # success can be "0/1" or "True/False"
            s = str(row.get("success", "0")).strip().lower()
            success = (s in {"1", "true", "t", "yes", "y"})
            failed = not success

            for bit in combo:
                per_node[(node, bit)]["total"] += 1
                global_counts[bit]["total"] += 1
                if failed:
                    per_node[(node, bit)]["fail"] += 1
                    global_counts[bit]["fail"] += 1

    # Write per-node counts
    by_node_path = f"{out_prefix}_position_counts_by_node.csv"
    rows = [(n, b, d["total"], d["fail"]) for (n, b), d in per_node.items()]
    rows.sort(key=lambda x: (-x[3], -x[2], x[0], x[1]))  # most failures first

    with open(by_node_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node", "bit_index", "total_occurrences", "fail_occurrences"])
        w.writerows(rows)

    # Write global counts
    global_path = f"{out_prefix}_position_counts_global.csv"
    grows = [(b, d["total"], d["fail"]) for b, d in global_counts.items()]
    grows.sort(key=lambda x: (-x[2], -x[1], x[0]))  # most failures first

    with open(global_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bit_index", "total_occurrences", "fail_occurrences"])
        w.writerows(grows)

    print("[counts] wrote:", by_node_path)
    print("[counts] wrote:", global_path)
