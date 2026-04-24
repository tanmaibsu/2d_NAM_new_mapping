import csv
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILES = {
    "First 2 Nodes": os.path.join(SCRIPT_DIR, "decoded_first_two_nodes_ior.csv"),
    "Last 2 Nodes": os.path.join(SCRIPT_DIR, "decoded_last_two_nodes_4_ior.csv"),
}

NODES = [0, 1, 2, 3]


def load_stats(filepath):
    with open(filepath) as f:
        rows = list(csv.DictReader(f))

    stats = {}
    for n in NODES:
        node_rows = [r for r in rows if r["node"] == str(n)]
        total = len(node_rows)
        successes = sum(1 for r in node_rows if r["success"] == "1")
        stats[n] = {"total": total, "successes": successes, "failures": total - successes}
    return stats


def plot_success_rate(stats_40, stats_24):
    x = np.arange(len(NODES))
    width = 0.35

    rates_40 = [100 * stats_40[n]["successes"] / stats_40[n]["total"] for n in NODES]
    rates_24 = [100 * stats_24[n]["successes"] / stats_24[n]["total"] for n in NODES]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, rates_40, width, label="First 2 Nodes", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, rates_24, width, label="Last 2 Nodes", color="#DD8452")

    ax.set_xlabel("Node")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Decoding Success Rate by Node (2-bit Error Exhaustive Test)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in NODES])
    ax.set_ylim(96, 100.5)
    ax.legend()

    for bar, rate in zip(bars1, rates_40):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar, rate in zip(bars2, rates_24):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "success_rate_by_node.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_failure_count(stats_40, stats_24):
    x = np.arange(len(NODES))
    width = 0.35

    fails_40 = [stats_40[n]["failures"] for n in NODES]
    fails_24 = [stats_24[n]["failures"] for n in NODES]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, fails_40, width, label="First 2 Nodes", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, fails_24, width, label="Last 2 Nodes", color="#DD8452")

    ax.set_xlabel("Node")
    ax.set_ylabel("Number of Failures")
    ax.set_title("Decoding Failures by Node (2-bit Error Exhaustive Test)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {n}" for n in NODES])
    ax.legend()

    for bar, count in zip(bars1, fails_40):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, count in zip(bars2, fails_24):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{count}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "failure_count_by_node.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    stats_first = load_stats(DATA_FILES["First 2 Nodes"])
    stats_last = load_stats(DATA_FILES["Last 2 Nodes"])

    print("\n=== Summary ===")
    print(f"{'Node':<6} {'First2 total':<13} {'First2 success':<16} {'Last2 total':<12} {'Last2 success':<16}")
    for n in NODES:
        sf, sl = stats_first[n], stats_last[n]
        print(f"{n:<6} {sf['total']:<13} {sf['successes']}/{sf['total']} ({100*sf['successes']/sf['total']:.1f}%)      "
              f"{sl['total']:<12} {sl['successes']}/{sl['total']} ({100*sl['successes']/sl['total']:.1f}%)")

    plot_success_rate(stats_first, stats_last)
    plot_failure_count(stats_first, stats_last)
