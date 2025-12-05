import pandas as pd
from collections import Counter
from pathlib import Path
import argparse

def majority_vote_per_node(input_folder, output_folder):
    """
    Aggregates decoded streams from all CSV files in a folder and performs majority voting per node.
    - input_folder: directory containing CSVs
    - output_folder: directory where final CSV/TXT will be saved
    """
    folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_files = list(folder.glob("*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    combined = []
    for f in all_files:
        print(f"Reading {f.name} ...")
        df = pd.read_csv(f)

        # Try to detect node and decoded_stream columns
        node_col = next((c for c in ["node_id", "node", "Node", "node_index"] if c in df.columns), None)
        if node_col is None:
            raise KeyError(f"Could not find node column in {f.name}")

        stream_col = next((c for c in ["decoded_stream", "DecodedStream", "decoded", "decoded_data"] if c in df.columns), None)
        if stream_col is None:
            raise KeyError(f"Could not find decoded_stream column in {f.name}")

        df = df[[node_col, stream_col]].dropna()
        df.rename(columns={node_col: "node_id", stream_col: "decoded_stream"}, inplace=True)
        combined.append(df)

    # Combine all data from all CSVs
    full_df = pd.concat(combined, ignore_index=True)
    print(f"Total records loaded: {len(full_df)}")

    # Perform majority voting per node
    results = []
    for node, group in full_df.groupby("node_id"):
        counts = Counter(group["decoded_stream"])
        best_stream, best_votes = counts.most_common(1)[0]
        total = len(group)
        results.append({
            "node_id": node,
            "majority_decoded_stream": best_stream,
            "vote_count": best_votes
        })

    result_df = pd.DataFrame(results).sort_values("node_id").reset_index(drop=True)

    # Output paths
    output_csv = output_folder / "final_decoded_majority.csv"
    output_txt = output_folder / "final_decoded_majority.txt"

    # Save both CSV and TXT summary
    result_df.to_csv(output_csv, index=False)
    with open(output_txt, "w") as f:
        for _, row in result_df.iterrows():
            f.write(f"{row['majority_decoded_stream']}\n")

    print(f"\n✅ Majority-decoded results saved to:\n - {output_csv}\n - {output_txt}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform majority voting on decoded streams across multiple CSVs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the folder containing decoded CSV files."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the folder where output files will be saved."
    )

    args = parser.parse_args()
    majority_vote_per_node(args.input, args.output)
