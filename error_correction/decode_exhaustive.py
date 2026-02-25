import argparse
from processfile import ProcessFile
from process_results import compute_position_error_numbers_from_ior
import os
from pathlib import Path
import itertools
import csv
import cProfile
import pstats
import re


def read_args():
    """
    Parse command line arguments for the exhaustive decoder.  This largely mirrors the
    arguments available to the existing `decode.py` but adds options to control
    exhaustive bit‑flip testing.
    """
    parser = argparse.ArgumentParser(description="Decode a given origami matrices to a text file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-bulk", "--bulk_folder", help="Folder to decode", default="")
    group.add_argument("-f", "--file_in", help="File to decode")

    parser.add_argument("-o", "--file_out", help="File to write output", required=True)
    parser.add_argument("-fz", "--file_size", help="File size that will be decoded", type=int, default=20)
    parser.add_argument("-pn", "--parity_number", help="Number of Parity to decode", type=int, default=40)
    parser.add_argument("-tp", "--threshold_parity",
                        help='Minimum weight for a parity bit cell to be consider that as an error', default=2, type=int)
    parser.add_argument("-td", "--threshold_data",
                        help='Minimum weight for a data bit cell to be consider as an error', default=2, type=int)
    parser.add_argument("-v", "--verbose", help="Print details on the console. "
                                                "0 -> error, 1 -> debug, 2 -> info, 3 -> warning", default=0, type=int)
    parser.add_argument("-r", "--redundancy", help="How much redundancy was used during encoding",
                        default=50, type=float)
    parser.add_argument("-ior", "--individual_origami_info", help="Store individual origami information",
                        action='store_true', default=True)
    parser.add_argument("-e", "--error", help="Maximum number of error that the algorithm "
                                              "will try to fix", type=int, default=8)
    parser.add_argument("-fp", "--false_positive", help="0 can also be 1.", type=int, default=1)

    parser.add_argument("-d", "--degree", help="Degree old/new", default="new", type=str)
    parser.add_argument("--exhaustive", choices=["none", "1", "2", "3", "12", "13", "23", "123"], default="none",
                        help="Run exhaustive n-choose-k tests (flip 1->0 only). Produces CSVs of failing combos and per-bit votes.")
    parser.add_argument("--exhaustive_report_prefix", default="exhaustive_report",
                        help="Prefix for exhaustive CSV outputs (creates <prefix>_failures.csv, <prefix>_votes.csv and <prefix>_errors.csv).")
    parser.add_argument("--exhaustive_max_origami", type=int, default=None,
                        help="Optional limit on how many origami files to test (debugging).")

    parser.add_argument("-cf", "--correct_file", help="Original encoded file. Helps to check the status automatically.",
                        type=str, default=False)

    return parser.parse_args()


def create_file_name(args):
    """Create an IOR CSV file header if individual origami info is requested."""
    ior_file_name = f"{args.file_out}_ior.csv" if args.individual_origami_info else None
    if ior_file_name:
        try:
            with open(ior_file_name, "a") as ior_file:
                ior_file.write(
                    "node, origami data, Error positions, decoded stream, success, decoding time\n"
                )
        except Exception:
            # Logging is not available here; silently ignore
            pass


def convert_to_single_arr(data):
    """
    Convert a list of strings (rows of a matrix) into a single concatenated binary string.  If the
    list already contains a single string, return it as is.
    """
    if len(data) == 1:
        return data
    single_data = ""
    for row in data:
        for elm in row:
            if elm in ("0", "1"):
                single_data += str(elm)
    return [single_data]


def node_from_filename(path) -> int:
    """
    Supports: origami0.txt, origami1.txt, origami2.txt, origami3.txt
    """
    m = re.search(r"origami(\d+)", path.stem)  # stem = filename without extension
    if not m:
        raise ValueError(f"Cannot parse node from filename: {path.name}")
    return int(m.group(1))



def do_exhaustive_choose_tests(dnam_decode: ProcessFile, folder: Path, choose_spec: str = "123",
                               max_origami: int = None, report_prefix: str = "exhaustive_report",
                               args=None):
    """
    Exhaustively test n-choose-k bit flips (1->0 only) for k in choose_spec (e.g., "1", "23", "123").

    For each origami file in the provided folder:

      * For each requested k (1, 2, or 3), enumerate all combinations of k bit positions where the original
        bit is '1'.  Flip those bits to '0' in a copy of the origami and decode the mutated origami.  The
        decoding uses the provided ProcessFile instance.

      * Record any combination that fails to decode correctly (status != 1) in `<prefix>_failures.csv`.
        Each row of this file contains the node index, the k value, the list of indices flipped, and
        statistics returned by the decoder.

      * For failing pairs and triplets, tally a “vote” for each participating bit.  A summary of these
        per-bit votes (for pairs and triplets only) is written to `<prefix>_votes.csv` sorted by severity.

      * For every failing combination (singles, pairs or triplets), count how many times that specific
        combination caused a failure across nodes and record the set of nodes involved.  At the end of
        processing, this summary is written to `<prefix>_errors.csv`.  Columns are: k, indices, error_count,
        and nodes (semicolon-separated list of node indices).

    The function prints a summary of the testing statistics when complete.
    """

    # Determine ks to test
    ks = sorted({int(ch) for ch in choose_spec if ch in {"1", "2", "3"}})
    if not ks:
        print("[exhaustive] No valid k in choose_spec; nothing to do.")
        return

    # failures_path = f"{report_prefix}_failures.csv"
    # votes_path = f"{report_prefix}_votes.csv"
    # errors_path = f"{report_prefix}_errors.csv"

    # # votes[bit] = {"pair": count, "triplet": count}
    # votes: dict[int, dict[str, int]] = {}
    # # error_counts[k][indices_tuple] = {"count": int, "nodes": set}
    # error_counts: dict[int, dict[tuple, dict[str, any]]] = {1: {}, 2: {}, 3: {}}
    # total_tests = {1: 0, 2: 0, 3: 0}
    # total_failures = {1: 0, 2: 0, 3: 0}

    # def bump_vote(bit_idx: int, kind: str):
    #     """Increment vote count for a particular bit in pair or triplet context."""
    #     if bit_idx not in votes:
    #         votes[bit_idx] = {"pair": 0, "triplet": 0}
    #     votes[bit_idx][kind] += 1

    # Prepare CSV writers
 
    orig_idx = 0
    # Iterate over origami files
    for origami_file in sorted(folder.iterdir()):
        if max_origami is not None and orig_idx >= max_origami:
            break
        if origami_file.is_dir():
            continue
        
        node = node_from_filename(origami_file)
    
        # Read and flatten
        with open(origami_file, "r") as df:
            lines = df.readlines()
        origami_data = convert_to_single_arr(lines)
        if not origami_data:
            orig_idx += 1
            continue
        original = origami_data[0]
        # Only consider positions where bit == '1'
        one_positions = [i for i, b in enumerate(original) if b == "1"]

        for k in ks:
            if len(one_positions) < k:
                continue
            for combo in itertools.combinations(one_positions, k):
                print("<---error positions--->", combo)
                # total_tests[k] += 1
                # Flip bits in combo
                mut_list = list(original)
                for idx_bit in combo:
                    mut_list[idx_bit] = "0"
                mutated = "".join(mut_list)
                # Decode mutated origami. Use dummy file_out to avoid writing output.
                # print("<------original_origami-------->", list(original))
                print("<--------induced_errors------->", list(combo))
                status, incorrect_count, correct_count, total_error_fixed, _ = dnam_decode.decode(
                    [mutated],
                    original,
                    node,
                    induced_errors=";".join(map(str, combo)),
                    errors_positions=[list(combo)],
                    file_out=args.file_out,
                    file_size=args.file_size,
                    parity_number=int(args.parity_number),
                    threshold_data=args.threshold_data,
                    threshold_parity=args.threshold_parity,
                    maximum_number_of_error=args.error,
                    individual_origami_info=False,
                    false_positive=args.false_positive,
                    correct_file=args.correct_file,
                    accumulate=True,
                    write_csv=True,
                )
        orig_idx += 1
              
    compute_position_error_numbers_from_ior(
        ior_csv_path=f"{args.file_out}_ior.csv",
        out_prefix="errors_per_position_pairs"
    )
        


def import_original_origami_list():
    """Return a fixed list of example origami strings (unused in exhaustive testing)."""
    return [
        "01000100011100011101110000100111101010000111000010011001000011010000100001101000",
        "01011100111111110001000100000000111101000010101110111100010011000011000000011001",
        "11110111011111101001001011100011000001100000100001000010001110000011000011100110",
        "00000100101000010101000010000010010100000100100000000000000010000111100000000011",
    ]


def import_original_origami_list_6_nodes():
    """Return a fixed list of six-node origami strings (unused in exhaustive testing)."""
    return [
        "01000100011010110111101010111001101001010101011011011010010110110101100001100000",
        "00100100001110110111000101110010110101110000111110111110111110011001101001100001",
        "10000001101001011001100011010000110000111110011001101010110011011101100010000010",
        "00110111101100111101110000000111101110001000011011110101101010011011000100010011",
        "00000100011111011101010100010001010101101011101100011010101110010110000000010100",
        "01000010001110110011000000010101111100100110100100000011001011110011000000000101",
    ]


def decode_encoded_wetlab_data(args: argparse.Namespace, dnam_decode: ProcessFile):
    """
    Example helper to decode a wet‑lab CSV input.  Not used for exhaustive tests but retained
    to mirror the original decode.py functionality.  Reads a fixed CSV and decodes each row.
    """
    file_path = "encoded_6_nodes_wetlab/2025-11-26_mixed_6_nodes_rep_3.csv"
    original_origami_list = import_original_origami_list_6_nodes()
    # Read CSV rows
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"[wetlab] File not found: {file_path}")
        return
    for node_id in range(len(original_origami_list)):
        print(f"\n--- Processing Node {node_id} ---")
        node_rows = [row for row in rows if row.get("ID") and int(float(row["ID"])) == node_id]
        for row in node_rows:
            binary_string = row.get("Binary String", "")
            if binary_string.startswith("b"):
                origami_data = binary_string[1:]
            else:
                origami_data = binary_string
            false_negatives = int(row.get("False Negatives", 0))
            false_positives = int(row.get("False Positives", 0))
            if false_negatives + false_positives > 9:
                continue
            dnam_decode.decode(
                [origami_data],
                original_origami_list[node_id],
                node_id,
                [],
                [],
                args.file_out,
                args.file_size,
                int(args.parity_number),
                threshold_data=args.threshold_data,
                threshold_parity=args.threshold_parity,
                maximum_number_of_error=args.error,
                false_positive=args.false_positive,
                individual_origami_info=args.individual_origami_info,
                correct_file=args.correct_file,
                false_negatives=false_negatives,
                false_positives=false_positives,
            )


def main():
    args = read_args()
    # Instantiate ProcessFile once; verbose level comes from args
    dnam_decode = ProcessFile(verbose=args.verbose)
    create_file_name(args)
    # Run exhaustive tests if requested
    if args.exhaustive != "none":
        folder = Path(args.bulk_folder)
        do_exhaustive_choose_tests(
            dnam_decode,
            folder=folder,
            choose_spec=args.exhaustive,
            max_origami=args.exhaustive_max_origami,
            report_prefix=args.exhaustive_report_prefix,
            args=args,
        )
        return
    # Otherwise process wet‑lab CSV (mimicking original behaviour)
    decode_encoded_wetlab_data(args, dnam_decode)


if __name__ == '__main__':
    with cProfile.Profile() as profile:
        main()
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats()