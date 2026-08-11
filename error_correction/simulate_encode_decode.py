"""
Automated encode/decode simulation for the marker-free dNAM scheme.

For each trial it:
  1. generates a random binary data string + random index,
  2. encodes it into an 8x10 origami matrix,
  3. (optionally) applies a physical orientation flip,
  4. injects a chosen number of random single-cell bit errors,
  5. decodes and compares the recovery to the ground truth.

Outcomes are bucketed into three categories that matter for an
error-correcting code:

  CORRECT   -- decoder returned the right data AND right index.
  FALSE     -- decoder returned *something* but it was wrong (a silent
               miscorrection; the most dangerous failure mode).
  FAIL      -- decoder gave up (returned -1).

Run:
    python simulate_encode_decode.py                 # default sweep
    python simulate_encode_decode.py -t 300 -e 0 1 2 3 4 --flip
    python simulate_encode_decode.py --seed 1 -p 24 -d 29
"""
import os
import sys
import random
import argparse
import contextlib
from datetime import datetime

import numpy as np

import origami_greedy as og

ROW, COLUMN = 8, 10
FLIP_FUNCS = {
    0: lambda m: m,
    1: np.flipud,
    2: np.fliplr,
    3: lambda m: np.flipud(np.fliplr(m)),
}
FLIP_NAMES = {0: "none", 1: "flipud", 2: "fliplr", 3: "both"}


@contextlib.contextmanager
def silence():
    """Swallow the chatty print()/stdout from encode & decode."""
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        yield


def parse_args():
    p = argparse.ArgumentParser(description="Encode/decode Monte-Carlo simulation.")
    p.add_argument("-t", "--trials", type=int, default=200,
                   help="Trials per error level.")
    p.add_argument("-e", "--errors", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5],
                   help="Bit-error counts to sweep.")
    p.add_argument("-p", "--parity_number", type=int, default=24,
                   choices=[16, 24, 40], help="Parity scheme.")
    p.add_argument("-d", "--data_bits", type=int, default=29,
                   help="Data bits per origami (rest of the 52 become index bits).")
    p.add_argument("--flip", action="store_true",
                   help="Also apply a random orientation flip before injecting errors "
                        "(tests marker-free orientation recovery).")
    # decode tuning -- defaults mirror origami_greedy.__main__
    p.add_argument("--threshold_data", type=int, default=2)
    p.add_argument("--threshold_parity", type=int, default=3)
    p.add_argument("--max_error", type=int, default=5)
    # false_positive is the budget for "reverse" (1->0) corrections. The decoder
    # treats a 0 as a possible dropout (free to flip up to 1) but a 1 as high
    # confidence, so with false_positive=0 it can ONLY fix 0->1 errors and is
    # blind to ~half of all symmetric bit flips. Default it to max_error so the
    # decoder can correct in both directions up to its search depth. Set it to 0
    # to model the asymmetric dropout-only regime dNAM was originally tuned for.
    p.add_argument("--false_positive", type=int, default=None,
                   help="Reverse-flip (1->0) budget. Defaults to --max_error.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--show_failures", type=int, default=0,
                   help="Print details for up to N non-CORRECT trials.")
    return p.parse_args()


def random_binary(length):
    return "".join(random.choice("01") for _ in range(length))


def inject_errors(stream, num_errors):
    """Flip `num_errors` distinct random cells of the 80-cell stream."""
    if num_errors == 0:
        return stream, []
    bits = list(stream)
    positions = random.sample(range(len(bits)), num_errors)
    for pos in positions:
        bits[pos] = "1" if bits[pos] == "0" else "0"
    # report positions as (row, col) for readability
    locs = [(pos // COLUMN, pos % COLUMN) for pos in positions]
    return "".join(bits), locs


def run_trial(origami, args, num_errors, max_index):
    """One encode -> corrupt -> decode round-trip. Returns a result dict."""
    data = random_binary(args.data_bits)
    index = random.randrange(max_index)

    with silence():
        encoded = origami._encode(data, index, args.data_bits, args.parity_number)

    matrix = origami.data_stream_to_matrix(encoded)

    flip_opt = random.choice([0, 1, 2, 3]) if args.flip else 0
    matrix = np.array(FLIP_FUNCS[flip_opt](matrix))

    corrupted_stream, err_locs = inject_errors(
        origami.matrix_to_data_stream(matrix), num_errors)

    with silence():
        result = origami.decode(
            corrupted_stream,
            args.threshold_data, args.threshold_parity,
            args.max_error, args.false_positive)

    info = {"data": data, "index": index, "flip": flip_opt,
            "errors": err_locs, "result": result}

    if result == -1:
        info["outcome"] = "FAIL"
    elif result.get("binary_data") == data and result.get("index") == index:
        info["outcome"] = "CORRECT"
    else:
        info["outcome"] = "FALSE"
    return info


def main():
    args = parse_args()
    if args.false_positive is None:
        args.false_positive = args.max_error
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    origami = og.Origami(verbose=0)
    # Prime matrix_details / parity relations once so we know the index range.
    with silence():
        origami._encode(random_binary(args.data_bits), 0,
                        args.data_bits, args.parity_number)
    index_bit_count = len(origami.matrix_details["indexing_bits"])
    max_index = 2 ** index_bit_count

    print("=" * 70)
    print("dNAM encode/decode simulation")
    print(f"  parity_number   : {args.parity_number}")
    print(f"  data bits/origami: {args.data_bits}   index bits: {index_bit_count} "
          f"(max index {max_index - 1})")
    print(f"  orientation flips: {'ON (random)' if args.flip else 'off'}")
    print(f"  decode params   : threshold_data={args.threshold_data} "
          f"threshold_parity={args.threshold_parity} "
          f"max_error={args.max_error} false_positive={args.false_positive}")
    print(f"  trials/level    : {args.trials}     seed: {args.seed}")
    print("=" * 70)

    header = f"{'errors':>6} | {'CORRECT':>8} {'FALSE':>6} {'FAIL':>6} | {'accuracy':>9}"
    print(header)
    print("-" * len(header))

    shown_failures = 0
    grand = {"CORRECT": 0, "FALSE": 0, "FAIL": 0}

    for num_errors in args.errors:
        counts = {"CORRECT": 0, "FALSE": 0, "FAIL": 0}
        for _ in range(args.trials):
            info = run_trial(origami, args, num_errors, max_index)
            counts[info["outcome"]] += 1
            grand[info["outcome"]] += 1
            if info["outcome"] != "CORRECT" and shown_failures < args.show_failures:
                shown_failures += 1
                r = info["result"]
                got = "-1" if r == -1 else f"idx={r.get('index')} data_ok={r.get('binary_data')==info['data']}"
                print(f"   [{info['outcome']}] flip={FLIP_NAMES[info['flip']]} "
                      f"errors@{info['errors']} -> {got}")
        acc = 100.0 * counts["CORRECT"] / args.trials
        print(f"{num_errors:>6} | {counts['CORRECT']:>8} {counts['FALSE']:>6} "
              f"{counts['FAIL']:>6} | {acc:>8.1f}%")

    total = sum(grand.values())
    print("-" * len(header))
    print(f"{'TOTAL':>6} | {grand['CORRECT']:>8} {grand['FALSE']:>6} "
          f"{grand['FAIL']:>6} | {100.0*grand['CORRECT']/total:>8.1f}%")
    if grand["FALSE"]:
        print(f"\n  WARNING: {grand['FALSE']} silent miscorrection(s) "
              f"({100.0*grand['FALSE']/total:.2f}% of all trials).")


if __name__ == "__main__":
    main()
