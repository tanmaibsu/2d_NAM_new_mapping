# headmart
"""
Generate an ASYMMETRIC parity mapping for dNAM with 8x10 origami.

Background
----------
The marker-based scheme built the parity mapping flip-SYMMETRIC on purpose:
it picked a handful of "unique" parity bits, then manufactured their x/y/xy
mirror copies (the old generate_all / generate_mirrors_across_axis). A
flip-symmetric mapping cannot tell a flipped origami from a correct one, so 4
orientation marker bits -- (1,0),(1,9),(6,0),(6,9) holding [1,1,1,0] -- were
reserved to disambiguate orientation at decode time.

This version does the opposite. Every parity bit gets an INDEPENDENT random
coverage set, so the mapping is NOT flip-symmetric. The resulting mapping is
then VALIDATED: for each of the 3 flips (flipud, fliplr, both) we measure how
many parity equations a flipped codeword violates. That violation count is the
safety margin the marker bits used to provide -- now bought from the mapping's
asymmetry instead. Because orientation is recoverable from parity consistency
alone, the 4 orientation cells are freed and become ordinary data bits.

The mapping is written as a Python dict literal that can be pasted into
get_parity_n_checksum.parity_mapping_24().
"""
import random
import argparse
from datetime import datetime
from functools import reduce

import numpy as np
import origami_design as od

# Origami Dimensions
ROW = 8
COLUMN = 10
FLIPS = ("ud", "lr", "both")


def read_n_parse_args():
    """Read arguments from the command line."""
    parser = argparse.ArgumentParser(
        description="Create an asymmetric, orientation-marker-free parity mapping."
    )
    parser.add_argument("-pn", "--parity_number", type=int, default=24,
                        help="Number of parity bits (16, 24, or 40).")
    parser.add_argument("-pc", "--parity_coverage", type=int, default=12,
                        help="Data cells covered by each parity bit. More coverage widens "
                             "the orientation margin but makes each parity bit more "
                             "sensitive to single read-errors.")
    parser.add_argument("-mn", "--min_margin", type=int, default=1,
                        help="Reject the mapping if any flip lets some sampled codeword "
                             "violate fewer than this many parity equations.")
    parser.add_argument("-me", "--mean_margin", type=float, default=None,
                        help="Reject if a flip's mean violation count falls below this. "
                             "Defaults to 0.35 * parity_number.")
    parser.add_argument("-s", "--samples", type=int, default=5000,
                        help="Random codewords used to estimate the flip margin.")
    parser.add_argument("-a", "--attempts", type=int, default=200,
                        help="Max regeneration attempts before giving up.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible mappings.")
    return parser.parse_args()


def define_parity_positions(number_of_parity: int):
    """Return the list of cells that hold parity bits."""
    if number_of_parity == 24:
        return list(od.parity_mapping_24())
    elif number_of_parity == 16:
        return list(od.parity_mapping_16())
    elif number_of_parity == 40:
        return list(od.parity_mapping_40())
    raise ValueError(f"Unsupported parity number: {number_of_parity}")


def create_matrix():
    """All cell coordinates of the origami."""
    return [(i, j) for i in range(ROW) for j in range(COLUMN)]


def non_parity_pool(parity_positions):
    """Cells a parity bit may cover: everything that is not itself a parity bit.

    The 4 former orientation cells are intentionally NOT special-cased -- in the
    marker-free scheme they are ordinary data cells and are eligible coverage
    like any other.
    """
    parity_set = set(parity_positions)
    return [cell for cell in create_matrix() if cell not in parity_set]


def flip_cell(transform, cell):
    """Map a cell to its position under a flip."""
    r, c = cell
    if transform == "ud":
        return (ROW - 1 - r, c)
    if transform == "lr":
        return (r, COLUMN - 1 - c)
    if transform == "both":
        return (ROW - 1 - r, COLUMN - 1 - c)
    return cell


def flip_matrix(transform, matrix):
    """Apply a flip to a 2-D matrix."""
    if transform == "ud":
        return np.flipud(matrix)
    if transform == "lr":
        return np.fliplr(matrix)
    if transform == "both":
        return np.flipud(np.fliplr(matrix))
    return matrix


def generate_asymmetric_mapping(parity_positions, pool, coverage):
    """Assign each parity bit an INDEPENDENT random coverage set.

    No mirror copies are manufactured, so the mapping is asymmetric by
    construction. Validation (below) confirms the margin is actually wide.
    """
    return {parity: random.sample(pool, coverage) for parity in parity_positions}


def constraints_preserved(mapping, transform):
    """How many parity equations survive a flip unchanged (0 == ideal).

    A fast structural check: with independent random coverage this is almost
    always 0, but a nonzero count is a hard red flag for accidental symmetry.
    """
    canon = {(k, frozenset(v)) for k, v in mapping.items()}
    preserved = 0
    for k, v in mapping.items():
        tk = flip_cell(transform, k)
        tv = frozenset(flip_cell(transform, s) for s in v)
        if (tk, tv) in canon:
            preserved += 1
    return preserved


def flip_violations(mapping, transform, samples):
    """Estimate the orientation margin for one flip.

    Build random valid codewords (set each parity bit = XOR of its coverage),
    flip them, and count how many parity equations the flipped matrix violates.
    Returns (min, mean) violation counts across the samples. The min is the
    worst-case safety margin; if it reaches 0 some data pattern is
    orientation-ambiguous under this flip.
    """
    worst = None
    total = 0
    items = list(mapping.items())
    for _ in range(samples):
        m = np.random.randint(0, 2, size=(ROW, COLUMN))
        for (pr, pc), cells in items:
            m[pr, pc] = reduce(lambda a, b: a ^ b, [m[r, c] for r, c in cells])
        f = flip_matrix(transform, m)
        viol = 0
        for (pr, pc), cells in items:
            xored = reduce(lambda a, b: a ^ b, [int(f[r, c]) for r, c in cells])
            if int(f[pr, pc]) != xored:
                viol += 1
        worst = viol if worst is None else min(worst, viol)
        total += viol
    return worst, total / samples


def validate(mapping, samples, min_margin, mean_margin):
    """Run the asymmetry gate over all 3 flips.

    Returns (passed, stats) where stats[transform] = (min, mean, preserved).
    """
    stats = {}
    passed = True
    for transform in FLIPS:
        mn, me = flip_violations(mapping, transform, samples)
        preserved = constraints_preserved(mapping, transform)
        stats[transform] = (mn, me, preserved)
        if preserved != 0 or mn < min_margin or me < mean_margin:
            passed = False
    return passed, stats


def generate_validated_mapping(parity_positions, pool, args, mean_margin):
    """Regenerate until a mapping clears the asymmetry gate."""
    for attempt in range(1, args.attempts + 1):
        mapping = generate_asymmetric_mapping(parity_positions, pool, args.parity_coverage)
        passed, stats = validate(mapping, args.samples, args.min_margin, mean_margin)
        if passed:
            return mapping, stats, attempt
        worst = ", ".join(
            f"{t}(min={stats[t][0]}, mean={stats[t][1]:.1f}, kept={stats[t][2]})"
            for t in FLIPS
        )
        print(f"attempt {attempt}: rejected -> {worst}")
    raise RuntimeError(
        "No mapping cleared the gate. Lower --min_margin/--mean_margin, "
        "raise --parity_coverage, or increase --attempts."
    )


def create_file_name(parity_number):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"parity_mapping_{parity_number}_asym_{timestamp}.txt"


def write_mapping(mapping, file_name, stats, args):
    """Write the mapping as a paste-ready Python dict, with the margins on top."""
    with open(file_name, "w") as fh:
        fh.write("# Asymmetric parity mapping -- no orientation markers required.\n")
        fh.write(f"# parity_number={len(mapping)}  parity_coverage={args.parity_coverage}  "
                 f"samples={args.samples}  seed={args.seed}\n")
        fh.write("# Orientation is recovered at decode time by picking the flip with the\n")
        fh.write("# fewest parity violations. Margins below (min/mean violations per flip):\n")
        for transform in FLIPS:
            mn, me, preserved = stats[transform]
            fh.write(f"#   flip {transform:>4}: min={mn:<3} mean={me:6.2f} "
                     f"constraints_preserved={preserved}\n")
        fh.write("{\n")
        for k, v in mapping.items():
            fh.write(f"    {k}: {v},\n")
        fh.write("}\n")


def main():
    args = read_n_parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    mean_margin = args.mean_margin
    if mean_margin is None:
        mean_margin = 0.35 * args.parity_number

    parity_positions = define_parity_positions(args.parity_number)
    pool = non_parity_pool(parity_positions)
    print(f"parity bits: {len(parity_positions)}   coverage pool: {len(pool)} cells   "
          f"coverage/bit: {args.parity_coverage}")
    print(f"gate: min_margin>={args.min_margin}, mean_margin>={mean_margin:.2f}, "
          f"constraints_preserved==0 for all flips")

    mapping, stats, attempt = generate_validated_mapping(
        parity_positions, pool, args, mean_margin
    )

    print(f"\nValidated asymmetric mapping found on attempt {attempt}:")
    for transform in FLIPS:
        mn, me, preserved = stats[transform]
        print(f"  flip {transform:>4}: min_violations={mn:<3} mean_violations={me:6.2f} "
              f"constraints_preserved={preserved}")

    file_name = create_file_name(args.parity_number)
    write_mapping(mapping, file_name, stats, args)
    print(f"\nMapping written to {file_name}")
    print("Paste its dict body into get_parity_n_checksum.parity_mapping_24().")


if __name__ == "__main__":
    main()
