# DNA Origami Error-Correcting Decoder — Detailed Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Physical Context](#2-physical-context)
3. [The 8x10 Origami Matrix Layout](#3-the-8x10-origami-matrix-layout)
4. [Encoding Pipeline](#4-encoding-pipeline)
5. [Error Model](#5-error-model)
6. [Decoding Pipeline (High-Level)](#6-decoding-pipeline-high-level)
7. [Orientation Recovery](#7-orientation-recovery)
8. [Parity & Checksum System](#8-parity--checksum-system)
9. [Decoder Tier 0 — Strict Accept](#9-decoder-tier-0--strict-accept)
10. [Decoder Tier 1 — Beam-Search Syndrome Decoder](#10-decoder-tier-1--beam-search-syndrome-decoder)
11. [Decoder Tier 2 — Legacy Greedy Heuristic Decoder](#11-decoder-tier-2--legacy-greedy-heuristic-decoder)
12. [Majority Voting & File Reconstruction](#12-majority-voting--file-reconstruction)
13. [Configuration Parameters](#13-configuration-parameters)
14. [Architecture & Class Hierarchy](#14-architecture--class-hierarchy)
15. [Worked Example](#15-worked-example)

---

## 1. Overview

This system encodes arbitrary binary files into a set of **DNA origami matrices**, each an 8x10 grid of binary values (80 bits). Each matrix carries a portion of the payload data along with redundancy in the form of **parity bits**, **checksum bits**, **orientation markers**, and **index bits**. After physical readout (which introduces bit-flip errors), the decoder attempts to recover the original matrices using a hybrid, multi-tier error-correction strategy.

**Key files:**

| File | Role |
|---|---|
| `origami_greedy.py` | Core `Origami` class — encoding, single-matrix decoding (all three tiers) |
| `processfile.py` | `ProcessFile(Origami)` — batch orchestration, threading, majority voting, CSV I/O |
| `decode.py` | CLI entry point — argument parsing, exhaustive testing harnesses, wetlab data import |
| `get_parity_n_checksum.py` | Static mappings: parity and checksum relations for 16/24/40 parity configurations |

---

## 2. Physical Context

A **3D DNA origami** nanostructure has discrete physical locations that can be probed to read a binary value (0 or 1). The readout process (e.g., via fluorescence microscopy or AFM) is inherently noisy:

- **False negatives (1→0)** are the dominant error type: a staple that *should* be present is missed during readout.
- **False positives (0→1)** are rarer but possible: noise causes a location to appear occupied when it is not.

This asymmetry is explicitly modeled in the decoder via the `false_positive` budget parameter.

---

## 3. The 8x10 Origami Matrix Layout

Each origami is represented as an 8-row by 10-column matrix (80 cells total). The cells are partitioned into five functional regions:

```
         Col 0   1   2   3   4   5   6   7   8   9
Row 0   [ d   d   d   d   d   d   d   d   d   d  ]   ← data / checksum-covered
Row 1   [ OR  p   p   p   p   p   p   p   p   OR ]   ← orientation corners + parity ring
Row 2   [ d  p   P   P   P   P   P   P   p   d ]   ← index + parity (P = parity cell)
Row 3   [ d  p   P   P   CS  CS  P   P   p   d ]   ← index + parity + checksum center
Row 4   [ d  p   P   P   CS  CS  P   P   p   d ]   ← same
Row 5   [ d   p   P   P   P   P   P   P   p   d  ]   ← data + parity
Row 6   [ OR  p   p   p   p   p   p   p   p   OR ]   ← orientation corners + parity ring
Row 7   [ d   d   d   d   d   d   d   d   idx   idx  ]   ← data
```

*(Exact positions vary by parity configuration; diagram is illustrative for the 24-parity case.)*

### Cell types

| Type | Symbol | Count (24-parity) | Purpose |
|---|---|---|---|
| **Data bits** | `d` | ~29 | Carry payload information |
| **Index bits** | `ix` | ~2-5 | Binary-encode the origami's sequence number (for reassembly) |
| **Orientation bits** | `OR` | 4 | Fixed pattern `[1, 1, 1, 0]` at corners `(1,0), (1,9), (6,0), (6,9)` — detect physical flips |
| **Parity bits** | `p/P` | 16, 24, or 40 | Each is the XOR of a specific subset of other cells |
| **Checksum bits** | `CS` | 4 | Each is the XOR of a quadrant of the matrix — coarse error detection |

The layout is determined dynamically by `_matrix_details()` based on the chosen `parity_number` (16, 24, or 40) and the data capacity needed.

---

## 4. Encoding Pipeline

```
Binary file
    │
    ▼
Split into segments (each ≤ data_bits_per_origami bits)
    │
    ▼  For each segment i:
┌─────────────────────────────────────────┐
│ 1. Place data bits into matrix          │
│ 2. Place orientation bits [1,1,1,0]     │
│ 3. Place index bits (binary of i)       │
│ 4. Compute checksum bits (XOR)          │
│ 5. Compute parity bits (XOR)            │
│ 6. Flatten to 80-char binary string     │
└─────────────────────────────────────────┘
    │
    ▼
Written to file (one 80-bit string per line)
```

**Checksum computation** (`_xor_matrix` with `checksum_bit_relation`):
Each of the 4 checksum cells stores the XOR of ~9-15 cells in its quadrant of the matrix. This gives a coarse "quadrant parity" check.

**Parity computation** (`_xor_matrix` with `parity_bit_relation`):
Each parity cell stores the XOR of ~12-16 data/index/orientation cells scattered across the matrix. The parity mapping is designed so that each data cell participates in multiple parity equations (similar to an LDPC code), enabling the decoder to localize errors by finding which parity checks fail.

### Index bit optimization

The number of index bits is minimized by `_find_optimum_index_bits()`: it finds the smallest number of index bits `i` such that `2^i ≥ number_of_segments`, maximizing the remaining capacity for data bits.

---

## 5. Error Model

The decoder is designed to handle the following error types:

| Error type | Physical cause | Frequency | Decoder treatment |
|---|---|---|---|
| **False negative (1→0)** | Missing staple, readout miss | Common | Default flip direction; no budget limit |
| **False positive (0→1)** | Noise, contamination | Rare | Controlled by `false_positive` budget |
| **Orientation flip** | Origami physically rotated/flipped during imaging | Occasional | Tried exhaustively (4 orientations) |
| **Missing origami** | Origami not read at all | Occasional | Handled by majority voting across redundant copies |

The `false_positive` parameter limits how many 1→0 corrections (i.e., flipping a 1 back to 0) the legacy decoder will attempt. This reflects the physical prior that most errors are 0-for-1 substitutions.

---

## 6. Decoding Pipeline (High-Level)

```
Noisy 80-bit binary string
    │
    ▼
Convert to 8x10 matrix
    │
    ▼
╔══════════════════════════════════════════════╗
║  TIER 0: Strict Accept                       ║
║  Try all 4 orientations. If orientation +    ║
║  ALL parity + ALL checksum pass → accept.    ║
╚══════════════════════════════════════════════╝
    │ (failed)
    ▼
╔══════════════════════════════════════════════╗
║  TIER 1: Beam-Search Syndrome Decoder        ║
║  Iterative bit-flipping guided by syndrome.  ║
║  Beam width = 6, max flips = max_errors,     ║
║  max iterations = 40.                        ║
║  Accept only if strict check passes.         ║
╚══════════════════════════════════════════════╝
    │ (failed)
    ▼
╔══════════════════════════════════════════════╗
║  TIER 2: Legacy Greedy Heuristic Decoder     ║
║  Weight-based scoring, greedy search over    ║
║  error combinations with false_positive      ║
║  budget. Accept only if strict check passes. ║
╚══════════════════════════════════════════════╝
    │ (failed)
    ▼
Return -1 (decoding failure)
```

All three tiers enforce **strict acceptance**: a result is returned only if orientation is fixable, all parity checks pass, and all checksum checks pass. This eliminates silent miscorrections.

---

## 7. Orientation Recovery

The origami can be physically read in any of 4 orientations:

| Option | Transform | Description |
|---|---|---|
| 0 | Identity | Correct orientation |
| 1 | `flipud` | Flipped vertically (top↔bottom) |
| 2 | `fliplr` | Flipped horizontally (left↔right) |
| 3 | `flipud(fliplr)` | Flipped both ways (180° rotation) |

**Detection method** (`_fix_orientation`):
The 4 orientation bits at positions `(1,0), (1,9), (6,0), (6,9)` are encoded as `[1, 1, 1, 0]`. The decoder tries each of the 4 flips and checks if the orientation bits match. The first match is accepted.

When errors are corrected, the error locations must be mapped back to the original (pre-flip) coordinate system using `_mirror_locations()`.

---

## 8. Parity & Checksum System

### Parity bits (inner code)

Each parity bit cell `(r, c)` is the XOR of a set of ~12-16 other cells:

```
parity_cell = XOR(dep_1, dep_2, ..., dep_k)
```

The mapping is defined in `get_parity_n_checksum.py` and comes in three configurations:
- **16-parity**: Positions in rows 2-5, columns 2-7. Each covers 16 cells.
- **24-parity**: Positions in rows 1-6, columns 1-8. Each covers 12 cells.
- **40-parity**: Most redundancy; covers the parity cells from both 16 and 24 configurations.

**Syndrome**: When a parity check fails, it means an odd number of its dependent cells have been flipped. The set of *failed* parity checks is the syndrome, and it localizes which cells are likely in error.

### Checksum bits (outer code)

4 checksum cells at `(3,4), (3,5), (4,4), (4,5)` each cover a quadrant of the matrix (~9-15 cells). They provide a coarse, independent check: even if a combination of errors happens to satisfy all parity equations (which is unlikely but possible), the checksum provides an additional layer of verification.

### Inverse mapping

`get_data_bit_to_parity_bit()` builds the reverse index: for each data cell, which parity cells depend on it. This is critical for the scoring function — when a data cell is suspected of being in error, the decoder can check how many of its associated parity checks are failing.

---

## 9. Decoder Tier 0 — Strict Accept

**Location**: `_decode()` → `_strict_matrix_ok()`

The simplest path: if the received matrix (after trying all 4 orientations) passes all parity and all checksum checks, it is accepted with zero corrections.

```python
def _strict_matrix_ok(self, matrix):
    ori, oriented = self._fix_orientation(matrix)  # try 4 flips
    if ori == -1: return False                      # no orientation matched
    _, bad_parity = self._find_possible_error_location(oriented)
    if bad_parity: return False                     # parity failures exist
    if not self.check_checksum(oriented): return False  # checksum failures
    return True
```

**When it works**: Error-free readouts, or errors that happen to cancel each other out in all check equations (extremely rare).

---

## 10. Decoder Tier 1 — Beam-Search Syndrome Decoder

**Location**: `_iterative_decode()`

This is the primary error-correction engine, inspired by **bit-flipping LDPC decoding** augmented with **beam search**.

### Algorithm

```
Input: noisy matrix M, max_flips, max_iters, beam_width
beam ← [(M, [])]   # (matrix, list_of_flips)

for iteration in 1..max_iters:
    next_candidates ← empty

    for each (mat, flips) in beam:
        if strict_ok(mat): return (mat, flips)     # success!

        failed ← set of failed parity+checksum checks
        scores ← Counter()
        for each failed check ch:
            for each variable v in ch:
                scores[v] += 1                       # suspicion score

        for each top-scoring variable v (up to beam_width * 2):
            if len(flips) < max_flips:
                nxt ← copy(mat) with v flipped
                next_candidates.append((nxt, flips + [v]))

    # Rank candidates: fewer failed checks = better; fewer flips = tiebreaker
    sort next_candidates by (num_failed_checks, num_flips)
    beam ← top beam_width candidates

return failure
```

### Intuition

1. **Syndrome computation**: Check which parity/checksum equations fail. Each failed equation implicates all the variables (cells) that participate in it.

2. **Suspicion scoring**: A cell that appears in *many* failed checks is more likely to be the actual error. This is the key insight from belief-propagation / bit-flipping decoders.

3. **Beam search**: Instead of greedily committing to the single most suspicious cell (which can get stuck in local minima), the decoder maintains a beam of the `beam_width` best partial solutions. This explores multiple correction paths in parallel.

4. **Strict validation**: Every candidate is checked against the strict acceptance criterion. The first candidate that passes all parity + checksum + orientation checks is returned.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `max_flips` | = `maximum_number_of_error` (typically 8) | Maximum bit flips per candidate path |
| `max_iters` | 40 | Maximum search iterations |
| `beam_width` | 6 | Number of candidate paths maintained |


---

## 11. Decoder Tier 2 — Legacy Greedy Heuristic Decoder

**Location**: `_decode_legacy()` → `_get_matrix_weight()`

A fallback decoder that uses a hand-crafted **weighted scoring system** to evaluate and search for error corrections.

### Scoring function (`_get_matrix_weight`)

Given a matrix and a set of proposed bit flips, this function:

1. **Flips the specified bits** in a copy of the matrix.
2. **Checks all parity equations** — collects the cells implicated by failed checks.
3. **Checks all checksum equations** — collects additional implicated cells.
4. **Assigns weights** to each implicated cell:

```
weight(cell) = parity_implication_count
             + 2  if cell is both a checksum cell AND in a failed checksum's coverage
             + 1  if cell is in either (but not both) of the above
```

   Cells with higher weights are more suspicious.

5. **Applies false-positive budget**: For 1→0 flips (correcting a false positive), the decoder limits these based on:
   - `fp_data_limit = (false_positive + 1) // 2` for data cells
   - `fp_parity_limit = false_positive // 2` for parity cells

6. **Returns**: the modified matrix, a normalized weight score, and the list of cells exceeding the threshold.

### Search strategy

```
1. Evaluate the matrix with no flips → get initial weight and probable errors

2. Try flipping each single probable error:
   - If weight drops to 0 AND strict check passes → return solution

3. Sort single-flip results by weight (ascending)

4. For each starting flip, iteratively try adding more flips:
   - At each step, pick the candidate(s) with lowest weight
   - Keep a retry queue of partial solutions
   - Continue until max_errors reached or no progress

5. At every step, if weight == 0 → strict check → accept or continue
```

### The threshold parameters

| Parameter | Controls | Typical value |
|---|---|---|
| `threshold_data` | Minimum suspicion weight for a data cell to be considered a candidate error | 2 |
| `threshold_parity` | Minimum suspicion weight for a parity cell to be considered a candidate error | 2 |

Higher thresholds make the decoder more conservative (fewer candidates, faster, but may miss errors). Lower thresholds make it more aggressive (more candidates, slower, but catches more errors).

### Strengths and limitations

- **Strengths**: Can handle error patterns that the syndrome decoder misses; considers false-positive asymmetry explicitly.
- **Limitations**: Greedy search can get stuck; combinatorial blowup for many errors; slower than the syndrome decoder.

---

## 12. Majority Voting & File Reconstruction

**Location**: `ProcessFile.decode()`

At the batch level, multiple noisy copies of each origami may be available (physical redundancy). The reconstruction proceeds as:

1. **Decode each origami independently** (threaded, up to 32 workers).
2. **Group decoded results by index** (the origami's sequence number).
3. **Majority vote**: For each index position, take the most common decoded binary string among all copies that decoded to that index.
4. **Reassemble**: Concatenate the voted binary strings in index order.
5. **Convert to bytes** and write the recovered file.

If any index position has no successful decodes, the reconstruction fails and returns the list of missing origami indices.

---

## 13. Configuration Parameters

### CLI arguments (`decode.py`)

| Flag | Parameter | Default | Description |
|---|---|---|---|
| `-e` | `maximum_number_of_error` | 8 | Max bit flips the decoder will attempt per origami |
| `-td` | `threshold_data` | 2 | Min suspicion weight to consider a data cell as an error candidate |
| `-tp` | `threshold_parity` | 2 | Min suspicion weight to consider a parity cell as an error candidate |
| `-fp` | `false_positive` | 1 | Budget for 1→0 corrections (false positive tolerance) |
| `-pn` | `parity_number` | 40 | Parity configuration: 16, 24, or 40 parity bits |
| `-fz` | `file_size` | 20 | Expected file size in bytes |
| `-r` | `redundancy` | 50 | Redundancy percentage used during encoding |
| `-v` | `verbose` | 0 | Logging level: 0=error, 1=debug, 2=info, 3=warning |

### Internal decoder parameters (hardcoded in `_decode`)

| Parameter | Value | Location |
|---|---|---|
| `beam_width` | 6 | `_decode()` call to `_iterative_decode()` |
| `max_iters` | 40 | `_iterative_decode()` |

---

## 14. Architecture & Class Hierarchy

```
Origami (origami_greedy.py)
│
│  Core responsibilities:
│  ├── Encoding: create_initial_matrix → _xor_matrix (checksum) → _xor_matrix (parity)
│  ├── Orientation: _fix_orientation, _mirror_locations
│  ├── Validation: _find_possible_error_location, check_checksum, _strict_matrix_ok
│  ├── Tier 1 decoder: _iterative_decode (beam-search syndrome)
│  ├── Tier 2 decoder: _decode_legacy + _get_matrix_weight (greedy heuristic)
│  ├── Hybrid decoder: _decode (orchestrates Tier 0 → 1 → 2)
│  └── Public API: encode(), decode()
│
└── ProcessFile (processfile.py) extends Origami
    │
    │  Batch responsibilities:
    │  ├── File I/O: read binary file, write recovered file
    │  ├── Segmentation: _find_optimum_index_bits
    │  ├── Threaded decoding: single_origami_decode, ThreadPoolExecutor
    │  ├── Majority voting: Counter().most_common()
    │  ├── CSV logging: write_ior_csv, _append_data_4_io
    │  └── Public API: encode(file), decode(batch)
    │
    └── decode.py (CLI entry point)
        ├── Argument parsing
        ├── Exhaustive test harnesses (single/double/triple bit flips)
        └── Wetlab data import and decoding
```

---

## 15. Worked Example

### Encoding

Suppose we want to store the byte `0x48` (`H` in ASCII = `01001000`) using 24-parity configuration:

1. `01001000` is 8 bits. With 29 data bits per origami, this fits in 1 segment.
2. `index = 0` → index bits = `0` (1 bit needed since only 1 segment).
3. Data bits `01001000` are placed at the designated data cell positions.
4. Remaining data positions are padded with `0`.
5. Orientation bits `[1, 1, 1, 0]` placed at `(1,0), (1,9), (6,0), (6,9)`.
6. 4 checksum bits computed as XOR of their respective quadrants.
7. 24 parity bits computed as XOR of their respective dependency sets.
8. Matrix flattened to an 80-character string.

### Decoding (with 2 errors)

Suppose positions `(0, 3)` and `(2, 7)` were flipped (1→0) during readout:

1. **Tier 0**: Strict accept fails — some parity checks are violated.
2. **Tier 1** (beam-search):
   - Iteration 1: 5 parity checks and 1 checksum check fail.
   - Both `(0, 3)` and `(2, 7)` appear in 3+ failed checks each → high suspicion scores.
   - Top candidates: flip `(0, 3)` (score=4), flip `(2, 7)` (score=3).
   - After flipping `(0, 3)`: 2 failed checks remain, `(2, 7)` now has highest score.
   - After flipping `(2, 7)`: 0 failed checks. Strict accept passes.
   - Return corrected matrix with `flips = [(0, 3), (2, 7)]`.
3. **Tier 2**: Not reached (Tier 1 succeeded).

### Reconstruction

If there are 4 redundant copies of this origami and 3/4 decode successfully with the same binary data, the majority vote confirms the result.
