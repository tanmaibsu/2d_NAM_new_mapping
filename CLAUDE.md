# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DNA Origami-based Nucleic Acid Memory (dNAM) system. Encodes binary files into 8×10 DNA origami matrices (80 bits each) and recovers them despite physical readout errors using multi-tier error correction. Research-stage code — no CI/CD, no test framework.

## Common Commands

```bash
# Install dependencies (Python 3.7+, virtual env in dnam/)
pip install -r requirements.txt

# Generate parity mapping (run once per configuration)
python3 error_correction/generate_parity_mapping.py -pn 24 -pc 12

# Encode a file
python3 error_correction/encode.py -f input.txt -o output.txt -pn 24

# Decode single file
python3 error_correction/decode.py -f origamis_24/origami1.txt -o decoded_output -pn 24

# Decode in bulk (folder of origami files)
python3 error_correction/decode.py -bulk origamis_24 -o decoded_output -pn 24

# Exhaustive error injection testing
python3 error_correction/decode_exhaustive.py
```

Key CLI flags for decode: `-e` max errors, `-td` data threshold, `-tp` parity threshold, `-fp` false positive budget.

## Architecture

### Pipeline
```
Binary file → encode.py → Origami text files → (physical readout with errors) → decode.py → Recovered file
```

### Core Classes (error_correction/)

- **`origami_greedy.py` — `Origami` class**: Single 8×10 matrix encode/decode. Three-tier decoder:
  - Tier 0: Strict accept (no errors detected)
  - Tier 1: Beam-search syndrome decoder (primary, up to ~8 errors, beam width 6)
  - Tier 2: Legacy greedy heuristic (fallback, weight-based bit flipping)

- **`processfile.py` — `ProcessFile(Origami)` class**: Batch operations — file segmentation, threaded decoding (up to 32 workers via ThreadPoolExecutor), majority voting across redundant copies, CSV logging.

### Entry Points

| File | Purpose |
|------|---------|
| `encode.py` | Binary file → origami matrices |
| `decode.py` | Origami matrices → binary file (includes exhaustive single-bit test mode) |
| `decode_exhaustive.py` | Multi-bit error injection testing harness |

### Parity & Mapping

- **`get_parity_n_checksum.py`**: Static parity/checksum relation configs for 16/24/40 parity bits
- **`generate_parity_mapping.py`**: Dynamic mapping generator (rules: no repeating positions, no mirrored points, axis mirrors required, some shared positions across parities)
- **`parity_relations.py`**: LDPC-style relation optimizer
- **`origami_design.py`**: Parity position definitions within the matrix

### 8×10 Matrix Layout

80 bits total: ~29 data bits, 2-5 index bits, 4 orientation bits (fixed `[1,1,1,0]` at corners), 16/24/40 parity bits (XOR-based), 4 checksum bits (quadrant-level). The `parity_number` parameter (16, 24, or 40) controls the redundancy/capacity trade-off.

### Error Model

Asymmetric: false negatives (1→0) are common from physical readout; false positives (0→1) are rare. The decoder handles orientation recovery (4 physical rotations/flips) before error correction.

### Support Modules

- `log.py` — logging config
- `utility_methods.py` — helpers (bit flipping, etc.)
- `turbo_code/` — experimental turbo code implementation
