# parity_relations.py
# ---------------------------------------------------------
# Drop-in optimizer for parity relations on an 8x10 (or any RxC) grid.
# It returns: { parity_pos (r,c) : [ (r1,c1), (r2,c2), ... ] }
# Compatible with your current random generator outputs.

from collections import defaultdict
import random, math

# ========== PUBLIC API ===================================

def generate_parity_relations(layout_roles, *, target_dv=3, iters=60000, seed=42,
                              use_optimizer=True, protect_roles=("data",),
                              allow_geometry_penalty=False, geo_weight=0.0):
    """
    Args
    ----
    layout_roles : dict[(r,c)->str]
        For every (row,col) cell, a role: one of
        {"data","parity","checksum","index","orientation", ...}
        (You can add roles; only those in `protect_roles` are treated as protected "data" to cover.)
    target_dv : int
        Target degree (how many parity checks cover each protected data bit).
    iters : int
        Annealing iterations (higher => better but slower).
    use_optimizer : bool
        If False, falls back to degree-balanced random generator.
    protect_roles : tuple[str]
        Which roles must be protected by parity checks (default: only "data").
        Include "parity" here if you want parity bits to protect each other.
    allow_geometry_penalty : bool
        If True, adds distance penalty so checks prefer local coverage.
    geo_weight : float
        Weight for the geometry penalty (0.0–1.0 typical).

    Returns
    -------
    parity_dict : dict[(r,c) -> list[(r_i,c_i)]]
        Mapping from *parity cell* to the list of protected cells it covers.
    meta : dict
        {"score": float, "Nd": int, "M": int, "dv": int, "dc": float}
    """

    data_positions = [pos for pos, role in layout_roles.items() if role in protect_roles]
    parity_positions = [pos for pos, role in layout_roles.items() if role == "parity"]

    if not parity_positions:
        raise ValueError("No 'parity' cells found in layout_roles.")
    if not data_positions:
        raise ValueError("No protected data cells found. Check 'protect_roles'.")

    Nd, M = len(data_positions), len(parity_positions)
    target_dc = max(1, round(Nd * target_dv / M))

    if not use_optimizer:
        rows = _random_regular_edges(Nd, M, target_dv, rng=random.Random(seed))
        Hcols, rows, _, _ = _build_H(data_positions, parity_positions, rows)
        return _rows_to_dict(rows, parity_positions, data_positions), {
            "score": _score(Hcols, rows, target_dv, target_dc,
                            allow_geometry_penalty, geo_weight, data_positions, parity_positions),
            "Nd": Nd, "M": M, "dv": target_dv, "dc": sum(len(r) for r in rows)/M
        }

    best_dict, best_score, rows = _optimize(
        data_positions, parity_positions,
        target_dv=target_dv, iters=iters, seed=seed,
        allow_geometry_penalty=allow_geometry_penalty, geo_weight=geo_weight
    )

    return best_dict, {"score": best_score, "Nd": Nd, "M": M, "dv": target_dv,
                       "dc": sum(len(r) for r in rows)/M}

# ========== INTEGRATION HELPERS ==========================

def merge_with_existing_roles(matrix_8x10, role_palette):
    """
    Convenience: build layout_roles from a 2D matrix and a legend.

    matrix_8x10 : list[list[str|int]]
        E.g., each cell contains one of {"data","parity","checksum","index","orientation"}.
        If your grid contains numbers (0/1), pass `role_palette` mapping values to role strings.
    role_palette : dict[Any->str]
        Maps grid cell values to role names, e.g. {0:"data", 1:"parity"} or similar.

    Returns: dict[(r,c)->role]
    """
    layout = {}
    for r, row in enumerate(matrix_8x10):
        for c, v in enumerate(row):
            layout[(r, c)] = role_palette.get(v, v if isinstance(v, str) else "data")
    return layout

# ========== CORE OPTIMIZER ===============================

def _optimize(data_positions, parity_positions, *, target_dv=3, iters=60000, seed=0,
              allow_geometry_penalty=False, geo_weight=0.0):
    rng = random.Random(seed)
    Nd, M = len(data_positions), len(parity_positions)
    target_dc = max(1, round(Nd * target_dv / M))

    rows = _random_regular_edges(Nd, M, target_dv, rng)
    Hcols, rows, _, _ = _build_H(data_positions, parity_positions, rows)
    best_rows = [set(s) for s in rows]
    best_score = _score(Hcols, rows, target_dv, target_dc,
                        allow_geometry_penalty, geo_weight, data_positions, parity_positions)

    temp0, tempf = 1.0, 0.01
    cur_score = best_score

    for t in range(1, iters + 1):
        T = temp0 * (tempf / temp0) ** (t / iters)

        r = rng.randrange(M)
        if not rows[r]:
            continue
        out_c = rng.choice(tuple(rows[r]))
        in_c = rng.randrange(Nd)
        if in_c in rows[r]:  # avoid duplicate edge
            continue

        # propose move
        rows[r].remove(out_c); rows[r].add(in_c)
        Hcols2, rows2, _, _ = _build_H(data_positions, parity_positions, rows)
        s2 = _score(Hcols2, rows2, target_dv, target_dc,
                    allow_geometry_penalty, geo_weight, data_positions, parity_positions)
        ds = s2 - cur_score
        accept = (ds >= 0) or (rng.random() < math.exp(ds / max(T, 1e-9)))

        if accept:
            Hcols, rows = Hcols2, rows2
            cur_score = s2
            if s2 > best_score:
                best_score = s2
                best_rows = [set(s) for s in rows]
        else:
            # revert
            rows[r].remove(in_c); rows[r].add(out_c)

    best_dict = _rows_to_dict(best_rows, parity_positions, data_positions)
    return best_dict, best_score, best_rows

# ========== SCORING ======================================

def _build_H(data_positions, parity_positions, edges_by_row):
    M = len(parity_positions)
    Nd = len(data_positions)
    rows = [set(s) for s in edges_by_row]
    Hcols = [0] * Nd
    for r, cols in enumerate(rows):
        for i in cols:
            Hcols[i] |= (1 << r)
    col_deg = [bin(Hcols[i]).count("1") for i in range(Nd)]
    row_deg = [len(rows[r]) for r in range(M)]
    return Hcols, rows, col_deg, row_deg

def _uniq1_score(Hcols):
    nonzero = [h for h in Hcols if h != 0]
    return (len(set(nonzero)) / len(Hcols)) if Hcols else 0.0

def _uniq2_score(Hcols, sample_pairs=2000, rng=random):
    Nd = len(Hcols)
    if Nd < 2: return 0.0
    singles = set(Hcols)
    seen_pairs = set()
    ok = 0; tot = 0
    for _ in range(sample_pairs):
        i = rng.randrange(Nd); j = rng.randrange(Nd)
        if i == j: continue
        s = Hcols[i] ^ Hcols[j]
        tot += 1
        # valid if pair syndrome is nonzero and not colliding with singles or other pairs seen
        if s != 0 and s not in singles and s not in seen_pairs:
            ok += 1
            seen_pairs.add(s)
    return ok / tot if tot else 0.0

def _count_four_cycles(rows):
    overlaps = 0
    R = len(rows)
    for a in range(R):
        a_set = rows[a]
        for b in range(a + 1, R):
            if len(a_set.intersection(rows[b])) > 1:
                overlaps += 1
    return overlaps

def _rank_gf2(rows_bits, width):
    rows = rows_bits[:]
    rank = 0; col = width - 1
    while col >= 0 and rank < len(rows):
        pivot = next((r for r in range(rank, len(rows)) if (rows[r] >> col) & 1), None)
        if pivot is None:
            col -= 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and ((rows[r] >> col) & 1):
                rows[r] ^= rows[rank]
        rank += 1; col -= 1
    return rank

def _geometry_penalty(rows, data_positions, parity_positions):
    # average Manhattan distance per edge; lower is better
    tot = 0; cnt = 0
    for r, cols in enumerate(rows):
        pr, pc = parity_positions[r]
        for i in cols:
            dr, dc = data_positions[i]
            tot += abs(pr - dr) + abs(pc - dc)
            cnt += 1
    return (tot / max(1, cnt))

def _score(Hcols, rows, target_dv, target_dc,
           allow_geometry_penalty, geo_weight,
           data_positions, parity_positions):
    # weights tuned for small grids; you can tweak
    w1, w2, w3, w4, w5, w6 = 3.0, 2.0, 0.5, 1.0, 0.75, geo_weight

    Nd = len(Hcols); M = len(rows)
    col_deg = [bin(h).count("1") for h in Hcols]
    row_deg = [len(r) for r in rows]
    deg_var = (sum((d - target_dv) ** 2 for d in col_deg) / max(1, Nd)) \
            + (sum((d - target_dc) ** 2 for d in row_deg) / max(1, M))

    f4 = _count_four_cycles(rows)

    # row rank bonus (proxy for good constraints)
    row_bits = [0] * M
    for c, h in enumerate(Hcols):
        bit = 1
        r = 0
        while bit <= h:
            if h & bit:
                row_bits[r] |= (1 << c)
            bit <<= 1
            r += 1
    rk = _rank_gf2(row_bits, Nd) / max(1, min(M, Nd))

    # base score
    s = (
        w1 * _uniq1_score(Hcols) +
        w2 * _uniq2_score(Hcols) -
        w3 * deg_var -
        w4 * f4 +
        w5 * rk
    )

    if allow_geometry_penalty and w6 > 0:
        gp = _geometry_penalty(rows, data_positions, parity_positions)  # lower is better
        # normalize roughly by grid size (assume <= 64)
        s -= w6 * (gp / 8.0)

    return s

# ========== CONSTRUCTION =================================

def _random_regular_edges(Nd, M, target_dv, rng):
    # Approximate column-regular bipartite design
    stubs = []
    for c in range(Nd):
        stubs += [c] * target_dv
    rng.shuffle(stubs)
    rows = [set() for _ in range(M)]
    r = 0
    for c in stubs:
        rows[r].add(c)
        r = (r + 1) % M
    return rows

def _rows_to_dict(rows, parity_positions, data_positions):
    rel = {}
    for r, ppos in enumerate(parity_positions):
        rel[ppos] = [data_positions[c] for c in sorted(rows[r])]
    return rel
