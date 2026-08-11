from parity_relations import generate_parity_relations
layout_roles = {
    # Row 0
    (0,0): "data", (0,1): "data", (0,2): "data", (0,3): "data", (0,4): "data",
    (0,5): "data", (0,6): "data", (0,7): "data", (0,8): "data", (0,9): "data",

    # Row 1
    (1,0): "orientation", (1,1): "parity", (1,2): "parity", (1,3): "parity", (1,4): "parity",
    (1,5): "parity", (1,6): "parity", (1,7): "parity", (1,8): "parity", (1,9): "orientation",

    # Row 2
    (2,0): "data", (2,1): "parity", (2,2): "data", (2,3): "data", (2,4): "data",
    (2,5): "data", (2,6): "data", (2,7): "data", (2,8): "parity", (2,9): "data",

    # Row 3
    (3,0): "data", (3,1): "parity", (3,2): "data", (3,3): "data", (3,4): "checksum",
    (3,5): "checksum", (3,6): "data", (3,7): "data", (3,8): "parity", (3,9): "data",

    # Row 4
    (4,0): "data", (4,1): "parity", (4,2): "data", (4,3): "data", (4,4): "checksum",
    (4,5): "checksum", (4,6): "data", (4,7): "data", (4,8): "parity", (4,9): "data",

    # Row 5
    (5,0): "data", (5,1): "parity", (5,2): "data", (5,3): "data", (5,4): "data",
    (5,5): "data", (5,6): "data", (5,7): "data", (5,8): "parity", (5,9): "data",

    # Row 6
    (6,0): "orientation", (6,1): "parity", (6,2): "parity", (6,3): "parity", (6,4): "parity",
    (6,5): "parity", (6,6): "parity", (6,7): "parity", (6,8): "parity", (6,9): "orientation",

    # Row 7
    (7,0): "data", (7,1): "data", (7,2): "data", (7,3): "data", (7,4): "data",
    (7,5): "data", (7,6): "data", (7,7): "data", (7,8): "index", (7,9): "index",
}

# layout_roles: {(r,c): "data"/"parity"/"checksum"/"index"/"orientation"/...}
parity_dict, meta = generate_parity_relations(
    layout_roles,
    target_dv=3,          # each protected bit participates in ~3 checks
    iters=80000,          # raise for better results
    seed=123,
    use_optimizer=True,   # set False to get degree-balanced random
    protect_roles=("data","checksum","index","orientation"),           # add "parity" here if you want parity-on-parity protection
    allow_geometry_penalty=True,       # prefer local edges (optional)
    geo_weight=0.3                     # tune 0.0–1.0
)

print("Optimized score:", meta["score"])
print("parity_dict:", parity_dict)
