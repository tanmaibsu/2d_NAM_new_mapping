"""Generate a block diagram of the _iterative_decode method."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# --- Style helpers ---
box_kw = dict(boxstyle="round,pad=0.4", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=1.5)
decision_kw = dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=1.5)
terminal_kw = dict(boxstyle="round,pad=0.4", facecolor="#C8E6C9", edgecolor="#2E7D32", linewidth=1.5)
fail_kw = dict(boxstyle="round,pad=0.4", facecolor="#FFCDD2", edgecolor="#C62828", linewidth=1.5)
arrow_kw = dict(arrowstyle="-|>", color="#333333", linewidth=1.5)
label_kw = dict(fontsize=9, color="#555555", fontstyle="italic")

def arrow(ax, xy1, xy2, **kw):
    merged = {**arrow_kw, **kw}
    ax.annotate("", xy=xy2, xytext=xy1, arrowprops=merged)

# ===================== NODES =====================

# 1. Start
ax.text(5, 19.3, "START", ha="center", va="center", fontsize=11, fontweight="bold",
        bbox=terminal_kw)

# 2. Build check graph
ax.text(5, 18.2, "Build Check Graph\n_build_check_graph()\nchecks, check_to_vars, _",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 3. Init beam
ax.text(5, 16.8, "Initialize Beam\nbeam = [(matrix_copy, [])]",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 4. Iteration check
ax.text(5, 15.5, "iter < max_iters\n(40)?",
        ha="center", va="center", fontsize=10, fontweight="bold", bbox=decision_kw)

# 5. For each candidate
ax.text(5, 14.2, "For each (mat, flips)\nin beam",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 6. Strict check
ax.text(5, 13.0, "All checks pass?\n_strict_matrix_ok(mat)",
        ha="center", va="center", fontsize=9, fontweight="bold", bbox=decision_kw)

# 6b. Return success
ax.text(8.5, 13.0, "Return\n(mat, flips)",
        ha="center", va="center", fontsize=9, fontweight="bold", bbox=terminal_kw)

# 7. Compute syndrome
ax.text(5, 11.6, "Compute Syndrome\nfailed = _failed_checks(mat)",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 8. Score variables
ax.text(5, 10.2, "Score Variables\nFor each failed check,\n+1 to every variable in it\nscores = Counter()",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 9. Generate children
ax.text(5, 8.6, "Generate Children\nFor top beam_width×2 scored vars:\n  flip bit → new candidate\n  (if len(flips) < max_flips)",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 10. Any candidates?
ax.text(5, 7.0, "next_candidates\nnon-empty?",
        ha="center", va="center", fontsize=10, fontweight="bold", bbox=decision_kw)

# 11. Evaluate & prune
ax.text(5, 5.5, "Evaluate & Prune\nRe-score each candidate:\n  (# failed checks, # flips)\nSort → keep best beam_width (12)",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 12. Update beam
ax.text(5, 4.0, "Update Beam\nbeam = top 12 candidates",
        ha="center", va="center", fontsize=9, bbox=box_kw)

# 13. Fail
ax.text(1.5, 5.5, "Return\n(-1, [])\nFAILURE",
        ha="center", va="center", fontsize=9, fontweight="bold", bbox=fail_kw)

# Fail from iteration exhaustion
ax.text(8.5, 15.5, "Return\n(-1, [])\nFAILURE",
        ha="center", va="center", fontsize=9, fontweight="bold", bbox=fail_kw)

# ===================== ARROWS =====================

# Start → Build
arrow(ax, (5, 19.0), (5, 18.6))
# Build → Init
arrow(ax, (5, 17.85), (5, 17.15))
# Init → Iter check
arrow(ax, (5, 16.45), (5, 15.85))
# Iter check → For each (Yes)
arrow(ax, (5, 15.15), (5, 14.55))
ax.text(4.5, 15.35, "Yes", **label_kw)
# Iter check → Fail (No)
arrow(ax, (6.3, 15.5), (7.5, 15.5))
ax.text(6.6, 15.65, "No", **label_kw)
# For each → Strict check
arrow(ax, (5, 13.85), (5, 13.35))
# Strict check → Return success (Yes)
arrow(ax, (6.5, 13.0), (7.5, 13.0))
ax.text(6.7, 13.15, "Yes", **label_kw)
# Strict check → Syndrome (No)
arrow(ax, (5, 12.65), (5, 12.0))
ax.text(4.5, 12.35, "No", **label_kw)
# Syndrome → Score
arrow(ax, (5, 11.2), (5, 10.7))
# Score → Generate
arrow(ax, (5, 9.7), (5, 9.15))
# Generate → Any candidates?
arrow(ax, (5, 8.05), (5, 7.4))
# Any candidates? → Evaluate (Yes)
arrow(ax, (5, 6.6), (5, 6.05))
ax.text(5.2, 6.35, "Yes", **label_kw)
# Any candidates? → Fail (No)
arrow(ax, (3.7, 7.0), (2.5, 6.1))
ax.text(2.7, 6.8, "No", **label_kw)
# Evaluate → Update beam
arrow(ax, (5, 5.0), (5, 4.4))
# Update beam → loop back to iteration check
arrow(ax, (5, 3.6), (1.0, 3.6), color="#1565C0")
ax.annotate("", xy=(1.0, 15.5), xytext=(1.0, 3.6),
            arrowprops=dict(arrowstyle="-|>", color="#1565C0", linewidth=1.5))
ax.annotate("", xy=(3.7, 15.5), xytext=(1.0, 15.5),
            arrowprops=dict(arrowstyle="-|>", color="#1565C0", linewidth=1.5))
ax.text(0.5, 9.5, "Next\nIteration", fontsize=8, color="#1565C0", fontweight="bold",
        ha="center", rotation=90)

# ===================== LEGEND / TITLE =====================
ax.text(5, 20.0, "_iterative_decode() — Beam-Search Syndrome Decoder",
        ha="center", va="center", fontsize=13, fontweight="bold")

# Legend
legend_items = [
    mpatches.Patch(facecolor="#E3F2FD", edgecolor="#1565C0", label="Process"),
    mpatches.Patch(facecolor="#FFF9C4", edgecolor="#F9A825", label="Decision"),
    mpatches.Patch(facecolor="#C8E6C9", edgecolor="#2E7D32", label="Success"),
    mpatches.Patch(facecolor="#FFCDD2", edgecolor="#C62828", label="Failure"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8,
          framealpha=0.9, edgecolor="#AAAAAA")

plt.tight_layout()
plt.savefig("/Users/Tanmai/Academic/Research/Codes/3d_NAM_new_mapping/iterative_decode_diagram.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved: iterative_decode_diagram.png")
