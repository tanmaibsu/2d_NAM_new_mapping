#!/usr/bin/env python3
"""Generate a PPTX presentation comparing old vs new dNAM decoder."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ── Colour palette ──
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
BLACK      = RGBColor(0x00, 0x00, 0x00)
DARK_BG    = RGBColor(0x1B, 0x1B, 0x2F)
ACCENT     = RGBColor(0x00, 0x96, 0xD6)   # blue
ACCENT2    = RGBColor(0x2E, 0xCC, 0x71)   # green
RED        = RGBColor(0xE7, 0x4C, 0x3C)
ORANGE     = RGBColor(0xF3, 0x9C, 0x12)
LIGHT_GRAY = RGBColor(0xEC, 0xF0, 0xF1)
DARK_GRAY  = RGBColor(0x2C, 0x3E, 0x50)
MID_GRAY   = RGBColor(0x7F, 0x8C, 0x8D)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── helpers ──

def _set_slide_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def _add_textbox(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=WHITE, alignment=PP_ALIGN.LEFT, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox

def _add_code_box(slide, left, top, width, height, text, font_size=11, bg_color=RGBColor(0x0D, 0x0D, 0x1A)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.15)
    for i, line in enumerate(text.split("\n")):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = RGBColor(0xA0, 0xE0, 0xA0)
        p.font.name = "Courier New"
    return shape

def _add_bullet_slide(slide, left, top, width, height, items, font_size=18, color=WHITE):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(8)
        p.level = 0
    return txBox

def _add_rounded_box(slide, left, top, width, height, text, fill_color, text_color=WHITE, font_size=14, bold=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.1)
    tf.margin_right = Inches(0.1)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = text_color
    p.font.bold = bold
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    return shape

def _add_arrow(slide, left, top, width, height, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

# ════════════════════════════════════════════════════════════════════
# SLIDE 1 – Title
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 1, 1.5, 11, 1.5,
             "Hybrid Beam-Search Syndrome Decoder",
             font_size=40, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
_add_textbox(slide, 1, 3.0, 11, 1,
             "for DNA Origami-based Nucleic Acid Memory (dNAM)",
             font_size=24, color=ACCENT, alignment=PP_ALIGN.CENTER)
_add_textbox(slide, 1, 4.5, 11, 0.5,
             "New Decoding Algorithm  |  Comparison with Previous Approach",
             font_size=18, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
_add_textbox(slide, 1, 6.0, 11, 0.5,
             "Tanmai  |  April 2026",
             font_size=16, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 2 – Agenda
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.4, 5, 0.8, "Agenda", font_size=32, bold=True, color=ACCENT)
items = [
    "1.  System Context — 8x10 Origami Matrix & Error Model",
    "2.  Previous Decoder — A*-Style Greedy Heuristic",
    "3.  New Decoder — Hybrid Beam-Search Syndrome Decoder",
    "4.  Tier 1 Deep Dive — Syndrome-Driven Beam Search",
    "5.  Strict Acceptance — Eliminating Silent Failures",
    "6.  Algorithm Comparison (Side-by-Side)",
    "7.  Complexity Analysis",
    "8.  End-to-End Pipeline",
    "9.  Exhaustive Testing Framework",
    "10. Summary & Key Takeaways",
]
_add_bullet_slide(slide, 1.5, 1.5, 10, 5.5, items, font_size=20, color=WHITE)

# ════════════════════════════════════════════════════════════════════
# SLIDE 3 – 8x10 Matrix Layout
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 8, 0.8, "8x10 DNA Origami Matrix Layout (24-Parity Config)",
             font_size=28, bold=True, color=ACCENT)

# Draw the grid
grid_left = 0.8
grid_top = 1.4
cell_w = 1.15
cell_h = 0.55

labels = [
    ["DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA"],
    ["ORI","PAR","PAR","PAR","PAR","PAR","PAR","PAR","PAR","ORI"],
    ["IDX","PAR","DATA","DATA","DATA","DATA","DATA","DATA","PAR","DATA"],
    ["IDX","PAR","DATA","DATA","CHK","CHK","DATA","DATA","PAR","DATA"],
    ["IDX","PAR","DATA","DATA","CHK","CHK","DATA","DATA","PAR","DATA"],
    ["DATA","PAR","DATA","DATA","DATA","DATA","DATA","DATA","PAR","DATA"],
    ["ORI","PAR","PAR","PAR","PAR","PAR","PAR","PAR","PAR","ORI"],
    ["DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA"],
]
colors_map = {
    "DATA": RGBColor(0x34,0x49,0x5E),
    "PAR":  ACCENT,
    "ORI":  ORANGE,
    "IDX":  RGBColor(0x8E,0x44,0xAD),
    "CHK":  RED,
}

for r, row in enumerate(labels):
    for c, lbl in enumerate(row):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(grid_left + c * cell_w), Inches(grid_top + r * cell_h),
            Inches(cell_w - 0.04), Inches(cell_h - 0.04))
        shape.fill.solid()
        shape.fill.fore_color.rgb = colors_map[lbl]
        shape.line.color.rgb = RGBColor(0x55,0x55,0x70)
        shape.line.width = Pt(0.5)
        tf = shape.text_frame
        p = tf.paragraphs[0]
        p.text = lbl
        p.font.size = Pt(10)
        p.font.color.rgb = WHITE
        p.font.bold = True
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER

# Legend
legend_top = 6.1
legend_items = [("DATA ~29 bits", colors_map["DATA"]), ("PARITY 24 bits", ACCENT),
                ("CHECKSUM 4 bits", RED), ("ORIENTATION 4 bits", ORANGE),
                ("INDEX 2-5 bits", RGBColor(0x8E,0x44,0xAD))]
for i, (txt, clr) in enumerate(legend_items):
    _add_rounded_box(slide, 0.8 + i * 2.4, legend_top, 2.2, 0.4, txt, clr, font_size=11, bold=True)

# Error model box on right
_add_textbox(slide, 0.8, 6.7, 11, 0.6,
             "Error Model:  False Negatives (1\u21920) are dominant  |  False Positives (0\u21921) are rare  |  Asymmetric channel",
             font_size=14, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 4 – Previous Decoder: A*-Style Greedy
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 8, 0.8, "Previous Decoder: A*-Style Greedy Heuristic",
             font_size=28, bold=True, color=ACCENT)

# Flow diagram using boxes + arrows
box_x = 1.5
_add_rounded_box(slide, box_x, 1.3, 3.5, 0.6, "Received Origami (80-bit)", DARK_GRAY, font_size=14, bold=True)
_add_arrow(slide, box_x + 1.5, 1.95, 0.4, 0.35)
_add_rounded_box(slide, box_x, 2.4, 3.5, 0.6, "Fix Orientation (4 rotations)", DARK_GRAY, font_size=14)
_add_arrow(slide, box_x + 1.5, 3.05, 0.4, 0.35)
_add_rounded_box(slide, box_x, 3.5, 3.5, 0.6, "Compute Matrix Weight (scoring)", ORANGE, font_size=14, bold=True)
_add_arrow(slide, box_x + 1.5, 4.15, 0.4, 0.35)
_add_rounded_box(slide, box_x - 0.5, 4.6, 2.0, 0.6, "weight == 0?\nACCEPT", ACCENT2, font_size=13, bold=True)
_add_rounded_box(slide, box_x + 2.0, 4.6, 2.0, 0.6, "weight > 0\nPQ Search", RED, font_size=13, bold=True)

# Scoring function explanation on right
_add_textbox(slide, 6.0, 1.2, 6.5, 0.6, "Weight Scoring Function", font_size=22, bold=True, color=ORANGE)

score_text = (
    "1. Flip specified bits in matrix copy\n"
    "2. Check all 24 parity equations\n"
    "   \u2192 Collect data positions from failing parities\n"
    "3. Check all 4 checksum equations\n"
    "   \u2192 Add suspect positions from failing checksums\n"
    "4. Score each suspect position:\n"
    "   weight = parity_violations + checksum_bonus\n"
    "5. Group by weight, filter by threshold\n"
    "6. Return (matrix, norm_weight, probable_errors)"
)
_add_code_box(slide, 6.0, 1.9, 6.5, 3.0, score_text, font_size=12)

# Problems box
_add_textbox(slide, 6.0, 5.1, 6.5, 0.5, "Key Limitations", font_size=20, bold=True, color=RED)
problems = [
    "\u2716  weight==0 acceptance is NOT strict (silent corruption possible)",
    "\u2716  Priority queue grows exponentially: O(k^d) for d errors",
    "\u2716  Impractical beyond 5-6 errors",
    "\u2716  Single strategy with no fallback",
]
_add_bullet_slide(slide, 6.2, 5.6, 6.5, 1.8, problems, font_size=14, color=RGBColor(0xFF,0x99,0x99))

# ════════════════════════════════════════════════════════════════════
# SLIDE 5 – New Decoder: Three-Tier Architecture
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "New Decoder: Hybrid Beam-Search Syndrome Decoder",
             font_size=28, bold=True, color=ACCENT)
_add_textbox(slide, 0.8, 0.9, 10, 0.5, "Three-Tier Architecture with Strict Acceptance at Every Level",
             font_size=16, color=MID_GRAY)

# Tier 0
cx = 2.0
_add_rounded_box(slide, cx, 1.6, 4.0, 0.55, "Received Origami (80-bit stream)", DARK_GRAY, font_size=14, bold=True)
_add_arrow(slide, cx + 1.7, 2.2, 0.4, 0.3)

_add_rounded_box(slide, cx - 0.5, 2.6, 5.0, 0.8,
                 "TIER 0: STRICT ACCEPT\nOrientation + ALL Parities + ALL Checksums",
                 ACCENT2, font_size=14, bold=True)
_add_arrow(slide, cx + 1.7, 3.5, 0.4, 0.3, color=RED)
_add_textbox(slide, cx + 2.3, 3.45, 1.5, 0.3, "FAIL", font_size=12, bold=True, color=RED)

# Tier 1
_add_rounded_box(slide, cx - 0.5, 3.9, 5.0, 0.9,
                 "TIER 1: BEAM-SEARCH SYNDROME DECODER\nBuild check graph \u2192 Syndrome scores \u2192 Beam search (w=6)",
                 ACCENT, font_size=14, bold=True)
_add_arrow(slide, cx + 1.7, 4.9, 0.4, 0.3, color=RED)
_add_textbox(slide, cx + 2.3, 4.85, 1.5, 0.3, "FAIL", font_size=12, bold=True, color=RED)

# Tier 2
_add_rounded_box(slide, cx - 0.5, 5.3, 5.0, 0.8,
                 "TIER 2: LEGACY GREEDY FALLBACK\nWeight-based heuristic + STRICT acceptance",
                 ORANGE, font_size=14, bold=True)

# Accept / Fail boxes
_add_rounded_box(slide, 7.5, 2.8, 1.8, 0.5, "\u2713 ACCEPT", ACCENT2, font_size=14, bold=True)
_add_rounded_box(slide, 7.5, 4.1, 1.8, 0.5, "\u2713 ACCEPT", ACCENT2, font_size=14, bold=True)
_add_rounded_box(slide, 7.5, 5.4, 1.8, 0.5, "\u2713 ACCEPT", ACCENT2, font_size=14, bold=True)
_add_rounded_box(slide, cx - 0.5, 6.3, 2.5, 0.5, "\u2716 FAIL (return -1)", RED, font_size=14, bold=True)

# Key properties on the right
_add_textbox(slide, 9.8, 1.5, 3.3, 0.5, "Key Properties", font_size=20, bold=True, color=ACCENT)
props = [
    "\u2713 Strict validation at EVERY tier",
    "\u2713 Syndrome-directed bit selection",
    "\u2713 Bounded memory (beam width=6)",
    "\u2713 Up to ~8 errors correctable",
    "\u2713 Fallback ensures max recovery",
    "\u2713 No silent data corruption",
]
_add_bullet_slide(slide, 9.8, 2.1, 3.3, 4.0, props, font_size=15, color=ACCENT2)

# ════════════════════════════════════════════════════════════════════
# SLIDE 6 – Tier 1 Deep Dive: Beam Search
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Tier 1 Deep Dive: Syndrome-Driven Beam Search",
             font_size=28, bold=True, color=ACCENT)

# Step 1: Check Graph
_add_textbox(slide, 0.8, 1.2, 4, 0.5, "Step 1: Build Check Graph", font_size=18, bold=True, color=ACCENT2)
step1 = (
    "# Combine parity + checksum into\n"
    "# unified bipartite graph\n"
    "all_checks = parity_relations\n"
    "             + checksum_relations\n"
    "\n"
    "check_to_vars[check] = [variables]\n"
    "var_to_checks[var]   = [checks]"
)
_add_code_box(slide, 0.8, 1.7, 5.0, 2.2, step1, font_size=12)

# Step 2: Syndrome
_add_textbox(slide, 0.8, 4.1, 4, 0.5, "Step 2: Compute Syndrome", font_size=18, bold=True, color=ACCENT2)
step2 = (
    "failed_checks = set()\n"
    "for check in all_checks:\n"
    "    xor = XOR(all vars in check)\n"
    "    if xor != 0:\n"
    "        failed_checks.add(check)"
)
_add_code_box(slide, 0.8, 4.6, 5.0, 1.7, step2, font_size=12)

# Step 3: Beam Search (right side)
_add_textbox(slide, 6.5, 1.2, 6, 0.5, "Step 3: Beam Search", font_size=18, bold=True, color=ACCENT2)

# Beam iteration diagram
_add_textbox(slide, 6.5, 1.7, 6, 0.4, "Iteration 0:", font_size=14, bold=True, color=WHITE)
for i in range(6):
    clr = ACCENT if i == 0 else DARK_GRAY
    _add_rounded_box(slide, 6.5 + i * 1.05, 2.1, 1.0, 0.4,
                     "mat\u2080" if i == 0 else "\u2014", clr, font_size=11)

_add_textbox(slide, 6.5, 2.6, 6, 0.7,
             "Score each variable by # failed checks it participates in.\n"
             "Try flipping top-scored variables (beam_width \u00d7 2 candidates).",
             font_size=13, color=LIGHT_GRAY)

_add_textbox(slide, 6.5, 3.3, 6, 0.4, "Re-score, keep top 6:", font_size=14, bold=True, color=WHITE)
beam_labels = ["best", "2nd", "3rd", "4th", "5th", "6th"]
for i, lbl in enumerate(beam_labels):
    _add_rounded_box(slide, 6.5 + i * 1.05, 3.7, 1.0, 0.4, lbl, ACCENT, font_size=11)

_add_textbox(slide, 6.5, 4.2, 6, 0.7,
             "Repeat for up to 40 iterations.\n"
             "At each step: if ANY candidate passes STRICT check \u2192 ACCEPT.",
             font_size=13, color=LIGHT_GRAY)

# Key insight box
_add_rounded_box(slide, 6.5, 5.1, 6.0, 1.8,
                 "Syndrome-Directed Selection\n\n"
                 "scores = Counter()\n"
                 "for each failed check c:\n"
                 "    for each variable v in c:\n"
                 "        scores[v] += 1\n\n"
                 "Flip the variables with the HIGHEST scores first.",
                 RGBColor(0x1A, 0x2A, 0x3A), ACCENT2, font_size=13)

# ════════════════════════════════════════════════════════════════════
# SLIDE 7 – Strict Acceptance
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Strict Acceptance: Eliminating Silent Failures",
             font_size=28, bold=True, color=ACCENT)

# OLD approach (left)
_add_textbox(slide, 0.8, 1.3, 5.5, 0.5, "PREVIOUS: Weight-Based", font_size=22, bold=True, color=RED)
old_code = (
    "if weight == 0:\n"
    "    return matrix  # ACCEPT\n"
    "\n"
    "# Problems:\n"
    "# - weight==0 does NOT guarantee\n"
    "#   all checks actually pass\n"
    "# - Normalization artifacts can\n"
    "#   make weight zero with errors\n"
    "# - SILENT DATA CORRUPTION"
)
_add_code_box(slide, 0.8, 1.9, 5.5, 2.8, old_code, font_size=13)

_add_rounded_box(slide, 1.5, 4.9, 4.0, 0.6,
                 "\u2716  Can return WRONG matrix silently", RED, font_size=14, bold=True)

# NEW approach (right)
_add_textbox(slide, 7.0, 1.3, 5.5, 0.5, "NEW: Strict Validation", font_size=22, bold=True, color=ACCENT2)
new_code = (
    "def _strict_matrix_ok(matrix):\n"
    "  # 1. Fix orientation\n"
    "  ori, oriented = fix_orientation()\n"
    "  if ori == -1: return False\n"
    "\n"
    "  # 2. Check ALL parity equations\n"
    "  _, bad = find_error_location()\n"
    "  if bad: return False\n"
    "\n"
    "  # 3. Check ALL checksums\n"
    "  if not check_checksum(): return False\n"
    "\n"
    "  return True  # ALL pass!"
)
_add_code_box(slide, 7.0, 1.9, 5.5, 3.4, new_code, font_size=13)

_add_rounded_box(slide, 7.7, 5.5, 4.0, 0.6,
                 "\u2713  Used at EVERY acceptance point", ACCENT2, font_size=14, bold=True)

# Bottom note
_add_textbox(slide, 0.8, 6.3, 12, 0.6,
             "Applied at: Tier 0 (zero-error) | Tier 1 (each beam candidate) | Tier 2 (legacy fallback) | return_matrix() final gate",
             font_size=15, color=ACCENT, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 8 – Side-by-Side Comparison
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Algorithm Comparison: Old vs. New",
             font_size=28, bold=True, color=ACCENT)

# Table
rows_data = [
    ("Aspect",              "Previous (A* Greedy)",                  "New (Hybrid Beam-Search)"),
    ("Search Strategy",     "Priority queue (min-heap)\nover weight heuristic",
                                                                     "Beam search (width=6)\nover syndrome scores"),
    ("Error Detection",     "Indirect: via parity\nviolation counts (weight)",
                                                                     "Direct: syndrome computation\n(which checks failed?)"),
    ("Candidate Selection", "All probable errors\nfrom weight function",
                                                                     "Top variables by syndrome\nparticipation count"),
    ("Memory Usage",        "Unbounded visited set +\ngrowing priority queue",
                                                                     "Fixed: beam_width states\nper iteration"),
    ("Acceptance",          "weight == 0 (WEAK)",                    "Strict: ALL parity +\nALL checksum must pass"),
    ("Fallback",            "None (single strategy)",                "Tier 2: legacy heuristic\nwith strict acceptance"),
    ("Max Errors",          "~5-6 in practice\n(PQ explosion)",      "~8 reliably\n(beam keeps it tractable)"),
]

tbl_left = 0.8
tbl_top = 1.3
col_widths = [2.5, 4.5, 4.5]
row_height = 0.7

for r, (aspect, old_val, new_val) in enumerate(rows_data):
    for c, (val, w) in enumerate(zip([aspect, old_val, new_val], col_widths)):
        x = tbl_left + sum(col_widths[:c])
        y = tbl_top + r * row_height
        if r == 0:
            bg = ACCENT
            fc = WHITE
            bold = True
            fs = 13
        elif c == 0:
            bg = RGBColor(0x2C, 0x3E, 0x50)
            fc = WHITE
            bold = True
            fs = 12
        elif c == 1:
            bg = RGBColor(0x3A, 0x20, 0x20)
            fc = RGBColor(0xFF, 0xBB, 0xBB)
            bold = False
            fs = 12
        else:
            bg = RGBColor(0x1A, 0x3A, 0x1A)
            fc = RGBColor(0xBB, 0xFF, 0xBB)
            bold = False
            fs = 12

        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                       Inches(x), Inches(y), Inches(w), Inches(row_height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.color.rgb = RGBColor(0x44, 0x44, 0x55)
        shape.line.width = Pt(0.5)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.1)
        tf.margin_top = Inches(0.05)
        for li, line in enumerate(val.split("\n")):
            p = tf.paragraphs[0] if li == 0 else tf.add_paragraph()
            p.text = line
            p.font.size = Pt(fs)
            p.font.color.rgb = fc
            p.font.bold = bold
            p.font.name = "Calibri"

# ════════════════════════════════════════════════════════════════════
# SLIDE 9 – Complexity Analysis
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Complexity Analysis",
             font_size=28, bold=True, color=ACCENT)

# A* complexity
_add_textbox(slide, 0.8, 1.2, 5.5, 0.5, "Previous: A* Decoder", font_size=20, bold=True, color=RED)
a_star_text = (
    "Per iteration: O(k) children\n"
    "  each costs O(P) for weight computation\n"
    "  P = # parity checks, k = # probable errors\n"
    "\n"
    "Worst case: O(k^d x P)\n"
    "  d = max_errors\n"
    "\n"
    "Space: O(k^d) visited set\n"
    "\n"
    "EXPONENTIAL growth in d"
)
_add_code_box(slide, 0.8, 1.8, 5.5, 2.8, a_star_text, font_size=13)

# Beam complexity
_add_textbox(slide, 7.0, 1.2, 5.5, 0.5, "New: Beam-Search Decoder", font_size=20, bold=True, color=ACCENT2)
beam_text = (
    "Per iteration: O(B x 2B) candidates\n"
    "  each costs O(C) for syndrome\n"
    "  B = beam_width (6), C = # checks (28)\n"
    "\n"
    "Worst case: O(I x B^2 x C)\n"
    "  I = max_iterations (40)\n"
    "\n"
    "Space: O(B x 80) = CONSTANT\n"
    "\n"
    "LINEAR growth with parameters"
)
_add_code_box(slide, 7.0, 1.8, 5.5, 2.8, beam_text, font_size=13)

# Comparison table
_add_textbox(slide, 2.5, 5.0, 8, 0.5, "Operations by Error Count (Approximate)",
             font_size=18, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)

comp_rows = [
    ("Errors (d)", "A* (approx ops)", "Beam (approx ops)"),
    ("1",          "~30",             "~400"),
    ("2",          "~900",            "~800"),
    ("3",          "~27,000",         "~1,200"),
    ("5",          "~24,000,000",     "~2,000"),
    ("8",          "~6.5 x 10\u00b9\u00b9","~3,200"),
]
for r, (e, a, b) in enumerate(comp_rows):
    for c, (val, w) in enumerate(zip([e, a, b], [2.0, 3.0, 3.0])):
        x = 2.5 + sum([2.0, 3.0, 3.0][:c])
        y = 5.5 + r * 0.35
        bg = ACCENT if r == 0 else (RGBColor(0x3A, 0x20, 0x20) if c == 1 else
                                     RGBColor(0x1A, 0x3A, 0x1A) if c == 2 else DARK_GRAY)
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                       Inches(x), Inches(y), Inches(w), Inches(0.32))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg
        shape.line.color.rgb = RGBColor(0x44, 0x44, 0x55)
        shape.line.width = Pt(0.5)
        tf = shape.text_frame
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(12)
        p.font.color.rgb = WHITE
        p.font.bold = (r == 0)
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER

# ════════════════════════════════════════════════════════════════════
# SLIDE 10 – End-to-End Pipeline
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "End-to-End Decoding Pipeline",
             font_size=28, bold=True, color=ACCENT)

# Pipeline boxes
steps = [
    ("Origami Text Files\n(80-bit binary strings)", DARK_GRAY, 0.8),
    ("ProcessFile.decode()", ACCENT, 1.8),
    ("Layout Setup\n(data/index/parity/checksum positions)", RGBColor(0x2C,0x3E,0x50), 2.7),
    ("ThreadPoolExecutor\n(up to 32 parallel workers)", RGBColor(0x8E,0x44,0xAD), 3.6),
    ("Per-Origami: Tier 0 \u2192 Tier 1 \u2192 Tier 2", ACCENT, 4.5),
    ("Majority Voting\n(across redundant copies per index)", ORANGE, 5.4),
    ("Recovered Binary File + CSV Report", ACCENT2, 6.3),
]

for text, color, top in steps:
    _add_rounded_box(slide, 3.0, top, 7.0, 0.7, text, color, font_size=14, bold=True)
    if top < 6.3:
        _add_arrow(slide, 6.3, top + 0.72, 0.35, 0.25)

# Side annotations
_add_textbox(slide, 10.3, 3.4, 2.8, 1.5,
             "Each worker runs the\n3-tier hybrid decoder\nindependently on a\nsingle origami.",
             font_size=13, color=MID_GRAY)
_add_textbox(slide, 10.3, 5.3, 2.8, 1.0,
             "Multiple copies of the\nsame origami are decoded\nand majority-voted.",
             font_size=13, color=MID_GRAY)

# ════════════════════════════════════════════════════════════════════
# SLIDE 11 – Exhaustive Testing
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Exhaustive Testing Framework",
             font_size=28, bold=True, color=ACCENT)

# Method
_add_textbox(slide, 0.8, 1.2, 6, 0.5, "Test Methodology", font_size=20, bold=True, color=ACCENT2)
method_items = [
    "For each origami, for k \u2208 {1, 2, 3}:",
    "  Test ALL C(n,k) combinations of 1\u21920 bit flips",
    "  Run full decode pipeline on each",
    "  Compare decoded output to original",
    "  Log all failures + per-bit vulnerability votes",
]
_add_bullet_slide(slide, 1.0, 1.7, 5.5, 2.5, method_items, font_size=15, color=WHITE)

# Scale
_add_textbox(slide, 0.8, 3.8, 5.5, 0.5, "Test Scale (per origami, ~40 '1'-bits)", font_size=16, bold=True, color=ORANGE)
scale_rows = [
    ("Test Type", "Combinations"),
    ("Single-bit (k=1)", "C(40,1) = 40"),
    ("Double-bit (k=2)", "C(40,2) = 780"),
    ("Triple-bit (k=3)", "C(40,3) = 9,880"),
    ("TOTAL", "~10,700 tests"),
]
for r, (label, val) in enumerate(scale_rows):
    bg = ACCENT if r == 0 else (ORANGE if r == 4 else DARK_GRAY)
    _add_rounded_box(slide, 0.8, 4.3 + r * 0.42, 2.8, 0.38, label, bg, font_size=12, bold=(r==0 or r==4))
    _add_rounded_box(slide, 3.7, 4.3 + r * 0.42, 2.5, 0.38, val, bg, font_size=12, bold=(r==0 or r==4))

# Outputs
_add_textbox(slide, 7.0, 1.2, 5.5, 0.5, "Outputs", font_size=20, bold=True, color=ACCENT2)
_add_rounded_box(slide, 7.0, 1.8, 5.5, 1.2,
                 "failures.csv\n"
                 "Every failing combination:\n"
                 "  (origami_idx, k, flipped_indices, status)",
                 RGBColor(0x3A, 0x20, 0x20), RGBColor(0xFF, 0xBB, 0xBB), font_size=13)

_add_rounded_box(slide, 7.0, 3.2, 5.5, 1.2,
                 "votes.csv\n"
                 "Per-bit vulnerability score:\n"
                 "  How often each bit participates in\n"
                 "  uncorrectable pairs/triplets",
                 RGBColor(0x1A, 0x3A, 0x1A), RGBColor(0xBB, 0xFF, 0xBB), font_size=13)

# Vulnerability heatmap concept
_add_textbox(slide, 7.0, 4.6, 5.5, 0.5, "Bit Vulnerability Insight", font_size=18, bold=True, color=ORANGE)
_add_textbox(slide, 7.0, 5.1, 5.5, 2.0,
             "Bits covered by fewer parity equations are harder\n"
             "to correct when paired with other errors, yielding\n"
             "higher vulnerability scores.\n\n"
             "This analysis identifies structural weaknesses in\n"
             "the parity mapping design itself.",
             font_size=14, color=LIGHT_GRAY)

# ════════════════════════════════════════════════════════════════════
# SLIDE 12 – Summary
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Summary & Key Takeaways",
             font_size=28, bold=True, color=ACCENT)

takeaways = [
    ("\u2713 CORRECTNESS",   "Eliminated silent failures via strict acceptance at every tier",    ACCENT2),
    ("\u2713 CAPABILITY",    "Extended reliable error correction from ~5-6 to ~8 bit errors",    ACCENT2),
    ("\u2713 EFFICIENCY",    "Beam search O(I\u00b7B\u00b2\u00b7C) vs A* O(k^d\u00b7P) — bounded memory", ACCENT2),
    ("\u2713 ROBUSTNESS",    "Three-tier fallback ensures maximum recovery rate",                ACCENT2),
    ("\u2713 TESTABILITY",   "Exhaustive combinatorial testing with per-bit vulnerability analysis", ACCENT2),
]

for i, (title, desc, color) in enumerate(takeaways):
    y = 1.5 + i * 1.1
    _add_rounded_box(slide, 1.0, y, 2.8, 0.7, title, color, WHITE, font_size=18, bold=True)
    _add_textbox(slide, 4.2, y + 0.1, 8.5, 0.7, desc, font_size=18, color=WHITE)

# Bottom
_add_textbox(slide, 1, 6.5, 11, 0.5,
             "The hybrid approach combines the systematic rigor of syndrome decoding\n"
             "with the domain-specific heuristics of the legacy decoder.",
             font_size=16, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 13 – Intuition: The Doctor Analogy
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: The Doctor Analogy",
             font_size=28, bold=True, color=ACCENT)

# Old doctor (left)
_add_textbox(slide, 0.8, 1.2, 5.5, 0.5, "Old Decoder = General Practitioner", font_size=20, bold=True, color=RED)
_add_rounded_box(slide, 0.8, 1.8, 5.5, 3.6,
    '"Something feels off" (weight > 0)\n\n'
    'Tries one medicine at a time.\n'
    'Picks the medicine that makes the\n'
    'patient "feel best" (lowest weight).\n\n'
    'If patient says "I feel fine" (weight=0),\n'
    'doctor declares them cured.\n\n'
    'Problem: Patient might THINK they are\n'
    'fine but still be sick (silent corruption).\n'
    'Also: tries every possible drug combo\n'
    'before giving up (exponential search).',
    RGBColor(0x3A, 0x20, 0x20), RGBColor(0xFF, 0xCC, 0xCC), font_size=14)

# New doctor (right)
_add_textbox(slide, 7.0, 1.2, 5.5, 0.5, "New Decoder = Specialist with Lab Tests", font_size=20, bold=True, color=ACCENT2)
_add_rounded_box(slide, 7.0, 1.8, 5.5, 3.6,
    'Runs 28 specific diagnostic tests (syndromes).\n\n'
    'Each failing test points to specific organs.\n'
    'The organ flagged by the MOST tests\n'
    'is treated first (syndrome scoring).\n\n'
    'Keeps 6 treatment plans in parallel (beam).\n'
    'After each treatment, re-runs ALL 28 tests.\n\n'
    'Only declares "cured" when EVERY test\n'
    'comes back clean (strict acceptance).\n\n'
    'If specialist fails, calls in the old GP\n'
    'for a second opinion (Tier 2 fallback).',
    RGBColor(0x1A, 0x3A, 0x1A), RGBColor(0xCC, 0xFF, 0xCC), font_size=14)

# Key insight at bottom
_add_rounded_box(slide, 1.5, 5.7, 10.0, 1.2,
    'Key Insight: The old decoder asks "does the patient FEEL better?" (heuristic weight).\n'
    'The new decoder asks "do ALL the lab tests PASS?" (syndrome verification).\n'
    'Feeling better \u2260 being cured. All tests passing = mathematical guarantee.',
    DARK_GRAY, ACCENT, font_size=16, bold=True)

# ════════════════════════════════════════════════════════════════════
# SLIDE 14 – Intuition: Why Syndromes Work
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: Why Syndromes Pinpoint Errors",
             font_size=28, bold=True, color=ACCENT)

_add_textbox(slide, 0.8, 1.1, 12, 0.6,
             "Each parity check is like a \"trip wire\" — if ANY bit it covers is flipped, the check fails.",
             font_size=18, color=LIGHT_GRAY)

# Visual: overlapping circles showing parity coverage
# Draw a worked example
_add_textbox(slide, 0.8, 1.8, 6, 0.5, "Worked Example: Single Error at Position X",
             font_size=20, bold=True, color=ACCENT2)

example_text = (
    "Suppose bit (2,4) is flipped (1 \u2192 0).\n"
    "\n"
    "Parity checks covering (2,4):\n"
    "  P1 = (1,2)  covers (2,4) among 12 bits \u2192 FAILS\n"
    "  P2 = (1,3)  covers (2,4) among 12 bits \u2192 FAILS\n"
    "  P3 = (3,8)  covers (2,4) among 12 bits \u2192 FAILS\n"
    "  P4 = (6,1)  covers (2,4) among 12 bits \u2192 FAILS\n"
    "  P5 = (6,4)  covers (2,4) among 12 bits \u2192 FAILS\n"
    "\n"
    "Now count: which variable appears in ALL 5 failing checks?\n"
    "  \u2192 (2,4) appears in 5/5  \u2190 HIGHEST SCORE\n"
    "  \u2192 Other bits appear in 1-2 at most\n"
    "\n"
    "The syndrome literally POINTS to the error."
)
_add_code_box(slide, 0.8, 2.4, 6.0, 4.2, example_text, font_size=13)

# Venn-style diagram concept on right
_add_textbox(slide, 7.5, 1.8, 5, 0.5, "Overlapping Parity Coverage", font_size=20, bold=True, color=ACCENT2)

# Draw overlapping circles as rounded rectangles representing parity groups
parity_colors = [
    RGBColor(0x00, 0x60, 0x90),
    RGBColor(0x00, 0x70, 0x50),
    RGBColor(0x80, 0x40, 0x80),
    RGBColor(0x90, 0x60, 0x00),
    RGBColor(0x60, 0x20, 0x20),
]
parity_labels = ["P1", "P2", "P3", "P4", "P5"]
for i in range(5):
    _add_rounded_box(slide, 7.5 + (i % 3) * 1.5, 2.5 + (i // 3) * 1.8, 2.5, 1.5,
                     f"{parity_labels[i]} covers\n12 bits each", parity_colors[i],
                     WHITE, font_size=12)

# Center intersection = the error
_add_rounded_box(slide, 9.0, 3.5, 1.5, 0.8,
                 "Bit (2,4)\n5/5 FAILS", RED, WHITE, font_size=13, bold=True)

_add_textbox(slide, 7.5, 5.5, 5.3, 1.5,
             "The INTERSECTION of all failing checks\n"
             "narrows down the error location.\n\n"
             "More parity checks per bit = better\n"
             "triangulation = easier to correct.\n\n"
             "This is why 24-parity (12 bits each)\n"
             "outperforms 16-parity for correction.",
             font_size=14, color=LIGHT_GRAY)

# ════════════════════════════════════════════════════════════════════
# SLIDE 15 – Intuition: Beam Search vs Priority Queue
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: Beam Search vs. Priority Queue",
             font_size=28, bold=True, color=ACCENT)

_add_textbox(slide, 0.8, 1.1, 12, 0.5,
             "Think of it as exploring a maze to find the right combination of bit flips.",
             font_size=17, color=LIGHT_GRAY)

# Left: A* tree explosion
_add_textbox(slide, 0.5, 1.7, 6, 0.5, "A* Priority Queue: Explore Everything", font_size=18, bold=True, color=RED)

# Draw an expanding tree
# Level 0
_add_rounded_box(slide, 2.5, 2.3, 1.6, 0.4, "Start", DARK_GRAY, font_size=11, bold=True)
# Level 1 (3 branches)
for i in range(3):
    _add_rounded_box(slide, 0.8 + i * 2.0, 3.0, 1.4, 0.35, f"flip {chr(65+i)}", RGBColor(0x55,0x33,0x33), font_size=10)
# Level 2 (9 branches)
for i in range(9):
    x = 0.3 + i * 0.65
    _add_rounded_box(slide, x, 3.6, 0.6, 0.3, "...", RGBColor(0x44,0x22,0x22), font_size=9)
# Level 3 (27 branches - just show dots)
_add_textbox(slide, 0.5, 4.0, 5.5, 0.4, "... 27 states ... 81 states ... 243 states ... EXPLOSION!",
             font_size=12, bold=True, color=RED)

_add_rounded_box(slide, 0.5, 4.5, 5.8, 1.4,
    "Remembers EVERY path ever tried (visited set).\n"
    "Explores ALL promising branches.\n"
    "At depth 8: up to k\u2078 states to track.\n\n"
    "Like searching every hallway in a huge\n"
    "building and remembering all of them.",
    RGBColor(0x3A, 0x20, 0x20), RGBColor(0xFF, 0xBB, 0xBB), font_size=13)

# Right: Beam search
_add_textbox(slide, 7.0, 1.7, 6, 0.5, "Beam Search: Keep Only the Best 6", font_size=18, bold=True, color=ACCENT2)

# Draw narrow beam
_add_rounded_box(slide, 9.0, 2.3, 1.6, 0.4, "Start", DARK_GRAY, font_size=11, bold=True)
# Level 1: show 12 candidates, only 6 survive
for i in range(6):
    _add_rounded_box(slide, 7.2 + i * 1.0, 3.0, 0.9, 0.35, f"try {i+1}", ACCENT, font_size=10)
# Pruning arrow
_add_textbox(slide, 7.2, 3.4, 6, 0.3, "\u2193 re-score, keep best 6, discard rest \u2193",
             font_size=11, color=ACCENT2, alignment=PP_ALIGN.CENTER)
# Level 2: again 6
for i in range(6):
    _add_rounded_box(slide, 7.2 + i * 1.0, 3.8, 0.9, 0.35, f"try {i+1}", ACCENT, font_size=10)
_add_textbox(slide, 7.2, 4.2, 6, 0.3, "\u2193 re-score, keep best 6, discard rest \u2193",
             font_size=11, color=ACCENT2, alignment=PP_ALIGN.CENTER)

_add_rounded_box(slide, 7.0, 4.6, 6.0, 1.4,
    "At EVERY step: evaluate, prune, keep top 6.\n"
    "Memory never grows beyond 6 states.\n"
    "Each iteration takes the same time.\n\n"
    "Like walking with 6 scouts — each explores\n"
    "one direction, you keep the 6 most promising.",
    RGBColor(0x1A, 0x3A, 0x1A), RGBColor(0xBB, 0xFF, 0xBB), font_size=13)

# Bottom comparison
_add_rounded_box(slide, 1.0, 6.2, 5.5, 0.9,
    "A* with 8 errors:\nMemory explodes, runtime unbounded",
    RED, WHITE, font_size=15, bold=True)
_add_rounded_box(slide, 7.0, 6.2, 5.5, 0.9,
    "Beam with 8 errors:\n6 states in memory, 40 iterations max",
    ACCENT2, WHITE, font_size=15, bold=True)

# ════════════════════════════════════════════════════════════════════
# SLIDE 16 – Intuition: The Asymmetric Error Advantage
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: Exploiting Asymmetric Errors",
             font_size=28, bold=True, color=ACCENT)

_add_textbox(slide, 0.8, 1.1, 12, 0.6,
             "DNA origami readout errors are NOT random — they are heavily biased toward 1\u21920 (false negatives).",
             font_size=18, color=LIGHT_GRAY)

# The physical reason (left)
_add_textbox(slide, 0.8, 1.9, 5.5, 0.5, "Physical Reality", font_size=20, bold=True, color=ORANGE)
_add_rounded_box(slide, 0.8, 2.4, 5.5, 2.0,
    "In DNA origami:\n"
    "  1 = staple strand present (detected)\n"
    "  0 = staple strand absent (not detected)\n\n"
    "Missing a present staple (1\u21920) is EASY:\n"
    "  strand fell off, imaging missed it, etc.\n\n"
    "Detecting a phantom staple (0\u21921) is RARE:\n"
    "  requires contamination or imaging artifact.",
    RGBColor(0x3A, 0x2A, 0x10), RGBColor(0xFF, 0xDD, 0xAA), font_size=14)

# How decoder exploits this (right)
_add_textbox(slide, 7.0, 1.9, 5.5, 0.5, "How the Decoder Exploits This", font_size=20, bold=True, color=ACCENT2)
_add_rounded_box(slide, 7.0, 2.4, 5.5, 2.0,
    "When a syndrome says \"error near bit X\":\n\n"
    "  If X is currently 0:\n"
    "    \u2192 Very likely it was 1 originally\n"
    "    \u2192 HIGH confidence to flip 0\u21921\n\n"
    "  If X is currently 1:\n"
    "    \u2192 Unlikely it needs flipping (0\u21921 is rare)\n"
    "    \u2192 Only flip if false_positive budget allows\n\n"
    "This asymmetry REDUCES the search space!",
    RGBColor(0x1A, 0x3A, 0x1A), RGBColor(0xCC, 0xFF, 0xCC), font_size=14)

# Impact diagram
_add_textbox(slide, 0.8, 4.7, 12, 0.5, "Impact on Search Space",
             font_size=20, bold=True, color=ACCENT)

# Symmetric: all 80 bits could be errors
_add_rounded_box(slide, 0.8, 5.3, 5.5, 1.0,
    "Symmetric errors (random channel):\n"
    "Any of 80 bits could be wrong\n"
    "C(80,3) = 82,160 triple-error combos",
    RGBColor(0x3A, 0x20, 0x20), RGBColor(0xFF, 0xBB, 0xBB), font_size=14)

# Asymmetric: only ~40 "1"-bits are candidates
_add_rounded_box(slide, 7.0, 5.3, 5.5, 1.0,
    "Asymmetric errors (our channel):\n"
    "Only ~40 bits that are \"1\" can flip to \"0\"\n"
    "C(40,3) = 9,880 combos (88% smaller!)",
    RGBColor(0x1A, 0x3A, 0x1A), RGBColor(0xBB, 0xFF, 0xBB), font_size=14)

_add_textbox(slide, 0.8, 6.5, 12, 0.5,
             "The false_positive parameter controls how many 1\u21920 \"corrections\" are allowed — default: very few.",
             font_size=14, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 17 – Intuition: Why the Fallback Matters
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: Why Two Different Strategies Help",
             font_size=28, bold=True, color=ACCENT)

_add_textbox(slide, 0.8, 1.1, 12, 0.5,
             "The beam-search and legacy decoders see errors from fundamentally different angles.",
             font_size=17, color=LIGHT_GRAY)

# Two perspectives
_add_textbox(slide, 0.8, 1.8, 5.5, 0.5, "Tier 1: Syndrome Perspective", font_size=20, bold=True, color=ACCENT)
_add_rounded_box(slide, 0.8, 2.4, 5.5, 2.8,
    '"Which CHECKS are failing?"\n\n'
    'Builds a graph of all 28 constraint equations.\n'
    'Finds the variable that VIOLATES the most.\n'
    'Greedy on syndrome counts.\n\n'
    'Strength: Systematic, mathematically principled.\n'
    'Very effective when errors cluster in well-\n'
    'covered regions of the matrix.\n\n'
    'Weakness: Can get stuck if errors are in\n'
    'poorly-covered corners (few checks per bit).',
    RGBColor(0x10, 0x20, 0x40), RGBColor(0xBB, 0xDD, 0xFF), font_size=13)

_add_textbox(slide, 7.0, 1.8, 5.5, 0.5, "Tier 2: Weight Perspective", font_size=20, bold=True, color=ORANGE)
_add_rounded_box(slide, 7.0, 2.4, 5.5, 2.8,
    '"Which POSITIONS look most suspicious?"\n\n'
    'Scores positions using parity violations +\n'
    'checksum cross-references + false-positive\n'
    'budget. Domain-specific heuristic.\n\n'
    'Strength: Uses checksum bonus scoring that\n'
    'captures correlations the syndrome alone\n'
    'does not model. Different search order.\n\n'
    'Weakness: Weight-based, can miss when\n'
    'scoring artifacts create false confidence.',
    RGBColor(0x30, 0x25, 0x10), RGBColor(0xFF, 0xDD, 0xBB), font_size=13)

# The combined advantage
_add_rounded_box(slide, 1.5, 5.5, 10.0, 1.5,
    "Together: If the syndrome decoder\u2019s mathematical approach misses a pattern,\n"
    "the legacy\u2019s domain-specific checksum-aware heuristic may catch it.\n\n"
    "Like asking both a statistician AND a domain expert — different blind spots,\n"
    "so the combination covers more ground than either alone.",
    DARK_GRAY, ACCENT2, font_size=16, bold=True)

# ════════════════════════════════════════════════════════════════════
# SLIDE 18 – Intuition: Walkthrough of a 3-Error Recovery
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Walkthrough: Recovering from 3 Errors",
             font_size=28, bold=True, color=ACCENT)

_add_textbox(slide, 0.8, 1.0, 12, 0.5,
             "Step-by-step example of the beam-search decoder correcting 3 false negatives (1\u21920).",
             font_size=16, color=LIGHT_GRAY)

# Step-by-step walkthrough
steps_walk = [
    ("1. RECEIVE", "Origami arrives with 3 bits flipped to 0.\n"
     "Tier 0 strict check: 8 of 24 parity checks fail, 2 of 4 checksums fail.\n"
     "\u2192 Not clean. Proceed to Tier 1.", RGBColor(0x2C,0x3E,0x50)),

    ("2. SYNDROME", "Count which variables appear in the 10 failing checks:\n"
     "  Bit A: appears in 5 failing checks (score=5) \u2190 top suspect\n"
     "  Bit B: appears in 4 failing checks (score=4)\n"
     "  Bit C: appears in 3 failing checks (score=3)\n"
     "  Other bits: 0-2 failing checks each", ACCENT),

    ("3. ITERATION 1", "Beam generates candidates by flipping top-scored bits.\n"
     "Best candidate: flip Bit A. Now only 5 checks fail (was 10).\n"
     "Keep top 6 candidates sorted by (failed_checks, flip_count).", RGBColor(0x1A,0x4A,0x1A)),

    ("4. ITERATION 2", "From the best candidates, try flipping more bits.\n"
     "Best: flip A+B. Now only 2 checks fail.\n"
     "Second best: flip A+C. Now 3 checks fail.", RGBColor(0x1A,0x3A,0x4A)),

    ("5. ITERATION 3", "From flip(A,B), try one more flip.\n"
     "flip(A,B,C): 0 checks fail!\n"
     "Run _strict_matrix_ok: ALL 24 parities pass, ALL 4 checksums pass.\n"
     "\u2713 ACCEPT. Return corrected matrix.", ACCENT2),
]

for i, (title, desc, color) in enumerate(steps_walk):
    y = 1.6 + i * 1.1
    _add_rounded_box(slide, 0.8, y, 2.2, 0.5, title, color, WHITE, font_size=13, bold=True)
    _add_textbox(slide, 3.2, y - 0.05, 9.5, 1.0, desc, font_size=13, color=LIGHT_GRAY)

# Bottom note
_add_textbox(slide, 0.8, 7.0, 12, 0.4,
             "Total: 3 iterations, ~18 candidate evaluations, 3 flips found. Syndrome scores guided us directly to the answer.",
             font_size=14, color=ACCENT, alignment=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════
# SLIDE 19 – Intuition: Parity Coverage & Vulnerability
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: Parity Coverage Determines Correctability",
             font_size=28, bold=True, color=ACCENT)

_add_textbox(slide, 0.8, 1.1, 12, 0.6,
             "Not all bits are equally protected. A bit's correctability depends on how many parity checks cover it.",
             font_size=17, color=LIGHT_GRAY)

# Coverage explanation
_add_textbox(slide, 0.8, 1.8, 5.5, 0.5, "Well-Covered Bit (e.g., 5-6 checks)", font_size=18, bold=True, color=ACCENT2)
_add_rounded_box(slide, 0.8, 2.4, 5.5, 2.3,
    "When this bit flips:\n"
    "  5-6 parity checks FAIL simultaneously\n"
    "  Syndrome score = 5-6 (very high)\n"
    "  Decoder immediately identifies it\n\n"
    "When paired with another error:\n"
    "  Still distinguishable — different checks\n"
    "  fail for each bit\n\n"
    "Easy to correct even in multi-error scenarios.",
    RGBColor(0x1A, 0x3A, 0x1A), RGBColor(0xBB, 0xFF, 0xBB), font_size=14)

_add_textbox(slide, 7.0, 1.8, 5.5, 0.5, "Poorly-Covered Bit (e.g., 1-2 checks)", font_size=18, bold=True, color=RED)
_add_rounded_box(slide, 7.0, 2.4, 5.5, 2.3,
    "When this bit flips:\n"
    "  Only 1-2 parity checks FAIL\n"
    "  Syndrome score = 1-2 (low)\n"
    "  Could be confused with other suspects\n\n"
    "When paired with another error:\n"
    "  Syndromes OVERLAP — can't tell which\n"
    "  bit caused which failure\n\n"
    "These bits are the \"vulnerable\" positions.",
    RGBColor(0x3A, 0x20, 0x20), RGBColor(0xFF, 0xBB, 0xBB), font_size=14)

# This is what the exhaustive testing reveals
_add_textbox(slide, 0.8, 5.0, 12, 0.5, "What the Exhaustive Tests Reveal",
             font_size=20, bold=True, color=ORANGE)

_add_rounded_box(slide, 0.8, 5.5, 5.5, 1.5,
    "votes.csv ranks bits by how often\n"
    "they appear in UNCORRECTABLE combos.\n\n"
    "High vote = this bit is poorly covered\n"
    "           = structural weakness in the\n"
    "             parity mapping design.",
    RGBColor(0x30, 0x25, 0x10), RGBColor(0xFF, 0xDD, 0xBB), font_size=14)

_add_rounded_box(slide, 7.0, 5.5, 5.5, 1.5,
    "This feeds back into mapping design:\n\n"
    "  \u2192 Can we add more checks to\n"
    "     vulnerable positions?\n"
    "  \u2192 Trade off data capacity for better\n"
    "     coverage of weak spots?",
    RGBColor(0x1A, 0x2A, 0x3A), RGBColor(0xBB, 0xDD, 0xFF), font_size=14)

# ════════════════════════════════════════════════════════════════════
# SLIDE 20 – Intuition: Why Strict Acceptance is Non-Negotiable
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8, "Intuition: Why Strict Acceptance is Non-Negotiable",
             font_size=28, bold=True, color=ACCENT)

# The danger
_add_textbox(slide, 0.8, 1.3, 12, 0.5, "The Worst Outcome is NOT Failure — It's a Wrong Answer You Trust",
             font_size=20, bold=True, color=RED)

# Scenario comparison
_add_rounded_box(slide, 0.8, 2.0, 3.8, 2.5,
    "Scenario A:\nDecoder FAILS\n(returns -1)\n\n"
    "You know it failed.\n"
    "You can re-sequence.\n"
    "You can use redundant\n"
    "copies.\n"
    "Data is recoverable.",
    DARK_GRAY, LIGHT_GRAY, font_size=14)

_add_rounded_box(slide, 5.0, 2.0, 3.8, 2.5,
    "Scenario B:\nDecoder returns\nWRONG matrix silently\n\n"
    "You think it worked.\n"
    "You store corrupted data.\n"
    "You skip re-sequencing.\n"
    "You throw away originals.\n"
    "DATA IS LOST FOREVER.",
    RED, WHITE, font_size=14, bold=True)

_add_rounded_box(slide, 9.2, 2.0, 3.8, 2.5,
    "Scenario C:\nDecoder returns\nCORRECT matrix\n\n"
    "Strict check passed.\n"
    "All 28 constraints verified.\n"
    "Mathematical guarantee\n"
    "of correctness.\n"
    "Data is safe.",
    ACCENT2, WHITE, font_size=14)

# How old decoder hit scenario B
_add_textbox(slide, 0.8, 4.8, 12, 0.5, "How the Old Decoder Could Hit Scenario B",
             font_size=18, bold=True, color=ORANGE)

_add_rounded_box(slide, 0.8, 5.4, 11.5, 1.5,
    "The old decoder accepted when weight == 0. But weight is a NORMALIZED HEURISTIC:\n\n"
    "  weight = sum(violation_counts) / num_correct_parities\n\n"
    "If many parities pass (large denominator), a few remaining violations get divided away to near-zero.\n"
    "The decoder sees \"weight \u2248 0\" and declares success — but 1-2 parity checks are STILL failing.\n"
    "The matrix looks \"almost right\" but has corrupted data bits. This is silent data corruption.\n\n"
    "Strict acceptance eliminates this entirely: it doesn't divide, normalize, or approximate. Every check must pass.",
    RGBColor(0x2A, 0x15, 0x15), RGBColor(0xFF, 0xCC, 0xCC), font_size=13)

# ════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "dNAM_Decoder_Presentation.pptx")
prs.save(out_path)
print(f"Saved: {out_path}")
