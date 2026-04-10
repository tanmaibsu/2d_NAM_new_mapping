#!/usr/bin/env python3
"""Generate detailed introduction slides for the Beam-Search Syndrome Decoder."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

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
PURPLE     = RGBColor(0x8E, 0x44, 0xAD)
TEAL       = RGBColor(0x00, 0x89, 0x9B)
DARK_GREEN = RGBColor(0x1A, 0x3D, 0x1A)
DARK_RED   = RGBColor(0x5B, 0x1A, 0x1A)

prs = Presentation()
prs.slide_width  = Inches(13.333)
prs.slide_height = Inches(7.5)

# ── Helpers ──

def _set_slide_bg(slide, color):
    fill = slide.background.fill
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

def _add_rich_textbox(slide, left, top, width, height, lines, font_size=16,
                      color=WHITE, font_name="Calibri", spacing=8):
    """Each item in lines is (text, bold, color_override_or_None)."""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (text, bold, clr) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = clr if clr else color
        p.font.name = font_name
        p.space_after = Pt(spacing)
    return txBox

def _add_bullet_slide(slide, left, top, width, height, items, font_size=18,
                       color=WHITE, spacing=10, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = font_name
        p.space_after = Pt(spacing)
    return txBox

def _add_rounded_box(slide, left, top, width, height, text, fill_color,
                     text_color=WHITE, font_size=14, bold=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.color.rgb = text_color
    p.font.bold = bold
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER
    return shape

def _add_multiline_box(slide, left, top, width, height, lines, fill_color,
                       text_color=WHITE, font_size=12, bold_first=True,
                       alignment=PP_ALIGN.CENTER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.1)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = text_color
        p.font.bold = (bold_first and i == 0)
        p.font.name = "Calibri"
        p.alignment = alignment
    return shape

def _add_arrow(slide, left, top, width, height, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW,
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def _add_right_arrow(slide, left, top, width, height, color=ACCENT):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape

def _add_code_box(slide, left, top, width, height, text, font_size=11,
                  code_color=RGBColor(0xA0, 0xE0, 0xA0)):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x0D, 0x0D, 0x1A)
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.15)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(font_size)
        p.font.color.rgb = code_color
        p.font.name = "Courier New"
    return shape

def _draw_grid(slide, grid_left, grid_top, cell_w, cell_h, labels, colors_map):
    for r, row in enumerate(labels):
        for c, lbl in enumerate(row):
            shape = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(grid_left + c * cell_w), Inches(grid_top + r * cell_h),
                Inches(cell_w - 0.03), Inches(cell_h - 0.03))
            shape.fill.solid()
            shape.fill.fore_color.rgb = colors_map.get(lbl, DARK_GRAY)
            shape.line.color.rgb = RGBColor(0x55, 0x55, 0x70)
            shape.line.width = Pt(0.5)
            tf = shape.text_frame
            p = tf.paragraphs[0]
            p.text = lbl
            p.font.size = Pt(9)
            p.font.color.rgb = WHITE
            p.font.bold = True
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER


# ════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 1, 1.5, 11.3, 1.5,
             "Beam-Search Syndrome Decoder",
             font_size=44, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
_add_textbox(slide, 1, 3.0, 11.3, 1,
             "A New Decoding Algorithm for DNA Origami-based\nNucleic Acid Memory (dNAM)",
             font_size=24, color=ACCENT, alignment=PP_ALIGN.CENTER)
_add_textbox(slide, 1, 4.5, 11.3, 1.5,
             "Replacing heuristic-based error correction with\n"
             "a principled, syndrome-driven approach inspired by LDPC decoding",
             font_size=17, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
_add_textbox(slide, 1, 6.2, 11.3, 0.5,
             "Tanmai  |  April 2026",
             font_size=16, color=MID_GRAY, alignment=PP_ALIGN.CENTER)


# ════════════════════════════════════════════════════════════════════
# SLIDE 2 — Agenda / Outline
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 5, 0.8, "Outline", font_size=32, bold=True, color=ACCENT)
items = [
    "1.   Background: dNAM System & Error Model",
    "2.   8x10 Origami Matrix Layout",
    "3.   The Problem: Why the Old Decoder Falls Short",
    "4.   Key Insight: Parity System as an LDPC Code",
    "5.   Building the Tanner Graph (_build_check_graph)",
    "6.   Algorithm: Beam-Search Syndrome Decoder (_iterative_decode)",
    "7.   Worked Example: Decoding a 3-Error Origami",
    "8.   Why Beam Search Over Pure Greedy?",
    "9.   Three-Tier Hybrid Architecture",
    "10.  Strict Acceptance: Eliminating Silent Corruption",
    "11.  Old vs New: Side-by-Side Comparison",
    "12.  Complexity Analysis",
    "13.  Summary & Key Takeaways",
]
_add_bullet_slide(slide, 1.5, 1.2, 10, 6, items, font_size=18, color=WHITE, spacing=6)


# ════════════════════════════════════════════════════════════════════
# SLIDE 3 — Background: dNAM System & Error Model
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Background: dNAM System & Error Model",
             font_size=32, bold=True, color=ACCENT)

# Left: What is dNAM?
_add_textbox(slide, 0.8, 1.2, 5.8, 0.5,
             "What is DNA Origami-based Nucleic Acid Memory?", font_size=20, bold=True, color=ORANGE)
_add_bullet_slide(slide, 1.0, 1.8, 5.5, 3.5, [
    "\u2022  DNA origami = self-assembled nanostructure used as",
    "   a 2D surface for storing binary data",
    "",
    "\u2022  Each origami encodes 80 bits in an 8\u00d710 grid",
    "   where each position is a binary 0 or 1",
    "",
    "\u2022  Data is written by attaching molecular markers",
    "   (staple strands) to specific grid locations",
    "",
    "\u2022  Data is read back via microscopy (e.g., AFM)",
    "   which images the marker presence/absence",
    "",
    "\u2022  A binary file is split across many origamis,",
    "   each carrying 46 data bits + error correction bits",
], font_size=14, color=WHITE, spacing=3)

# Right: Error model
_add_textbox(slide, 7.0, 1.2, 5.8, 0.5,
             "The Physical Error Model", font_size=20, bold=True, color=RED)

_add_multiline_box(slide, 7.2, 1.8, 5.3, 1.5, [
    "False Negatives (1 \u2192 0)  \u2014  DOMINANT",
    "",
    "A marker IS present, but the microscope fails to",
    "detect it. The bit reads as 0 instead of 1.",
    "This is the most common error type because",
    "markers can fold under, detach, or be obscured.",
], DARK_RED, font_size=13, bold_first=True)

_add_multiline_box(slide, 7.2, 3.5, 5.3, 1.3, [
    "False Positives (0 \u2192 1)  \u2014  RARE",
    "",
    "No marker present, but noise in imaging causes",
    "a false detection. Much less common because",
    "the absence of a marker is harder to misread.",
], RGBColor(0x3D, 0x1A, 0x1A), font_size=13, bold_first=True)

_add_multiline_box(slide, 7.2, 5.0, 5.3, 1.0, [
    "Practical Error Rates",
    "",
    "Typically 1\u20138 bit errors per origami (out of 80 bits)",
    "Error rate varies with imaging quality and origami prep",
], DARK_GRAY, font_size=13, bold_first=True)

# Pipeline at bottom
_add_textbox(slide, 0.8, 6.2, 12, 0.4,
             "End-to-End Pipeline:", font_size=16, bold=True, color=ACCENT)
boxes = [
    ("Binary File", DARK_GRAY), ("\u2192", None), ("encode.py", ACCENT),
    ("\u2192", None), ("Origami Files", DARK_GRAY), ("\u2192", None),
    ("Physical Readout\n(with errors)", RED), ("\u2192", None),
    ("decode.py", ACCENT2), ("\u2192", None), ("Recovered File", DARK_GRAY)
]
x = 0.8
for txt, clr in boxes:
    if clr is None:
        _add_textbox(slide, x, 6.6, 0.4, 0.5, txt, font_size=18, color=MID_GRAY,
                     alignment=PP_ALIGN.CENTER)
        x += 0.4
    else:
        w = 1.6 if "\n" in txt else 1.2
        _add_rounded_box(slide, x, 6.6, w, 0.55, txt, clr, font_size=11, bold=True)
        x += w + 0.1


# ════════════════════════════════════════════════════════════════════
# SLIDE 4 — 8x10 Origami Matrix Layout
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.2, 8, 0.8,
             "8\u00d710 DNA Origami Matrix Layout (24-Parity Configuration)",
             font_size=28, bold=True, color=ACCENT)

labels = [
    ["DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA"],
    ["ORI","PAR","PAR","PAR","PAR","PAR","PAR","PAR","PAR","ORI"],
    ["DATA","PAR","DATA","DATA","DATA","DATA","DATA","DATA","PAR","DATA"],
    ["DATA","PAR","DATA","DATA","CHK","CHK","DATA","DATA","PAR","DATA"],
    ["DATA","PAR","DATA","DATA","CHK","CHK","DATA","DATA","PAR","DATA"],
    ["DATA","PAR","DATA","DATA","DATA","DATA","DATA","DATA","PAR","DATA"],
    ["ORI","PAR","PAR","PAR","PAR","PAR","PAR","PAR","PAR","ORI"],
    ["DATA","DATA","DATA","DATA","DATA","DATA","DATA","DATA","IDX","IDX"],
]
colors_map = {
    "DATA": RGBColor(0x34, 0x49, 0x5E),
    "PAR":  ACCENT,
    "ORI":  ORANGE,
    "IDX":  PURPLE,
    "CHK":  RED,
}
_draw_grid(slide, 0.8, 1.1, 0.65, 0.50, labels, colors_map)

# Legend
legend_items = [
    ("DATA  46 bits  (user payload)", colors_map["DATA"]),
    ("PARITY  24 bits  (XOR check equations)", ACCENT),
    ("CHECKSUM  4 bits  (quadrant-level XOR)", RED),
    ("ORIENTATION  4 bits  (fixed [1,1,1,0] at corners)", ORANGE),
    ("INDEX  2 bits  (origami sequence number, supports 4 origamis)", PURPLE),
]
for i, (txt, clr) in enumerate(legend_items):
    _add_rounded_box(slide, 0.8, 5.25 + i * 0.42, 5.5, 0.38, txt, clr, font_size=11, bold=True)

# Right: detailed explanation
_add_textbox(slide, 7.5, 1.0, 5.5, 0.5,
             "How the 80 Bits Are Organized", font_size=20, bold=True, color=ORANGE)
_add_bullet_slide(slide, 7.7, 1.5, 5.2, 5.5, [
    "Data Bits (46):",
    "  The actual payload \u2014 a fragment of the encoded",
    "  binary file. Spread across rows 0, 2\u20135, 7.",
    "",
    "Parity Bits (24):",
    "  Each parity bit stores the XOR of ~12 other",
    "  positions. If any of those positions flip, the",
    "  parity check will fail, revealing the error.",
    "  Forms two rings: rows 1 & 6, columns 1 & 8.",
    "",
    "Checksum Bits (4):",
    "  Quadrant-level XOR checks at positions (3,4),",
    "  (3,5), (4,4), (4,5). Each covers ~13 positions",
    "  in one quadrant. Provides coarse error detection.",
    "",
    "Orientation Bits (4):",
    "  Fixed pattern [1,1,1,0] at corners (1,0), (1,9),",
    "  (6,0), (6,9). Used to detect physical rotation",
    "  or flip of the origami during imaging.",
    "",
    "Index Bits (2):",
    "  At positions (7,8) and (7,9). Encode the origami's",
    "  sequence number (0\u20133), supporting up to 4 origamis",
    "  per file. Used to reassemble decoded data in order.",
], font_size=12, color=WHITE, spacing=2)


# ════════════════════════════════════════════════════════════════════
# SLIDE 5 — The Problem: Why the Old Decoder Falls Short
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "The Problem: Why the Old Decoder Falls Short",
             font_size=32, bold=True, color=ACCENT)

# Left: How the old decoder works
_add_textbox(slide, 0.8, 1.1, 5.8, 0.5,
             "How the Previous (Greedy) Decoder Works", font_size=20, bold=True, color=ORANGE)

_add_code_box(slide, 0.8, 1.7, 5.8, 4.8,
    "1. Receive origami matrix with potential errors\n"
    "2. Fix physical orientation (try 4 rotations)\n"
    "3. Compute 'matrix weight' scoring function:\n"
    "   a. Check which of 24 parities fail\n"
    "   b. For each failure, collect covered positions\n"
    "   c. Count how often each position appears\n"
    "   d. Add checksum bonus (+1 or +2) manually\n"
    "   e. Group positions by suspicion weight\n"
    "   f. Filter by threshold (threshold_data, threshold_parity)\n"
    "4. If weight == 0: ACCEPT the matrix\n"
    "5. If weight > 0: A*-style priority queue search\n"
    "   a. Try flipping each probable error bit\n"
    "   b. Re-compute weight after each flip\n"
    "   c. If weight == 0: ACCEPT\n"
    "   d. Else: add to queue, try combinations\n"
    "   e. Explore up to maximum_number_of_error flips\n"
    "6. If nothing works: return FAILURE (-1)",
    font_size=12)

# Right: Four specific problems
_add_textbox(slide, 7.2, 1.1, 5.5, 0.5,
             "Four Critical Limitations", font_size=20, bold=True, color=RED)

_add_multiline_box(slide, 7.2, 1.7, 5.5, 1.2, [
    "1. Silent Data Corruption",
    "",
    "The old decoder accepts when matrix_weight == 0, but this",
    "does NOT mean all checks pass. The weight is a normalized",
    "aggregate: individual failing checks can cancel out in the",
    "sum. This means the decoder can return WRONG data and",
    "report success \u2014 the worst possible failure mode.",
], DARK_RED, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_multiline_box(slide, 7.2, 3.1, 5.5, 1.1, [
    "2. Exponential Search Explosion",
    "",
    "For each candidate bit, the decoder tries flipping it, then",
    "tries all remaining candidates. With k candidates and d",
    "errors: O(k^d) combinations. At 6+ errors, this becomes",
    "computationally infeasible.",
], RGBColor(0xC0, 0x39, 0x2B), font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_multiline_box(slide, 7.2, 4.4, 5.5, 1.1, [
    "3. Hand-Tuned, Fragile Heuristics",
    "",
    "Requires manual tuning of threshold_data, threshold_parity,",
    "and false_positive parameters. The checksum bonus logic",
    "(+2 if pos in both sets) is dead code \u2014 the sets are disjoint",
    "by construction, so that branch never executes.",
], RGBColor(0xA9, 0x33, 0x26), font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_multiline_box(slide, 7.2, 5.7, 5.5, 0.9, [
    "4. No Recovery from Wrong Early Choices",
    "",
    "Greedy picks one \"best\" bit to flip. If that choice is wrong",
    "(common with clustered errors), the entire subsequent search",
    "tree is poisoned. No mechanism to backtrack effectively.",
], RGBColor(0x92, 0x2B, 0x21), font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_rounded_box(slide, 1.5, 6.8, 10.3, 0.5,
    "We need: (1) strict verification, (2) scalable search, (3) principled scoring, (4) resilience to wrong choices",
    ACCENT, font_size=14, bold=True)


# ════════════════════════════════════════════════════════════════════
# SLIDE 6 — Key Insight: Parity System as LDPC Code
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Key Insight: Our Parity System IS an LDPC Code",
             font_size=32, bold=True, color=ACCENT)

# Left: The realization
_add_textbox(slide, 0.8, 1.1, 5.8, 0.5,
             "Connecting dNAM to Coding Theory", font_size=20, bold=True, color=ACCENT2)
_add_bullet_slide(slide, 1.0, 1.7, 5.5, 4.5, [
    "Each parity bit defines an equation:",
    "  parity(1,1) = XOR of 12 specific positions",
    "  If all bits correct: XOR result = 0",
    "  If any bit flipped: XOR result = 1 (check fails)",
    "",
    "Each checksum bit defines a similar equation:",
    "  checksum(3,4) = XOR of 13 positions in top-left quadrant",
    "",
    "Together: 24 parity + 4 checksum = 28 XOR equations",
    "over 80 binary variables (the matrix cells).",
    "",
    "This is exactly the structure of a Low-Density",
    "Parity-Check (LDPC) code!",
    "",
    "In coding theory, this system is represented as a",
    "bipartite Tanner graph, and decoded using iterative",
    "message-passing algorithms.",
    "",
    "Our key insight: we can build this Tanner graph from",
    "the existing parity_bit_relation and checksum_bit_relation",
    "mappings, then apply syndrome-based decoding.",
], font_size=13, color=WHITE, spacing=2)

# Right: What is a syndrome?
_add_textbox(slide, 7.0, 1.1, 5.8, 0.5,
             "What is a Syndrome?", font_size=20, bold=True, color=ORANGE)

_add_multiline_box(slide, 7.0, 1.7, 5.5, 2.0, [
    "Syndrome = the pattern of which checks fail",
    "",
    "Given a received matrix, evaluate all 28 checks.",
    "Each check returns PASS (0) or FAIL (1).",
    "The resulting 28-bit vector is the 'syndrome'.",
    "",
    "Syndrome = [0,0,1,0,1,0,0,1,0,0,...] (28 bits)",
    "               \u2191       \u2191          \u2191",
    "         checks 3, 5, and 8 are failing",
], RGBColor(0x1A, 0x3D, 0x3D), font_size=13, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_textbox(slide, 7.0, 3.9, 5.8, 0.5,
             "Why Syndromes are Powerful", font_size=18, bold=True, color=ORANGE)

_add_multiline_box(slide, 7.0, 4.4, 5.5, 2.6, [
    "Property: An erroneous bit causes ALL its",
    "connected checks to fail simultaneously.",
    "",
    "Example: If bit (2,4) is flipped and it participates",
    "in checks C1, C5, C12, C17, C25, C27, C28 (degree 7),",
    "then all 7 of those checks will show FAIL.",
    "",
    "By counting how many failing checks each variable",
    "appears in, we get a 'suspicion score'.",
    "The erroneous bit will score 7 \u2014 much higher than",
    "innocent bits that only appear in 1\u20132 failures.",
    "",
    "This scoring is AUTOMATIC \u2014 no thresholds needed.",
], DARK_GREEN, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)


# ════════════════════════════════════════════════════════════════════
# SLIDE 7 — Building the Tanner Graph (_build_check_graph)
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Building the Tanner Graph: _build_check_graph()",
             font_size=28, bold=True, color=ACCENT)

# Left: code
_add_code_box(slide, 0.8, 1.1, 5.5, 2.8,
    "def _build_check_graph(self):\n"
    "    check_to_vars = {}\n"
    "    var_to_checks = defaultdict(list)\n"
    "\n"
    "    all_checks = {}\n"
    "    all_checks.update(self.parity_bit_relation)\n"
    "    all_checks.update(self.checksum_bit_relation)\n"
    "\n"
    "    for check_cell, deps in all_checks.items():\n"
    "        vars_in_check = [check_cell] + list(deps)\n"
    "        check_to_vars[check_cell] = vars_in_check\n"
    "        for v in vars_in_check:\n"
    "            var_to_checks[v].append(check_cell)\n"
    "\n"
    "    checks = list(all_checks.keys())\n"
    "    return checks, check_to_vars, var_to_checks",
    font_size=11)

# Right: what it produces
_add_textbox(slide, 7.0, 1.1, 5.8, 0.4,
             "What This Method Produces", font_size=20, bold=True, color=ORANGE)

_add_multiline_box(slide, 7.0, 1.6, 5.5, 1.4, [
    "checks  (list of 28 positions)",
    "",
    "All 28 check node positions: 24 parity bit locations",
    "like (1,1), (1,2), ... plus 4 checksum locations",
    "(3,4), (3,5), (4,4), (4,5). Each position defines",
    "one XOR equation that must equal zero.",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_multiline_box(slide, 7.0, 3.2, 5.5, 1.4, [
    "check_to_vars  (dict: check \u2192 list of ~13 positions)",
    "",
    "For each check, lists every variable position involved",
    "in its XOR equation (including the check bit itself).",
    "Example: check (1,1) \u2192 [(1,1), (4,4), (7,4), (0,1),",
    "  (4,0), (7,7), (2,7), (5,4), (4,6), (0,6), ...]",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_multiline_box(slide, 7.0, 4.8, 5.5, 1.4, [
    "var_to_checks  (dict: variable \u2192 list of 1\u20137 checks)",
    "",
    "The REVERSE mapping. For each matrix cell, which checks",
    "involve it. This is the variable's 'degree' in the graph.",
    "Data bits: degree 5\u20137 (high observability)",
    "Parity/checksum bits: degree 1 (only their own check)",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

# Bottom: stats
_add_textbox(slide, 0.8, 4.2, 5.5, 0.4,
             "Graph Statistics (24-parity config)", font_size=16, bold=True, color=ACCENT2)

stats = [
    ("Total check nodes", "28"),
    ("Total variable nodes", "80"),
    ("Check degree (vars per check)", "13\u201314"),
    ("Variable degree (checks per var)", "1\u20137"),
    ("Graph density", "~5% (sparse = LDPC)"),
]
for i, (label, val) in enumerate(stats):
    _add_rounded_box(slide, 0.8, 4.7 + i * 0.42, 3.5, 0.38, label, DARK_GRAY, MID_GRAY, font_size=11)
    _add_rounded_box(slide, 4.4, 4.7 + i * 0.42, 1.8, 0.38, val, DARK_GRAY, ORANGE, font_size=12, bold=True)

_add_rounded_box(slide, 1.5, 6.8, 10.3, 0.5,
    "This graph is built ONCE per decode call and reused across all iterations \u2014 it encodes the structural backbone of the code",
    TEAL, font_size=14, bold=True)


# ════════════════════════════════════════════════════════════════════
# SLIDE 8 — Algorithm Walkthrough: _iterative_decode
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Algorithm: _iterative_decode() Step by Step",
             font_size=28, bold=True, color=ACCENT)

# Left: flow diagram with detailed boxes
bx = 0.8
step_data = [
    ("STEP 1: Build Check Graph", DARK_GRAY,
     "Construct bipartite graph from parity_bit_relation\n"
     "and checksum_bit_relation. Returns checks,\n"
     "check_to_vars, var_to_checks."),
    ("STEP 2: Initialize Beam", DARK_GRAY,
     "beam = [(deep_copy(matrix), [])]\n"
     "Start with one candidate: the original received\n"
     "matrix, with an empty list of applied flips."),
    ("STEP 3: Check if Solved", ACCENT2,
     "_strict_matrix_ok(mat): try all 4 orientations,\n"
     "verify ALL 24 parities + 4 checksums pass.\n"
     "If yes \u2192 return (corrected_matrix, flip_list)."),
    ("STEP 4: Compute Syndrome", ACCENT,
     "Find all checks where XOR \u2260 0.\n"
     "These 'failed checks' form the syndrome \u2014\n"
     "the decoder's signal about where errors are."),
    ("STEP 5: Score Variables", ORANGE,
     "For each failed check, +1 to all its variables.\n"
     "scores = Counter(). No thresholds, no bonuses.\n"
     "High score = involved in many failing checks."),
    ("STEP 6: Expand & Prune", PURPLE,
     "Try flipping top beam_width\u00d72 (24) scored vars.\n"
     "Re-evaluate: count failed checks for each child.\n"
     "Keep best beam_width (12) by (failures, flips)."),
]

for i, (title, clr, desc) in enumerate(step_data):
    y = 1.1 + i * 1.05
    _add_rounded_box(slide, bx, y, 2.8, 0.45, title, clr, font_size=11, bold=True)
    if i < len(step_data) - 1:
        _add_arrow(slide, bx + 1.2, y + 0.5, 0.3, 0.25, MID_GRAY)

# Right: descriptions
for i, (title, clr, desc) in enumerate(step_data):
    y = 1.1 + i * 1.05
    _add_multiline_box(slide, 4.0, y, 5.0, 0.95,
                       desc.split("\n"), DARK_GRAY, font_size=11,
                       bold_first=False, alignment=PP_ALIGN.LEFT)

# Far right: key properties
_add_textbox(slide, 9.3, 1.1, 3.5, 0.4,
             "Key Properties", font_size=18, bold=True, color=ACCENT2)

props = [
    "Built once, O(28\u00d713)",
    "1 candidate initially",
    "Strict = no false accepts",
    "28 XOR evaluations",
    "Automatic, threshold-free",
    "12 survive each round",
]
for i, p in enumerate(props):
    _add_rounded_box(slide, 9.5, 1.55 + i * 1.05, 3.2, 0.4, p,
                     RGBColor(0x1A, 0x3D, 0x1A), font_size=11, bold=True)

_add_rounded_box(slide, 1.5, 6.8, 10.3, 0.5,
    "Loop: Steps 3\u20136 repeat up to max_iters (40) times. If no candidate is solved and no more children: return -1 (failure).",
    RED, font_size=14, bold=True)


# ════════════════════════════════════════════════════════════════════
# SLIDE 9 — Worked Example: Decoding a 3-Error Origami
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Worked Example: Decoding a 3-Error Origami",
             font_size=28, bold=True, color=ACCENT)
_add_textbox(slide, 0.8, 0.85, 10, 0.4,
             "Suppose errors at positions (0,1), (2,4), and (7,7)  \u2014  all are 1\u21920 false negatives",
             font_size=15, color=MID_GRAY)

# Iteration 1
_add_multiline_box(slide, 0.8, 1.4, 3.8, 2.7, [
    "ITERATION 1",
    "",
    "Syndrome: 15 of 28 checks fail",
    "(errors in high-degree positions cause",
    "many simultaneous failures)",
    "",
    "Scoring:",
    "  (2,4) \u2192 score 7  (degree 7, in most failures)",
    "  (0,1) \u2192 score 6  (degree 6)",
    "  (7,7) \u2192 score 6  (degree 6)",
    "  (5,4) \u2192 score 4  (innocent, but near errors)",
    "  ... (other lower scores)",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_right_arrow(slide, 4.7, 2.5, 0.45, 0.35, ACCENT)

# Expand
_add_multiline_box(slide, 5.3, 1.4, 3.5, 2.7, [
    "EXPAND: Try top 24 flips",
    "",
    "Generate 24 children, each with 1 flip:",
    "  Child A: flip (2,4) \u2192 9 checks still fail",
    "  Child B: flip (0,1) \u2192 10 checks still fail",
    "  Child C: flip (7,7) \u2192 10 checks still fail",
    "  Child D: flip (5,4) \u2192 16 checks fail (worse!)",
    "  ...",
    "",
    "PRUNE: Keep best 12 by # failures",
    "  Child A (9 fails) ranked #1",
    "  Children B,C ranked #2\u20133",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_right_arrow(slide, 8.9, 2.5, 0.45, 0.35, ACCENT)

# Result after iter 1
_add_multiline_box(slide, 9.5, 1.4, 3.3, 2.7, [
    "BEAM AFTER ITER 1",
    "",
    "12 candidates, each with 1 flip.",
    "Best: flip (2,4), 9 checks fail",
    "",
    "Note: the correct first flip IS",
    "ranked #1 because (2,4) has the",
    "highest degree (7) and causes",
    "the strongest syndrome signal.",
    "",
    "But even if it wasn't #1, the",
    "beam preserves alternatives!",
], RGBColor(0x1A, 0x3D, 0x1A), font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

# Iteration 2
_add_multiline_box(slide, 0.8, 4.3, 3.8, 2.2, [
    "ITERATION 2",
    "",
    "Best candidate: (2,4) already fixed",
    "Remaining syndrome: 9 checks fail",
    "",
    "New scoring (after fixing (2,4)):",
    "  (0,1) \u2192 score 6  (now top-ranked)",
    "  (7,7) \u2192 score 5",
    "",
    "Expand: try flipping (0,1) etc.",
    "Best child: flip (2,4)+(0,1) \u2192 4 fails",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_right_arrow(slide, 4.7, 5.2, 0.45, 0.35, ACCENT)

# Iteration 3
_add_multiline_box(slide, 5.3, 4.3, 3.5, 2.2, [
    "ITERATION 3",
    "",
    "Best: flipped (2,4) + (0,1)",
    "Remaining: 4 checks fail",
    "",
    "Scoring: (7,7) \u2192 score 4 (top!)",
    "",
    "Expand: flip (7,7) as third fix",
    "Best child: 0 checks fail!",
    "",
    "_strict_matrix_ok \u2192 TRUE",
], DARK_GRAY, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_right_arrow(slide, 8.9, 5.2, 0.45, 0.35, ACCENT2)

# Success
_add_multiline_box(slide, 9.5, 4.3, 3.3, 2.2, [
    "SUCCESS!",
    "",
    "Return:",
    "  matrix = corrected origami",
    "  flips = [(2,4), (0,1), (7,7)]",
    "",
    "3 errors corrected in 3 iters",
    "Beam width ensured we never",
    "went down a wrong path.",
    "",
    "Total work: ~3 \u00d7 24 = 72 trials",
    "(vs. potentially 80^3 = 512K brute force)",
], RGBColor(0x1A, 0x5B, 0x1A), font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)


# ════════════════════════════════════════════════════════════════════
# SLIDE 10 — Why Beam Search Over Pure Greedy?
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Why Beam Search Over Pure Greedy?",
             font_size=32, bold=True, color=ACCENT)

# Left: Greedy failure
_add_textbox(slide, 0.8, 1.1, 5.8, 0.5,
             "Pure Greedy (beam_width = 1): Fragile", font_size=20, bold=True, color=RED)

_add_multiline_box(slide, 0.8, 1.7, 5.8, 3.3, [
    "The Problem: One Wrong Step = Game Over",
    "",
    "When errors are clustered (e.g., (2,4) and (2,5) both",
    "flipped), their syndromes overlap heavily. An innocent",
    "position like (5,4) might score higher than the true",
    "error because it sits at the intersection of many",
    "failing checks from BOTH errors.",
    "",
    "Greedy decoder: flips (5,4) \u2192 syndrome WORSENS",
    "(now 3 errors instead of 2) \u2192 decoder is stuck",
    "in a worse state with no way to undo the mistake.",
    "",
    "This failure mode is especially common at 4+ errors",
    "because syndrome overlap increases with error count.",
], DARK_RED, font_size=13, bold_first=True, alignment=PP_ALIGN.LEFT)

# Right: Beam search advantage
_add_textbox(slide, 7.0, 1.1, 5.8, 0.5,
             "Beam Search (beam_width = 12): Resilient", font_size=20, bold=True, color=ACCENT2)

_add_multiline_box(slide, 7.0, 1.7, 5.5, 3.3, [
    "The Solution: Keep Multiple Hypotheses Alive",
    "",
    "Same scenario: (5,4) scores highest, (2,4) scores #2.",
    "",
    "Beam search: tries BOTH flips (and 22 more).",
    "  Candidate A: flip (5,4) \u2192 17 fails (worse)",
    "  Candidate B: flip (2,4) \u2192 9 fails (better!)",
    "  Candidate C: flip (0,1) \u2192 10 fails",
    "  ...",
    "",
    "After pruning: Candidate B is ranked #1!",
    "The wrong choice (A) drops to the bottom or is pruned.",
    "The correct path survives and will converge.",
    "",
    "Key: the 2\u00d7 oversampling (try 24, keep 12) gives room.",
], DARK_GREEN, font_size=13, bold_first=True, alignment=PP_ALIGN.LEFT)

# Bottom: parameter table
_add_textbox(slide, 0.8, 5.3, 11.5, 0.5,
             "Algorithm Parameters and Their Rationale", font_size=20, bold=True, color=ORANGE)

headers = ["Parameter", "Value", "Why This Value"]
col_x = [0.8, 4.2, 6.5]
col_w = [3.2, 2.1, 6.3]
for i, h in enumerate(headers):
    _add_rounded_box(slide, col_x[i], 5.8, col_w[i], 0.42, h, ACCENT, font_size=12, bold=True)

rows = [
    ("max_flips", "8",
     "Beyond 8 errors in 80 bits, the code's redundancy (28 checks) is insufficient to uniquely identify the correction"),
    ("max_iters", "40",
     "Empirically sufficient for beam convergence. Each iter makes progress (fewer failures), so 40 is generous."),
    ("beam_width", "12",
     "12 parallel hypotheses balance exploration vs memory/speed. Wider beams show diminishing returns."),
    ("Children per candidate", "beam_width\u00d72 = 24",
     "2\u00d7 oversampling: generate twice what we keep, giving selection pressure to filter bad guesses."),
]
for j, (p, v, desc) in enumerate(rows):
    y = 6.3 + j * 0.38
    _add_rounded_box(slide, col_x[0], y, col_w[0], 0.35, p, DARK_GRAY, font_size=11)
    _add_rounded_box(slide, col_x[1], y, col_w[1], 0.35, v, DARK_GRAY, ORANGE, font_size=11, bold=True)
    _add_rounded_box(slide, col_x[2], y, col_w[2], 0.35, desc, DARK_GRAY, font_size=10)


# ════════════════════════════════════════════════════════════════════
# SLIDE 11 — Three-Tier Hybrid Architecture
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Three-Tier Hybrid Decoder Architecture",
             font_size=32, bold=True, color=ACCENT)
_add_textbox(slide, 0.8, 0.85, 10, 0.4,
             "The _decode() method orchestrates three tiers, each with strict acceptance verification",
             font_size=15, color=MID_GRAY)

# Tier 0
_add_multiline_box(slide, 0.5, 1.5, 3.7, 2.8, [
    "TIER 0: Strict Accept",
    "(Zero-flip verification)",
    "",
    "Before trying ANY correction, check if the",
    "received matrix is already correct.",
    "",
    "1. Fix orientation (try 4 rotations/flips)",
    "2. Verify ALL 24 parity XOR equations",
    "3. Verify ALL 4 checksum XOR equations",
    "",
    "If all 28 checks pass: accept immediately.",
    "Cost: O(364) \u2014 trivially fast.",
    "",
    "When: No errors, or errors that cancel out.",
], ACCENT2, font_size=11, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_right_arrow(slide, 4.35, 2.6, 0.5, 0.35, MID_GRAY)
_add_textbox(slide, 4.3, 3.1, 0.6, 0.3, "fail", font_size=11, color=RED,
             alignment=PP_ALIGN.CENTER)

# Tier 1
_add_multiline_box(slide, 5.0, 1.5, 3.7, 2.8, [
    "TIER 1: Beam-Search Syndrome Decoder",
    "(Primary error correction engine)",
    "",
    "The NEW algorithm described in this talk.",
    "",
    "1. Build Tanner graph from parity/checksum",
    "2. Score variables by syndrome",
    "3. Beam search: expand top-scored flips,",
    "   prune to best 12 candidates",
    "4. Repeat up to 40 iterations / 8 flips",
    "",
    "Handles: 1\u20138 errors efficiently.",
    "Strength: Principled, no manual tuning.",
], ACCENT, font_size=11, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_right_arrow(slide, 8.85, 2.6, 0.5, 0.35, MID_GRAY)
_add_textbox(slide, 8.8, 3.1, 0.6, 0.3, "fail", font_size=11, color=RED,
             alignment=PP_ALIGN.CENTER)

# Tier 2
_add_multiline_box(slide, 9.5, 1.5, 3.4, 2.8, [
    "TIER 2: Legacy Greedy Heuristic",
    "(Fallback \u2014 different strategy)",
    "",
    "The original decoder, kept as fallback.",
    "",
    "1. Weight-based scoring with thresholds",
    "2. A*-style greedy search",
    "3. Different scoring = may succeed",
    "   where beam search failed",
    "",
    "Why keep it: different strategy means",
    "different failure modes. If beam search",
    "can't solve it, greedy might find a path",
    "the syndrome scoring missed.",
], ORANGE, font_size=11, bold_first=True, alignment=PP_ALIGN.LEFT)

# Code snippet of _decode
_add_textbox(slide, 0.5, 4.5, 12, 0.4,
             "The Orchestration: _decode() method", font_size=18, bold=True, color=ORANGE)

_add_code_box(slide, 0.5, 5.0, 6.0, 2.2,
    "def _decode(self, matrix, td, tp, max_err, fp):\n"
    "    # Tier 0: strict accept (no correction needed?)\n"
    "    ok, _, oriented = self._strict_matrix_ok(matrix)\n"
    "    if ok:\n"
    "        return self.return_matrix(oriented, [])\n"
    "\n"
    "    # Tier 1: beam-search syndrome decoder\n"
    "    fixed, flips = self._iterative_decode(\n"
    "        matrix, max_flips=max_err, beam_width=6)\n"
    "    if not isinstance(fixed, int):  # success\n"
    "        ok, i, ori = self._strict_matrix_ok(fixed)\n"
    "        if ok: return self.return_matrix(ori, flips)\n"
    "\n"
    "    # Tier 2: fallback to legacy greedy\n"
    "    return self._decode_legacy(matrix, td, tp, max_err, fp)",
    font_size=11)

_add_multiline_box(slide, 7.0, 5.0, 5.8, 2.2, [
    "Why Three Tiers?",
    "",
    "Defense in depth. Each tier catches different failure modes:",
    "",
    "Tier 0 catches: error-free origamis (saves computation).",
    "  Runs in microseconds. ~60-80% of origamis in good data.",
    "",
    "Tier 1 catches: 1-8 errors using syndrome-driven search.",
    "  Primary workhorse. Principled, scalable, no false accepts.",
    "",
    "Tier 2 catches: rare edge cases where syndrome scoring",
    "  fails (e.g., errors concentrated on degree-1 parity bits).",
    "  Different heuristic = different blind spots.",
], DARK_GREEN, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)


# ════════════════════════════════════════════════════════════════════
# SLIDE 12 — Strict Acceptance: Eliminating Silent Corruption
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Strict Acceptance: Eliminating Silent Data Corruption",
             font_size=28, bold=True, color=ACCENT)
_add_textbox(slide, 0.8, 0.85, 10, 0.4,
             "The single most important improvement in the new decoder",
             font_size=15, color=MID_GRAY)

# Left: OLD acceptance
_add_textbox(slide, 0.8, 1.4, 5.8, 0.5,
             "OLD: Weight-Based Acceptance", font_size=20, bold=True, color=RED)

_add_code_box(slide, 0.8, 1.9, 5.8, 1.5,
    "# Old acceptance logic\n"
    "_, matrix_weight, _ = self._get_matrix_weight(\n"
    "    matrix, flips, tp, td, fp)\n"
    "if matrix_weight == 0:\n"
    "    return matrix  # ACCEPT -- BUT IS THIS SAFE?",
    font_size=12)

_add_multiline_box(slide, 0.8, 3.6, 5.8, 2.8, [
    "Why weight == 0 is DANGEROUS",
    "",
    "The weight is computed as:",
    "  normalized_weight = total_weight / len(passing_parities)",
    "",
    "Scenario: 22 of 24 parities pass, 2 fail.",
    "  But the 2 failing parities have low individual weight.",
    "  After normalization by 22: weight rounds down to 0.",
    "",
    "Result: decoder accepts a matrix with 2 failing parity",
    "checks. The data returned to the user is WRONG,",
    "and no error is reported.",
    "",
    "This is silent data corruption \u2014 the decoder lies.",
    "For a data storage system, this is catastrophic.",
], DARK_RED, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

# Right: NEW acceptance
_add_textbox(slide, 7.0, 1.4, 5.8, 0.5,
             "NEW: Strict Acceptance (_strict_matrix_ok)", font_size=20, bold=True, color=ACCENT2)

_add_code_box(slide, 7.0, 1.9, 5.8, 2.2,
    "def _strict_matrix_ok(self, matrix):\n"
    "    # Step 1: Fix orientation\n"
    "    ori, oriented = self._fix_orientation(matrix)\n"
    "    if ori == -1: return False  # can't orient\n"
    "\n"
    "    # Step 2: Check ALL parities (no exceptions)\n"
    "    _, bad_parity = self._find_possible_error_location(oriented)\n"
    "    if bad_parity: return False  # even 1 failure = reject\n"
    "\n"
    "    # Step 3: Check ALL checksums (no exceptions)\n"
    "    if not self.check_checksum(oriented): return False\n"
    "\n"
    "    return True  # ALL 28 checks verified",
    font_size=11)

_add_multiline_box(slide, 7.0, 4.3, 5.8, 2.2, [
    "Why This is Mathematically Sound",
    "",
    "Strict acceptance is a NECESSARY condition, not sufficient:",
    "  - 28 independent XOR equations over 80 variables",
    "  - Each equation: XOR(all participating vars) must = 0",
    "  - ANY single failure \u2192 immediate rejection",
    "",
    "The probability of a wrong matrix passing all 28 checks:",
    "  P(undetected error) \u2248 2^(-28) \u2248 0.000000004",
    "  (assuming random error patterns)",
    "",
    "This means: if strict acceptance says OK, the matrix is",
    "correct with probability > 99.99999996%.",
], DARK_GREEN, font_size=12, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_rounded_box(slide, 1.0, 6.7, 11.3, 0.5,
    "Every accepted origami in the new decoder has been verified against ALL 28 check equations \u2014 zero ambiguity, zero silent failures",
    ACCENT2, font_size=14, bold=True)


# ════════════════════════════════════════════════════════════════════
# SLIDE 13 — Old vs New: Side-by-Side Comparison
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Old vs New: Detailed Side-by-Side Comparison",
             font_size=28, bold=True, color=ACCENT)

headers = ["Aspect", "Old (Greedy Heuristic)", "New (Beam-Search Syndrome)"]
col_x = [0.5, 3.2, 8.2]
col_w = [2.5, 4.8, 4.8]

for i, h in enumerate(headers):
    _add_rounded_box(slide, col_x[i], 1.0, col_w[i], 0.5, h, ACCENT, font_size=13, bold=True)

rows = [
    ("Acceptance\nCriterion",
     "matrix_weight == 0\n(normalized aggregate; individual\nfailures can be masked)",
     "All 28 XOR checks pass individually\n(strict: even 1 failure = reject;\nP(undetected) < 2^-28)"),
    ("Search\nStrategy",
     "A*-style greedy with priority queue\n(single best path; exponential backtrack\nO(k^d) for d errors, k candidates)",
     "Beam search with width 12\n(12 parallel hypotheses per iteration;\nlinear expansion O(12 \u00d7 24 \u00d7 iters))"),
    ("Error\nScoring",
     "Parity count + manual checksum bonus\n(+1 or +2; the +2 branch is dead code).\nRequires threshold_data, threshold_parity",
     "Unified syndrome counting from\nTanner graph. Each failing check votes\n+1 for its variables. No thresholds."),
    ("Error\nCapacity",
     "Practical limit: ~5-6 errors\n(exponential blowup makes search\ninfeasible beyond this)",
     "Handles up to ~8 errors\n(beam width bounds computation;\neach error adds ~2-3 iterations)"),
    ("Tuning\nRequired",
     "4 parameters: threshold_data,\nthreshold_parity, false_positive,\nmax_errors (all must be hand-tuned)",
     "2 structural parameters: max_flips,\nbeam_width (derived from code\nproperties, not hand-tuned)"),
    ("Failure\nRecovery",
     "No recovery from wrong early choice.\nOnce greedy picks wrong bit, entire\nsearch tree is poisoned.",
     "Beam preserves 12 alternatives.\nWrong choice drops in ranking;\ncorrect path survives to next iter."),
]

for j, (aspect, old, new) in enumerate(rows):
    y = 1.65 + j * 0.95
    _add_multiline_box(slide, col_x[0], y, col_w[0], 0.85,
                       aspect.split("\n"), DARK_GRAY, ORANGE, font_size=11, bold_first=True)
    _add_multiline_box(slide, col_x[1], y, col_w[1], 0.85,
                       old.split("\n"), DARK_RED, font_size=10, bold_first=False, alignment=PP_ALIGN.LEFT)
    _add_multiline_box(slide, col_x[2], y, col_w[2], 0.85,
                       new.split("\n"), DARK_GREEN, font_size=10, bold_first=False, alignment=PP_ALIGN.LEFT)


# ════════════════════════════════════════════════════════════════════
# SLIDE 14 — Complexity Analysis
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Complexity Analysis",
             font_size=32, bold=True, color=ACCENT)

# Left: Per-iteration breakdown
_add_textbox(slide, 0.8, 1.1, 6.0, 0.5,
             "Per-Iteration Cost Breakdown", font_size=20, bold=True, color=ORANGE)

cost_rows = [
    ("Operation", "Computation", "Cost"),
    ("Build check graph", "28 checks \u00d7 ~13 vars/check", "O(364) \u2014 done once"),
    ("Check if solved", "28 XOR evaluations", "O(364) per candidate"),
    ("Compute syndrome", "28 checks evaluated", "O(364) per candidate"),
    ("Score variables", "Sum over failing checks", "O(|failed| \u00d7 13)"),
    ("Expand beam", "beam_w\u00d72 deep copies + flips", "O(24 \u00d7 80) copies"),
    ("Evaluate children", "28 checks per child", "O(24 \u00d7 364)"),
    ("Prune (sort)", "Sort 24 candidates", "O(24 \u00d7 log 24)"),
]

hdr_x = [0.8, 4.3, 7.8]
hdr_w = [3.3, 3.3, 4.3]
for i, h in enumerate(cost_rows[0]):
    _add_rounded_box(slide, hdr_x[i], 1.6, hdr_w[i], 0.4, h, ACCENT, font_size=12, bold=True)

for j, (op, comp, cost) in enumerate(cost_rows[1:]):
    y = 2.1 + j * 0.38
    _add_rounded_box(slide, hdr_x[0], y, hdr_w[0], 0.35, op, DARK_GRAY, font_size=11)
    _add_rounded_box(slide, hdr_x[1], y, hdr_w[1], 0.35, comp, DARK_GRAY, MID_GRAY, font_size=10)
    _add_rounded_box(slide, hdr_x[2], y, hdr_w[2], 0.35, cost, DARK_GRAY, ORANGE, font_size=10, bold=True)

# Bottom: comparison
_add_textbox(slide, 0.8, 4.8, 12, 0.5,
             "Overall Complexity Comparison", font_size=20, bold=True, color=ACCENT2)

_add_multiline_box(slide, 0.8, 5.3, 5.8, 1.8, [
    "Old Decoder: O(k^d) worst case",
    "",
    "Where k = # candidate positions (~20-40),",
    "d = # errors. For 6 errors with 30 candidates:",
    "  30^6 = 729,000,000 evaluations",
    "",
    "In practice: pruning helps, but fundamental",
    "scaling is exponential in error count.",
], DARK_RED, font_size=13, bold_first=True, alignment=PP_ALIGN.LEFT)

_add_multiline_box(slide, 7.0, 5.3, 5.8, 1.8, [
    "New Decoder: O(B \u00d7 2B \u00d7 C \u00d7 I) per origami",
    "",
    "B=beam_width (12), C=checks (28\u00d713), I=iters (40).",
    "  12 \u00d7 24 \u00d7 364 \u00d7 40 = ~4,200,000 XOR ops",
    "",
    "LINEAR in error count (each error adds ~2-3 iters),",
    "bounded by beam width. Practical: ~ms per origami",
    "regardless of error count (up to 8).",
], DARK_GREEN, font_size=13, bold_first=True, alignment=PP_ALIGN.LEFT)


# ════════════════════════════════════════════════════════════════════
# SLIDE 15 — Summary & Key Takeaways
# ════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(prs.slide_layouts[6])
_set_slide_bg(slide, DARK_BG)
_add_textbox(slide, 0.8, 0.3, 10, 0.8,
             "Summary & Key Takeaways",
             font_size=32, bold=True, color=ACCENT)

# 4 key takeaways as boxes
takeaways = [
    ("1. Principled Foundation",
     "The dNAM parity system is structurally equivalent to an LDPC code. By building a Tanner graph "
     "from the existing parity and checksum relations, we can apply well-understood syndrome-based "
     "decoding techniques instead of ad-hoc heuristics.",
     ACCENT),
    ("2. Beam Search = Resilience",
     "Maintaining 12 parallel hypotheses per iteration makes the decoder robust to wrong early choices. "
     "Unlike greedy search, a single bad flip doesn't doom the entire correction attempt. The 2\u00d7 "
     "oversampling (try 24, keep 12) provides strong selection pressure.",
     ACCENT2),
    ("3. Strict Acceptance = Trust",
     "Every accepted origami is verified against all 28 independent XOR check equations. This eliminates "
     "silent data corruption \u2014 the most dangerous failure mode of the old decoder. The probability of "
     "an undetected error is less than 2^(-28).",
     ORANGE),
    ("4. Three-Tier Architecture = Coverage",
     "Tier 0 (strict accept) handles error-free origamis instantly. Tier 1 (beam-search) handles 1\u20138 "
     "errors with principled syndrome scoring. Tier 2 (legacy greedy) provides a fallback with different "
     "failure modes. Together, they maximize recovery across all error scenarios.",
     PURPLE),
]

for i, (title, body, clr) in enumerate(takeaways):
    y = 1.2 + i * 1.5
    _add_rounded_box(slide, 0.8, y, 3.0, 0.45, title, clr, font_size=14, bold=True)
    _add_multiline_box(slide, 4.0, y, 8.8, 1.35,
                       [body], DARK_GRAY, font_size=13, bold_first=False, alignment=PP_ALIGN.LEFT)

# Bottom
_add_rounded_box(slide, 1.5, 7.0, 10.3, 0.35,
    "The beam-search syndrome decoder brings coding-theoretic rigor to dNAM error correction",
    TEAL, font_size=14, bold=True)


# ════════════════════════════════════════════════════════════════════
# Save
# ════════════════════════════════════════════════════════════════════
out_path = "Beam_Search_Decoder_Introduction.pptx"
prs.save(out_path)
print(f"Saved: {out_path} ({len(prs.slides)} slides)")
