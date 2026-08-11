#!/usr/bin/env python3
"""
dNAM codeword grid (dots style, muted palette) with the CORRECT data taken from
the numbered grid. Filled dot = bit 1, empty = bit 0; colour = category;
orientation cells are squares. The binary digit is printed in every cell.
Edit CAT / BITS / colours and re-run.  ->  fig_codeword_grid.{png,svg,pdf}
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D

# ---- data from the numbered grid (rows top->bottom) -----------------------
CAT = [list("DDDDDDDDDD"),
       list("OPPPPPPPPO"),
       list("DPPPPPPPPD"),
       list("DPPDCCDPPD"),
       list("DPPDCCDPPD"),
       list("DPPPPPPPPD"),
       list("OPPPPPPPPO"),
       list("DDDDDDDIII")]
BITS = [list("0100010001"),
        list("1010110111"),
        list("1010101110"),
        list("0110100101"),
        list("0101011011"),
        list("0110100101"),
        list("1011010110"),
        list("0001100000")]

# ---- muted palette (sampled from the dots grid) ---------------------------
COL = {"D":"#00557F", "P":"#E76F51", "O":"#2A9D8F", "C":"#F2A658", "I":"#8E70B0"}
EMPTY_FACE="#FFFFFF"; EMPTY_EDGE="#AEB8C4"; GRIDLINE="#C9D2DC"
TXT_ON="#FFFFFF"; TXT_OFF="#5A6472"; NAVY="#21375E"; SUB="#5A6170"
FONT="DejaVu Sans"
mpl.rcParams.update({"font.family":FONT,"svg.fonttype":"none","pdf.fonttype":42})

rows, cols = 8, 10
fig, ax = plt.subplots(figsize=(7.6, 6.0))
ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect("equal"); ax.axis("off")

# grid lines
for x in range(cols+1): ax.plot([x,x],[0,rows],color=GRIDLINE,lw=1.0,zorder=1)
for y in range(rows+1): ax.plot([0,cols],[y,y],color=GRIDLINE,lw=1.0,zorder=1)
# outer frame
ax.add_patch(Rectangle((0,0),cols,rows,fill=False,edgecolor=NAVY,lw=2.4,zorder=3))

for r in range(rows):
    for c in range(cols):
        cat=CAT[r][c]; b=BITS[r][c]
        cx=c+0.5; cy=rows-(r+0.5)
        if cat=="O":                                   # orientation marker = square
            ax.add_patch(Rectangle((cx-0.30,cy-0.30),0.60,0.60,
                         facecolor=COL["O"],edgecolor="none",zorder=2))
            ax.text(cx,cy,b,ha="center",va="center",fontsize=12,
                    fontweight="bold",color=TXT_ON,zorder=4)
        elif b=="1":                                   # filled dot
            ax.add_patch(Circle((cx,cy),0.33,facecolor=COL[cat],edgecolor="none",zorder=2))
            ax.text(cx,cy,b,ha="center",va="center",fontsize=12,
                    fontweight="bold",color=TXT_ON,zorder=4)
        else:                                          # empty circle (bit 0)
            ax.add_patch(Circle((cx,cy),0.33,facecolor=EMPTY_FACE,
                         edgecolor=EMPTY_EDGE,lw=1.6,zorder=2))
            ax.text(cx,cy,b,ha="center",va="center",fontsize=12,
                    fontweight="bold",color=TXT_OFF,zorder=4)

# title
fig.text(0.5,0.965,"DNA-origami codeword",ha="center",fontsize=20,fontweight="bold",color=NAVY)
fig.text(0.5,0.918,"(8 × 10 = 80 cells)",ha="center",fontsize=14,color=SUB)

# legend
items=[("Data (29)","D","o"),("Parity (40)","P","o"),("Orient. (4)","O","s"),
       ("Checksum (4)","C","o"),("Index (3)","I","o")]
handles=[Line2D([0],[0],marker=m,color="none",markerfacecolor=COL[k],
         markeredgecolor="none",markersize=13,label=lab) for lab,k,m in items]
ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,-0.04),
          ncol=3,frameon=False,fontsize=12,handletextpad=0.4,columnspacing=1.6)

plt.subplots_adjust(top=0.88,bottom=0.16,left=0.04,right=0.96)
base=os.path.join(os.path.dirname(os.path.abspath(__file__)),"fig_codeword_grid")
for ext in ("png","svg","pdf"):
    fig.savefig(f"{base}.{ext}",dpi=240,bbox_inches="tight")
print("wrote",base+".{png,svg,pdf}")