"""Shared style for all report figures. Sans-serif (matches the TU Delft template's
Arial body font), ONE uniform font scheme used by every script, PDF (vector) output.

Consistent apparent text size across the report is achieved by using this single font
scheme everywhere AND choosing each figure's width proportional to how wide it is
embedded: single-panel figures use figwidth ~7 and are included at 0.7\\textwidth;
multi-panel figures use figwidth ~10 and are included at \\textwidth. Both then scale
by the same factor, so the text renders at the same size."""
import os
import matplotlib.pyplot as plt


def apply():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    plt.rcParams.update({
        # sans-serif to match the template's Arial; matplotlib's default DejaVu Sans
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        # ONE uniform font scheme for every figure
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # clean axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        # vector output
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,          # embed real (editable) fonts in the PDF
    })
