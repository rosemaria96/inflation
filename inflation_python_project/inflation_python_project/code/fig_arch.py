"""Labelled architecture diagram (monochrome, 300 dpi). Usage: python fig_arch.py <fig_dir>"""
import sys, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager
fam = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in {x.name for x in font_manager.fontManager.ttflist}][0]
plt.rcParams.update({"font.family": fam, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(6.4, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(-6, 100); ax.axis("off")
def box(x, y, w, h, txt, fs=7.5, hatch=None, bold=False, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="k", lw=1, hatch=hatch))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1, color="k"))
# stage labels
for x, t in [(11, "A. Data"), (37, "B. Pre-processing"), (65, "C. Models"), (90, "D. Output")]: ax.text(x, 97, t, ha="center", fontsize=8.5, fontweight="bold")
box(1, 60, 21, 24, "Macro series\nCPI, Brent,\nUSD/INR, IIP\n(162 months)", fc="0.92")
box(1, 14, 21, 26, "Google Trends\n“inflation”\n“fuel price”\n“food price”\n“petrol price”", fc="0.92")
box(27, 68, 21, 18, "Log transform,\nfirst difference,\nADF test")
box(27, 30, 21, 26, "Min–max scaling\n(training rows)\n12-month windows\n120 train / 30 test")
box(54, 72, 22, 14, "VAR(3)\n+ FEVD", bold=True)
box(54, 44, 22, 18, "LSTM\n4 macro inputs", bold=True)
box(54, 14, 22, 20, "LSTM + Google Trends\n8 inputs\n(ablation)", bold=True, fc="0.85")
box(81, 50, 18, 34, "One-step-ahead\nCPI forecasts\n(30 test months)\n\nbenchmarks:\nnaive, drift,\nAR(1)", fc="0.92", fs=7)
box(81, 8, 18, 34, "Evaluation\n\nRMSE, MAE,\nMAPE\n10 seeds:\nmean ± SD\nDM, Welch,\nMann–Whitney", fs=7)
arrow(22, 74, 27, 77); arrow(22, 68, 27, 50); arrow(22, 28, 27, 36)
arrow(48, 77, 54, 79); arrow(48, 46, 54, 53); arrow(48, 38, 54, 26)
arrow(76, 79, 81, 76); arrow(76, 53, 81, 62); arrow(76, 24, 81, 52)
arrow(90, 50, 90, 42)
ax.text(50, -3, "Darker box = model with Google Trends added (ablation). Light grey = data and forecasts.", ha="center", fontsize=7)
plt.savefig(f"{sys.argv[1]}/fig_arch.png", bbox_inches="tight", pad_inches=.05); print("ok")
