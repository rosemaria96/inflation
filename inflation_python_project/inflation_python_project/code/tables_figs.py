"""Tables + 300-dpi monochrome-safe figures. Usage: python tables_figs.py <data_dir> <out_dir> <fig_dir> <prior_out_dir>"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
warnings.filterwarnings("ignore")
data_dir, out, figs, prior = sys.argv[1:5]
fam = [f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in {x.name for x in font_manager.fontManager.ttflist}][0]
plt.rcParams.update({"font.family": fam, "font.size": 9, "axes.linewidth": .7, "savefig.dpi": 300, "figure.dpi": 100,
                     "axes.spines.top": False, "axes.spines.right": False})
df = pd.read_csv(f"{data_dir}/final_merged_dataset.csv", parse_dates=["Date"], dayfirst=True)
V = json.load(open(f"{out}/var_results.json")); runs = pd.read_csv(f"{out}/lstm_paper_runs.csv"); P = np.load(f"{out}/lstm_paper_preds.npz")
sf = pd.read_csv(f"{out}/stat_forecasts.csv", parse_dates=["Date"]); act = sf.CPI_actual.values; dates = sf.Date
prev = df["CPI"].iloc[len(df) - len(act) - 1:len(df) - 1].values
def met(y, p): e = y - p; return np.sqrt(np.mean(e**2)), np.mean(np.abs(e)), np.mean(np.abs(e / y)) * 100

# ---- Forecast accuracy table
rows = []
naive_rmse = V["metrics"]["Naive"]["RMSE"]
for k, lab in [("Naive", "Naive (random walk)"), ("Drift", "Random walk with drift"), ("AR", "AR(1) on log-change"), ("VAR", f"VAR({V['p_main']})")]:
    m = V["metrics"][k]; rows.append([lab, m["RMSE"], m["MAE"], m["MAPE"], m["RMSE"] / naive_rmse, np.nan])
lab = {"L-Base": "LSTM, level target, 4 macro variables", "L-GT(all)": "LSTM, level target, + 4 Google Trends terms",
       "D-Base": "LSTM, change target, 4 macro variables", "D-GT(all)": "LSTM, change target, + 4 Google Trends terms (changes)",
       "D-GT(all, levels)": "LSTM, change target, + 4 Google Trends terms (levels)", "D-GT(fuel price)": "LSTM, change target, + “fuel price” only",
       "D-GT(food price)": "LSTM, change target, + “food price” only", "D-GT(inflation)": "LSTM, change target, + “inflation” only", "D-GT(petrol price)": "LSTM, change target, + “petrol price” only"}
for k in lab:
    r = runs[runs.model == k]; rows.append([lab[k], r.RMSE.mean(), r.MAE.mean(), r.MAPE.mean(), r.RMSE.mean() / naive_rmse, r.RMSE.std()])
acc = pd.DataFrame(rows, columns=["Model", "RMSE", "MAE", "MAPE", "RMSE_ratio_to_naive", "RMSE_sd_seeds"]); acc.to_csv(f"{out}/t_accuracy.csv", index=False)
print(acc.round(3).to_string(index=False))

# ---- seed-level tests vs matching baseline, Holm within family
def holm(p):
    p = np.array(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, idx in enumerate(o): run = max(run, (len(p) - i) * p[idx]); adj[idx] = min(1, run)
    return adj
fam_rows = []
for k in [x for x in lab if x.startswith("D-GT")]:
    a, b = runs[runs.model == k].RMSE.values, runs[runs.model == "D-Base"].RMSE.values
    fam_rows.append(dict(model=k, mean=a.mean(), base=b.mean(), pct=100 * (a.mean() - b.mean()) / b.mean(), welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue, mwu_p=stats.mannwhitneyu(a, b, alternative="two-sided").pvalue))
ft = pd.DataFrame(fam_rows); ft["holm_p"] = holm(ft.mwu_p.values); ft.to_csv(f"{out}/t_gt_tests.csv", index=False); print(ft.round(4).to_string(index=False))
a, b = runs[runs.model == "L-GT(all)"].RMSE.values, runs[runs.model == "L-Base"].RMSE.values
lv = dict(pct=100 * (a.mean() - b.mean()) / b.mean(), welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue, mwu_p=stats.mannwhitneyu(a, b).pvalue); json.dump(lv, open(f"{out}/level_gt_test.json", "w")); print(lv)

# ---- DM tests on seed-averaged forecasts
def dm(e1, e2):
    d = e1**2 - e2**2; n = len(d); s = d.mean() / np.sqrt(d.var(ddof=0) / n) * np.sqrt((n - 1) / n); return s, 2 * (1 - stats.t.cdf(abs(s), n - 1))
ens = {k: P[k.replace(" ", "_")].mean(0) for k in lab}; ens["Naive"] = sf.Naive.values; ens["VAR"] = sf.VAR.values; ens["AR"] = sf.AR.values; ens["Drift"] = sf.Drift.values
pairs = [("D-Base", "Naive"), ("D-Base", "VAR"), ("D-Base", "AR"), ("VAR", "Naive"), ("AR", "Naive"), ("D-GT(all)", "D-Base"), ("D-GT(fuel price)", "D-Base"), ("L-GT(all)", "L-Base"), ("L-Base", "Naive"), ("L-Base", "VAR"), ("L-Base", "AR"), ("L-Base", "Drift")]
dmr = []
for a_, b_ in pairs:
    s, p = dm(act - ens[a_], act - ens[b_]); dmr.append(dict(model_A=a_, model_B=b_, DM=s, p=p))
pd.DataFrame(dmr).to_csv(f"{out}/t_dm.csv", index=False); print(pd.DataFrame(dmr).round(3).to_string(index=False))

# ---- original-protocol reconciliation (scaled units) from earlier runs
old = pd.read_csv(f"{prior}/lstm_runs.csv"); o = old.groupby("model").A_RMSE.agg(["mean", "std", "min", "max"]); o.to_csv(f"{out}/t_original_protocol.csv"); print(o.round(4))
ob, og = old[old.model == "Baseline LSTM"].A_RMSE, old[old.model == "GT-LSTM"].A_RMSE
json.dump(dict(welch_p=float(stats.ttest_ind(og, ob, equal_var=False).pvalue), mwu_p=float(stats.mannwhitneyu(og, ob).pvalue)), open(f"{out}/orig_protocol_test.json", "w"))

# ================= FIGURES (monochrome-safe: line styles, hatches, direct labels) =================
# Fig 1: data
fig, axs = plt.subplots(2, 2, figsize=(6.2, 4.2), sharex=True)
for ax, (c, t) in zip(axs.ravel(), [("CPI", "CPI (index)"), ("CRUDE_BRENT", "Brent crude (US$/barrel)"), ("USD_INR", "USD/INR"), ("IIP", "IIP (index)")]):
    ax.plot(df.Date, df[c], color="k", lw=1); ax.axvline(pd.Timestamp("2023-07-01"), color="k", ls=":", lw=.8); ax.set_title(t, fontsize=9, loc="left"); ax.grid(alpha=.25)
axs[1, 0].text(pd.Timestamp("2023-08-01"), axs[1, 0].get_ylim()[0] + 0.03 * (axs[1, 0].get_ylim()[1] - axs[1, 0].get_ylim()[0]), "test period →", fontsize=7)
plt.tight_layout(); plt.savefig(f"{figs}/fig1_data.png"); plt.close()
# Fig 2: IRFs
I = json.load(open(f"{out}/irf_cpi_last.json")); fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=False)
for ax, (k, t) in zip(axs, [("CRUDE_BRENT", "Brent → CPI"), ("USD_INR", "USD/INR → CPI"), ("IIP", "IIP → CPI")]):
    h = np.arange(13); ax.fill_between(h, I[k]["lo"], I[k]["hi"], color="0.8", label="95% band"); ax.plot(h, I[k]["irf"], color="k", lw=1.3, marker="o", ms=2.5); ax.axhline(0, color="k", lw=.6, ls="--")
    ax.set_title(t, fontsize=9); ax.set_xlabel("Months after shock"); ax.grid(alpha=.25)
axs[0].set_ylabel("Response of Δlog CPI")
plt.tight_layout(); plt.savefig(f"{figs}/fig2_irf.png"); plt.close()
# Fig 3: FEVD stacked, hatched
F = pd.read_csv(f"{out}/fevd_path_cpi_last.csv", index_col=0); fig, ax = plt.subplots(figsize=(5.6, 3))
cols_ = ["CPI", "IIP", "USD_INR", "CRUDE_BRENT"]; hat = ["", "///", "xxx", "..."]; sh = ["0.85", "0.65", "0.45", "0.25"]
ax.stackplot(F.index, *[F[c] for c in cols_], labels=["CPI (own)", "IIP", "USD/INR", "Brent"], colors=sh, edgecolor="k", linewidth=.5, hatch=None)
for poly, h_ in zip(ax.collections, hat): poly.set_hatch(h_)
ax.set_xlim(1, 12); ax.set_ylim(0, 100); ax.set_xlabel("Forecast horizon (months)"); ax.set_ylabel("% of forecast-error variance"); ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{figs}/fig3_fevd.png"); plt.close()
# Fig 4: forecasts
import matplotlib.dates as mdates
fig, ax = plt.subplots(figsize=(6.3, 3.5)); ax.plot(dates, act, color="k", lw=2.4, label="Actual CPI")
sty = [(sf.VAR.values, "--", None, "VAR(3), one-step"), (ens["L-Base"], "-", "o", "LSTM, raw levels (10-seed mean)"),
       (ens["L-GT(all)"], "-.", "s", "GT-LSTM, raw levels (10-seed mean)")]
for y, ls, mk, nm in sty: ax.plot(dates, y, color="0.3", ls=ls, lw=1.3, marker=mk, ms=3, markevery=3, label=nm)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); plt.setp(ax.get_xticklabels(), fontsize=8)
ax.set_ylabel("CPI index"); ax.grid(alpha=.25); ax.legend(loc="upper left", frameon=False, fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{figs}/fig4_forecasts.png"); plt.close()
# Fig 5: seed dispersion
order = ["L-Base", "L-GT(all)"]
names = ["LSTM\n(macro only)", "LSTM\n(macro + Google Trends)"]
fig, ax = plt.subplots(figsize=(6.3, 3.3)); data = [runs[runs.model == k].RMSE.values for k in order]
bp = ax.boxplot(data, tick_labels=names, widths=.4, patch_artist=True, medianprops=dict(color="k"))
for i, b in enumerate(bp["boxes"]): b.set_facecolor("0.9" if "Base" in order[i] else "0.7"); b.set_edgecolor("k"); b.set_hatch("" if "Base" in order[i] else "//")
rng = np.random.default_rng(0)
for i, dd in enumerate(data, 1): ax.scatter(rng.normal(i, .05, len(dd)), dd, s=8, color="k", zorder=3)
ax.axhline(V["metrics"]["Naive"]["RMSE"], color="k", ls=":", lw=1); ax.text(0.45, V["metrics"]["Naive"]["RMSE"] + .1, "Naive", fontsize=7.5, ha="left")
vv = V["metrics"]["VAR"]["RMSE"]; ax.axhline(vv, color="k", ls="--", lw=1); ax.text(0.45, vv - .3, "VAR(3)", fontsize=7.5, ha="left")
ax.set_ylabel("Test RMSE (CPI points)"); ax.tick_params(axis="x", labelsize=7.5); ax.grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(f"{figs}/fig5_seeds.png"); plt.close()
print("figures done")
