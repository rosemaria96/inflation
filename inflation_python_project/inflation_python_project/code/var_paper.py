"""VAR analysis for the paper. Usage: python var_paper.py <data_dir> <out_dir>"""
import sys, json, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
from scipy import stats
warnings.filterwarnings("ignore")
data_dir, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{data_dir}/datasetwithoutgoogle.csv", parse_dates=["Date"], dayfirst=True).set_index("Date"); df.index.freq = "MS"
cols = ["CPI", "CRUDE_BRENT", "USD_INR", "IIP"]
lg = np.log(df[cols]); d = lg.diff().dropna(); N = len(df)
R = {}

# Table 1 descriptive stats (levels)
desc = df[cols].describe().T[["count", "mean", "std", "min", "max"]]; desc.to_csv(f"{out}/t_desc.csv")

# Table 2 ADF (log levels, constant & constant+trend; first difference, constant). KPSS on differences.
rows = []
for c in cols:
    a1 = adfuller(lg[c], regression="c", autolag="AIC"); a2 = adfuller(lg[c], regression="ct", autolag="AIC")
    a3 = adfuller(d[c], regression="c", autolag="AIC"); k = kpss(d[c], regression="c", nlags="auto")
    rows.append(dict(variable=c, lvl_c_p=a1[1], lvl_ct_p=a2[1], diff_c_stat=a3[0], diff_c_p=a3[1], kpss_diff_p=k[1]))
pd.DataFrame(rows).to_csv(f"{out}/t_adf.csv", index=False)

# Lag selection (full sample dlog)
sel = VAR(d).select_order(maxlags=8)
ic = pd.DataFrame({k: sel.ics[k] for k in ["aic", "bic", "hqic", "fpe"]}); ic.index.name = "lag"; ic.to_csv(f"{out}/t_lagsel.csv")
R["lag_selected"] = {k: int(v) for k, v in sel.selected_orders.items()}
# Lag rule: smallest order whose residuals pass a Portmanteau whiteness test (12 lags, 5%); AIC/BIC orders reported for reference
wp = {q: float(VAR(d).fit(q).test_whiteness(nlags=12, adjusted=True).pvalue) for q in range(1, 7)}
R["whiteness12_by_lag"] = wp
p = min(q for q in wp if wp[q] > 0.05); R["p_main"] = p
# sequential LR tests
ll = {q: VAR(d.iloc[6 - q:]).fit(q).llf for q in range(1, 7)}  # same effective sample (drop first 6-q rows)
R["lr_tests"] = {f"{q}vs{q+1}": float(1 - stats.chi2.cdf(2 * (ll[q + 1] - ll[q]), 16)) for q in range(1, 6)}
print("selected", R["lag_selected"])

def fevd_cpi(fit, order, h=12):
    """CPI row of FEVD at horizon h (Cholesky with given ordering)."""
    sub = d[order]; f = VAR(sub).fit(p) if fit is None else fit
    dec = f.fevd(h).decomp[order.index("CPI")][h - 1]
    return pd.Series(dec * 100, index=order)

def gfevd_cpi(fit, h=12):
    """Generalised FEVD (Pesaran-Shin 1998), CPI row, normalised."""
    Th = fit.ma_rep(h - 1); S = fit.sigma_u.values; k = S.shape[0]
    num = np.zeros(k); den = 0.0
    for l in range(h):
        e = Th[l] @ S                      # k x k, row i col j = (Theta_l Sigma)_{ij}
        num += (e[0, :] ** 2) / S.diagonal()
        den += (Th[l] @ S @ Th[l].T)[0, 0]
    th = num / den
    return pd.Series(th / th.sum() * 100, index=cols)

full = VAR(d).fit(p)
R["stable"] = bool(np.all(np.abs(full.roots) > 1)); R["max_root_mod"] = float(np.max(1 / np.abs(full.roots)))
R["nobs"] = int(full.nobs)
# Diagnostics
w = full.test_whiteness(nlags=12, adjusted=True); R["portmanteau12_p"] = float(w.pvalue)
w2 = full.test_whiteness(nlags=24, adjusted=True); R["portmanteau24_p"] = float(w2.pvalue)
nm = full.test_normality(); R["jb_p"] = float(nm.pvalue)
# Granger causality to CPI
gc = []
for x in ["CRUDE_BRENT", "USD_INR", "IIP"]:
    t = full.test_causality("CPI", [x], kind="f"); gc.append(dict(cause=x, F=t.test_statistic, p=t.pvalue))
t = full.test_causality("CPI", ["CRUDE_BRENT", "USD_INR", "IIP"], kind="f"); gc.append(dict(cause="All three jointly", F=t.test_statistic, p=t.pvalue))
pd.DataFrame(gc).to_csv(f"{out}/t_granger.csv", index=False)

# FEVD at h=12 under identifications
order_last = ["CRUDE_BRENT", "USD_INR", "IIP", "CPI"]; order_first = ["CPI", "CRUDE_BRENT", "USD_INR", "IIP"]
res = pd.DataFrame({
    "Cholesky: CPI last": fevd_cpi(VAR(d[order_last]).fit(p), order_last)[cols],
    "Cholesky: CPI first": fevd_cpi(VAR(d[order_first]).fit(p), order_first)[cols],
    "Generalised (order-free)": gfevd_cpi(full)}).T
# Robustness: seasonal (month) dummies + COVID dummy as exogenous
ex = pd.get_dummies(d.index.month, prefix="m", drop_first=True).astype(float); ex.index = d.index
ex["covid"] = ((d.index >= "2020-03-01") & (d.index <= "2020-06-01")).astype(float)
fx = VAR(d[order_last], exog=ex).fit(p)
dec = fx.fevd(12).decomp[3][11] * 100; res.loc["Cholesky: CPI last + month & COVID dummies"] = pd.Series(dec, index=order_last)[cols]
res.round(2).to_csv(f"{out}/t_fevd12.csv")
print(res.round(2))

# FEVD path (CPI last) for figure + sensitivity to lag length
fv = VAR(d[order_last]).fit(p).fevd(12); path = pd.DataFrame(fv.decomp[3] * 100, columns=order_last, index=range(1, 13)); path.to_csv(f"{out}/fevd_path_cpi_last.csv")
sens = []
for pp in [1, 2, 3, 4]:
    f = VAR(d[order_last]).fit(pp); dd = f.fevd(12).decomp[3][11] * 100
    g = VAR(d).fit(pp)
    sens.append(dict(lag=pp, CPI_own=dd[3], BRENT=dd[0], USD=dd[1], IIP=dd[2],
                     gc_brent_p=g.test_causality("CPI", ["CRUDE_BRENT"], kind="f").pvalue, gc_usd_p=g.test_causality("CPI", ["USD_INR"], kind="f").pvalue,
                     gc_iip_p=g.test_causality("CPI", ["IIP"], kind="f").pvalue, port12_p=g.test_whiteness(nlags=12, adjusted=True).pvalue if 12 > pp else np.nan))
pd.DataFrame(sens).round(4).to_csv(f"{out}/t_lag_sensitivity.csv", index=False); print(pd.DataFrame(sens).round(3))

# IRFs (Cholesky, CPI last) with MC 95% bands, responses of CPI
fl = VAR(d[order_last]).fit(p); irf = fl.irf(12)
lo, hi = irf.errband_mc(orth=True, repl=2000, seed=1, signif=0.05)
irfres = {}
for j, nm_ in enumerate(order_last[:3]):
    irfres[nm_] = dict(irf=irf.orth_irfs[:, 3, j].tolist(), lo=lo[:, 3, j].tolist(), hi=hi[:, 3, j].tolist())
json.dump(irfres, open(f"{out}/irf_cpi_last.json", "w"))
sig = {k: [int(h) for h in range(13) if (v["lo"][h] > 0 or v["hi"][h] < 0)] for k, v in irfres.items()}; R["irf_sig_horizons"] = sig; print("IRF significant horizons:", sig)
R["irf_peak"] = {k: int(np.argmax(np.abs(v["irf"]))) for k, v in irfres.items()}

# ---- Forecasting: same test window as the LSTM (rows 132..161 = Jul 2023-Dec 2025), one-step-ahead, coefficients from training sample only
first_test = 12 + int(0.8 * (N - 12)); train_d = d.loc[: df.index[first_test - 1]]
p_tr = p; R["p_train"] = int(p_tr)
tr = VAR(train_d).fit(p_tr)
ar_tr = {q: AutoReg(train_d["CPI"], lags=q, trend="c").fit() for q in range(1, 7)}
q_best = min(ar_tr, key=lambda q: ar_tr[q].aic); R["ar_lag"] = int(q_best); ar = ar_tr[q_best]
mu = train_d["CPI"].mean()
fc = {"Naive": [], "Drift": [], "AR": [], "VAR": []}
for t in range(first_test, N):
    prev = df["CPI"].iloc[t - 1]
    fc["Naive"].append(prev); fc["Drift"].append(prev * np.exp(mu))
    hist = d["CPI"].loc[: df.index[t - 1]].values[-q_best:][::-1]
    dh = ar.params.iloc[0] + float(np.dot(ar.params.iloc[1:].values, hist)); fc["AR"].append(prev * np.exp(dh))
    h2 = d.loc[: df.index[t - 1]].values[-p_tr:]; fc["VAR"].append(prev * np.exp(tr.forecast(h2, steps=1)[0][0]))
act = df["CPI"].iloc[first_test:].values; prevs = df["CPI"].iloc[first_test - 1:N - 1].values
out_df = pd.DataFrame({"Date": df.index[first_test:], "CPI_actual": act, **fc}); out_df.to_csv(f"{out}/stat_forecasts.csv", index=False)
def m(y, yh): e = y - np.array(yh); return dict(RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))), MAPE=float(np.mean(np.abs(e / y)) * 100))
vs = {}
for q in [1, 2, 3, 4]:
    trq = VAR(train_d).fit(q); pr = []
    for t in range(first_test, N):
        pr.append(df["CPI"].iloc[t - 1] * np.exp(trq.forecast(d.loc[: df.index[t - 1]].values[-q:], steps=1)[0][0]))
    vs[q] = m(act, pr)
R["var_lag_forecast_sensitivity"] = vs
R["metrics"] = {k: m(act, v) for k, v in fc.items()}; R["n_test"] = len(act); R["test_start"] = str(df.index[first_test].date()); R["train_end"] = str(df.index[first_test - 1].date())
json.dump(R, open(f"{out}/var_results.json", "w"), indent=1, default=str); print(json.dumps(R, indent=1, default=str))
