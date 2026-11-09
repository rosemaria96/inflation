import sys, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
warnings.filterwarnings("ignore")
data_dir, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{data_dir}/final_merged_dataset.csv", parse_dates=["Date"], dayfirst=True).set_index("Date")
macro = ["CPI", "CRUDE_BRENT", "USD_INR", "IIP"]; gtc = ["fuel price", "food price", "inflation", "petrol price"]
d = np.log(df[macro]).diff().dropna(); V = json.load(open(f"{out}/var_results.json"))
J = {}
J["desc"] = [[c, *[float(df[c].agg(f)) for f in ["mean", "std", "min", "max"]]] for c in macro + gtc]
J["adf"] = pd.read_csv(f"{out}/t_adf.csv").to_dict("records")
ic = pd.read_csv(f"{out}/t_lagsel.csv"); J["lagsel"] = []
for q in range(1, 6):
    r = ic[ic.lag == q].iloc[0]; J["lagsel"].append(dict(lag=q, aic=float(r.aic), bic=float(r.bic), hqic=float(r.hqic), w12=V["whiteness12_by_lag"][str(q)], lr=V["lr_tests"].get(f"{q}vs{q+1}")))
gcl = {}
for q in [2, 3, 4]:
    g = VAR(d).fit(q); gcl[q] = {}
    for nm, ca in [("CRUDE_BRENT", ["CRUDE_BRENT"]), ("USD_INR", ["USD_INR"]), ("IIP", ["IIP"]), ("joint", ["CRUDE_BRENT", "USD_INR", "IIP"])]:
        t = g.test_causality("CPI", ca, kind="f"); gcl[q][nm] = dict(F=float(t.test_statistic), p=float(t.pvalue))
J["granger"] = gcl
J["fevd"] = pd.read_csv(f"{out}/t_fevd12.csv", index_col=0).to_dict("index")
J["acc"] = pd.read_csv(f"{out}/t_accuracy.csv").to_dict("records")
gt = pd.read_csv(f"{out}/t_gt_tests.csv"); runs = pd.read_csv(f"{out}/lstm_paper_runs.csv"); P = np.load(f"{out}/lstm_paper_preds.npz"); act = P["act"]
def dm(e1, e2):
    dd = e1**2 - e2**2; n = len(dd); s = dd.mean() / np.sqrt(dd.var(ddof=0) / n) * np.sqrt((n - 1) / n); return float(s), float(2 * (1 - stats.t.cdf(abs(s), n - 1)))
base = P["D-Base"].mean(0); gts = []
for _, r in gt.iterrows():
    s, p = dm(act - P[r.model.replace(" ", "_")].mean(0), act - base); gts.append(dict(model=r.model, pct=float(r.pct), mwu_p=float(r.mwu_p), holm=float(r.holm_p), welch=float(r.welch_p), dm=s, dm_p=p))
J["gt"] = gts; J["level_gt"] = json.load(open(f"{out}/level_gt_test.json")); J["dm"] = pd.read_csv(f"{out}/t_dm.csv").to_dict("records")
J["orig"] = pd.read_csv(f"{out}/t_original_protocol.csv").to_dict("records"); J["orig_test"] = json.load(open(f"{out}/orig_protocol_test.json"))
J["var"] = V; J["irf"] = json.load(open(f"{out}/irf_cpi_last.json"))
J["var_lag_sens"] = pd.read_csv(f"{out}/t_lag_sensitivity.csv").to_dict("records")
# GT outlier note
J["gt_last"] = df[gtc].iloc[-3:].values.tolist()
# fraction of seeds where D-GT(all) rmse < D-Base min
J["seed_overlap"] = float((runs[runs.model == "D-GT(all)"].RMSE.values < runs[runs.model == "D-Base"].RMSE.values.min()).mean())
# original-protocol range overlap: min GT seed vs max baseline seed
import math
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o): return None
    return o
json.dump(clean(J), open(f"{out}/paper_data.json", "w"), indent=1, default=float)
print(json.dumps({k: J[k] for k in ["granger", "gt", "lagsel", "var_lag_sens"]}, indent=1, default=float)[:5000])
