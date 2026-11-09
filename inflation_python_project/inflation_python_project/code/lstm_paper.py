"""LSTM experiments for the paper. Targets: (L) CPI index level, min-max scaled on training rows only (original design, leak-free);
(D) monthly log-change of CPI, standardised on training rows only. 10 seeds. Usage: python lstm_paper.py <data_dir> <out_dir>"""
import sys, os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
warnings.filterwarnings("ignore")
data_dir, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f"{data_dir}/final_merged_dataset.csv", parse_dates=["Date"], dayfirst=True)
macro = ["CPI", "CRUDE_BRENT", "USD_INR", "IIP"]; gt = ["fuel price", "food price", "inflation", "petrol price"]
LOOK = 12; N = len(df); first_test = LOOK + int(0.8 * (N - LOOK)); SEEDS = range(10)
act = df["CPI"].iloc[first_test:].values; prev = df["CPI"].iloc[first_test - 1:N - 1].values

def fit_pred(X, y, s, seed):
    tf.keras.utils.set_random_seed(seed)
    m = Sequential([Input((LOOK, X.shape[2])), LSTM(50), Dense(1)]); m.compile("adam", "mse")
    m.fit(X[:s], y[:s], epochs=50, batch_size=16, verbose=0)
    return m.predict(X[s:], verbose=0).ravel()

def level_model(cols, seed):
    sc = MinMaxScaler().fit(df[cols].iloc[:first_test]); a = sc.transform(df[cols])
    X = np.array([a[i - LOOK:i] for i in range(LOOK, N)]); y = a[LOOK:, 0]
    return fit_pred(X, y, first_test - LOOK, seed) * (sc.data_max_[0] - sc.data_min_[0]) + sc.data_min_[0]

def change_model(gtcols, seed, gt_mode="diff"):
    f = pd.DataFrame({c: np.log(df[c]).diff() for c in macro})
    for c in gtcols: f[c] = df[c].diff() if gt_mode == "diff" else df[c].astype(float)
    arr = f.values; sc = StandardScaler().fit(arr[1:first_test]); a = np.nan_to_num(sc.transform(arr))
    X = np.array([a[i - LOOK:i] for i in range(LOOK, N)]); y = a[LOOK:, 0]
    dh = fit_pred(X, y, first_test - LOOK, seed) * sc.scale_[0] + sc.mean_[0]
    return prev * np.exp(dh)

variants = {"L-Base": lambda s: level_model(macro, s), "L-GT(all)": lambda s: level_model(macro + gt, s),
            "D-Base": lambda s: change_model([], s), "D-GT(all)": lambda s: change_model(gt, s),
            "D-GT(all, levels)": lambda s: change_model(gt, s, "level")}
for g in gt: variants[f"D-GT({g})"] = (lambda g: (lambda s: change_model([g], s)))(g)
rows, preds = [], {}
for name, fn in variants.items():
    P = []
    for s in SEEDS:
        p = fn(s); P.append(p); e = act - p
        rows.append(dict(model=name, seed=s, RMSE=np.sqrt(np.mean(e**2)), MAE=np.mean(np.abs(e)), MAPE=np.mean(np.abs(e / act)) * 100))
    preds[name] = np.array(P); print(name, round(np.mean([r["RMSE"] for r in rows if r["model"] == name]), 3), flush=True)
pd.DataFrame(rows).to_csv(f"{out}/lstm_paper_runs.csv", index=False)
np.savez(f"{out}/lstm_paper_preds.npz", act=act, **{k.replace(" ", "_"): v for k, v in preds.items()})
