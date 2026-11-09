"""Runs the whole project in Python: log/difference -> VAR(3) -> LSTM (macro) -> LSTM (macro + Google Trends) -> tables and figures.
Usage:  pip install pandas numpy scipy statsmodels scikit-learn tensorflow matplotlib
        python run_all.py          (LSTM step: 9 variants x 10 seeds; roughly 10-30 minutes on a laptop CPU)"""
import subprocess, sys, os
here = os.path.dirname(os.path.abspath(__file__)); py = sys.executable
data, out, figs, prior = (os.path.join(here, d) for d in ("data", "results", "figures", "prior"))
for d in (out, figs): os.makedirs(d, exist_ok=True)
steps = [("1/5 VAR (log, difference, ADF, VAR(3), Granger, IRF, FEVD, forecasts)", ["code/var_paper.py", data, out]),
         ("2/5 LSTM with and without Google Trends (10 seeds)", ["code/lstm_paper.py", data, out]),
         ("3/5 Tables, tests and figures", ["code/tables_figs.py", data, out, figs, prior]),
         ("4/5 Export numbers", ["code/export_numbers.py", data, out]),
         ("5/5 Architecture diagram", ["code/fig_arch.py", figs])]
for name, args in steps:
    print("\n==", name, flush=True)
    subprocess.run([py, os.path.join(here, args[0])] + args[1:], check=True, cwd=here)
print("\nDone. See results/ (tables) and figures/ (300-dpi figures).")
