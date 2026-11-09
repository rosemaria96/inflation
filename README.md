# Inflation Forecasting in India: VAR, LSTM and Google Trends-Enhanced LSTM

Code and results for the paper *"Inflation Forecasting in India: A Comparative Study of VAR, LSTM and Google Trends-Enhanced LSTM"* (Rose Maria Jose, Rajagiri College of Social Sciences, Kochi).

The project asks whether Google Trends search indices improve one-step-ahead forecasts of India's monthly consumer price index (CPI). It compares:

1. **VAR(3)** on log-differenced CPI, Brent crude, USD/INR and IIP (with ADF tests and forecast-error variance decomposition);
2. **LSTM** on the four macroeconomic variables (raw levels);
3. **LSTM + Google Trends** on the same four variables plus four search indices ("inflation", "fuel price", "food price", "petrol price").

Benchmarks: naive (last month), random walk with drift, AR(1).

## Main findings (30-month test window, July 2023 – December 2025)

| Model | RMSE (CPI points) |
|---|---|
| Naive | 1.557 |
| Random walk with drift | 1.484 |
| AR(1) | 1.408 |
| VAR(3) | 1.339 |
| LSTM, macro variables (mean of 10 seeds, SD 1.152) | 3.062 |
| LSTM + Google Trends (mean of 10 seeds, SD 1.742) | 4.934 |

- Google Trends **did not improve** forecasts: RMSE rose by 61% (Mann–Whitney p = 0.014; Diebold–Mariano p = 0.005).
- CPI inflation is highly persistent: its own shocks explain 83–91% of 12-month forecast-error variance in the VAR.
- A single-run "improvement" from Google Trends is consistent with random initialisation (see `results/t_original_protocol.csv`).

## Repository structure

```
run_all.py            one command that runs everything
code/
  var_paper.py        log/difference, ADF, lag selection, VAR(3), FEVD, benchmarks and VAR forecasts
  lstm_paper.py       LSTM models (10 seeds): macro-only, macro + Google Trends (plus a supplementary log-difference variant)
  tables_figs.py      accuracy tables, Diebold-Mariano / Welch / Mann-Whitney tests, figures
  export_numbers.py   collects results into results/paper_data.json
  fig_arch.py         architecture diagram
data/
  datasetwithoutgoogle.csv       CPI, Brent, USD/INR, IIP (162 months)
  final_merged_dataset.csv       the same plus four Google Trends indices
prior/lstm_runs.csv   results of the original (full-sample scaling) protocol, used for the seed-variability comparison
results/              output tables (CSV/JSON) and saved forecasts
figures/              300-dpi figures
```

## How to run

Requirements: Python 3.11 or 3.12 (3.13 also works if TensorFlow supports it), about 2 GB of free disk space, no GPU needed.

```bash
git clone <this repository URL>
cd <repository folder>
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` takes roughly 10–30 minutes on a laptop CPU (most of the time is the LSTM step: 9 variants × 10 seeds). It writes tables to `results/` and figures to `figures/`.

To run a single step: `python code/var_paper.py data results`, `python code/lstm_paper.py data results`, and so on (each script prints its usage line).

## Experimental setup

- **Data:** 162 monthly observations, July 2012 – December 2025; no missing values.
- **Windows:** 12-month input windows give 150 samples; the first 120 (targets July 2013 – June 2023) train the models and the last 30 (targets July 2023 – December 2025) test them. The split is chronological; no validation set was used and no hyperparameters were tuned.
- **Leak-free scaling:** min–max scalers are fitted on training rows only.
- **LSTM:** one LSTM layer (50 units) + Dense(1), Adam, MSE loss, 50 epochs, batch size 16.
- **Seeds:** 0–9 (`tf.keras.utils.set_random_seed`). Results can differ slightly on other hardware.
- **VAR:** lag order = smallest p whose residuals pass a Portmanteau test at 5% (12 lags) → VAR(3). Coefficients for forecasting are estimated once on the training sample.
- **Metrics:** RMSE, MAE, MAPE in CPI points; Diebold–Mariano test with the Harvey–Leybourne–Newbold correction on seed-averaged forecasts; Welch and Mann–Whitney tests on per-seed RMSE.

### Tested environment

2 vCPUs (Intel Xeon 2.80 GHz), 7 GB RAM, no GPU, Linux; Python 3.13.17, TensorFlow/Keras 2.21.0, statsmodels 0.15.0, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, Matplotlib 3.11.2.

## Data sources and terms of use

> - CPI and IIP: MoSPI, Government of India (https://mospi.gov.in).
> - Brent crude: World Bank Commodity Price Data ("Pink Sheet"). USD/INR source: RBI exchange data.
> - Google Trends: https://trends.google.com. Indices are relative (0–100) and aggregated; they contain no personal data. Use is subject to Google's terms of service.
>
> Check each source's licence and terms before redistributing the CSV files in a public repository. If redistribution is not allowed, remove `data/` and describe how to download the series instead.

## Reproducing the paper's numbers

Running `python run_all.py` regenerates every table and figure in the paper. Small differences in the last decimal places are possible on other hardware. The 10-seed means and tests are in `results/t_accuracy.csv`, `results/t_dm.csv` and `results/t_gt_tests.csv`.

## Limitations

A single chronological split with 30 test months gives low statistical power; only one-step-ahead forecasts are studied; LSTM hyperparameters were not tuned; the LSTM uses raw levels and test-period CPI lies above the training range; VAR residuals show remaining autocorrelation and non-normality. See the paper (Section 8) for details.

## Use of AI tools

The author designed the study and the original methodology. Claude (Anthropic) was used to help write the Python code; the author reviewed the output. See the paper's Declaration on Use of Generative AI.

## Licence

Code: MIT (see `LICENSE`). Data are subject to the terms of their original sources.

## Contact

Rose Maria Jose, ranirosejose96@gmail.com
