import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

print("=== Starting Economic Dashboard with Backtesting ===")
os.makedirs("docs", exist_ok=True)

# ====================== LOAD DATA ======================
macro = pd.read_csv("data/macro_data.csv", parse_dates=["date"], index_col="date")
corp = pd.read_csv("data/bellwether_corporate_index.csv", parse_dates=["date"], index_col="date")

df = macro.join(corp, how="outer", lsuffix="_macro", rsuffix="_corp").dropna()

df["Bellwether_Index"] = df[[
    "WMT_Inv_Growth", "PG_Inv_Growth", "AAPL_Inv_Growth",
    "JPM_Inv_Growth", "CAT_Inv_Growth", "FDX_Inv_Growth",
    "TSLA_Inv_Days", "SPX_Guidance_Revision_corp", "Duke_CFO_Capex_corp"
]].mean(axis=1)

predictors = [
    "ISM_New_Orders", "Philly_Capex", "UMich_Sentiment",
    "Bellwether_Index", "Duke_CFO_Capex_corp", "SPX_Guidance_Revision_corp"
]

df[predictors] = (df[predictors] - df[predictors].mean()) / df[predictors].std()

for var in predictors + ["GDP_Growth"]:
    df[f"{var}_lag1"] = df[var].shift(1)

df = df.dropna()

# ====================== MAIN MULTI-VARIABLE MODEL ======================
print("Fitting Full Multi-Variable Model...")

endog = df["GDP_Growth"]
exog = df[[f"{v}_lag1" for v in predictors]]

ms_model = MarkovRegression(endog, k_regimes=2, exog=exog,
                            switching_variance=True, switching_exog=True)
ms_results = ms_model.fit(disp=False)

df["Regime_Prob_Expansion"] = ms_results.smoothed_marginal_probabilities[0]
df["Regime_Prob_Recession"] = ms_results.smoothed_marginal_probabilities[1]
df["Regime_Prob_Recession_lag1"] = df["Regime_Prob_Recession"].shift(1)
df = df.dropna()

logit_exog = sm.add_constant(df[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
logit_model = Logit(df["Recession_Next4Q"], logit_exog)
logit_results = logit_model.fit(disp=False)
df["Recession_Prob"] = logit_results.predict()

# ====================== ROLLING WINDOW BACKTEST ======================
print("\n=== Running Rolling Window Backtest ===")

window_size = 32
full_model_preds, benchmark_preds, actuals = [], [], []

for i in range(window_size, len(df) - 1):
    train = df.iloc[:i].copy()
    test = df.iloc[i:i+1].copy()
    
    try:
        # --- Full Model ---
        endog_train = train["GDP_Growth"]
        exog_train = train[[f"{v}_lag1" for v in predictors]]
        
        ms_bt = MarkovRegression(endog_train, k_regimes=2, exog=exog_train,
                                 switching_variance=True, switching_exog=True)
        ms_bt_results = ms_bt.fit(disp=False)
        
        train["Regime_Prob_Recession"] = ms_bt_results.smoothed_marginal_probabilities[1]
        train["Regime_Prob_Recession_lag1"] = train["Regime_Prob_Recession"].shift(1)
        train = train.dropna()
        
        logit_exog_bt = sm.add_constant(train[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
        logit_bt = Logit(train["Recession_Next4Q"], logit_exog_bt)
        logit_bt_results = logit_bt.fit(disp=False)
        
        test_exog = sm.add_constant(test[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
        full_prob = logit_bt_results.predict(test_exog)[0]
        
        # --- Benchmark: UMich Sentiment Only ---
        benchmark_exog = sm.add_constant(train[["UMich_Sentiment_lag1"]])
        benchmark_model = Logit(train["Recession_Next4Q"], benchmark_exog)
        benchmark_results = benchmark_model.fit(disp=False)
        
        test_bench_exog = sm.add_constant(test[["UMich_Sentiment_lag1"]])
        bench_prob = benchmark_results.predict(test_bench_exog)[0]
        
        full_model_preds.append(full_prob)
        benchmark_preds.append(bench_prob)
        actuals.append(test["Recession_Next4Q"].values[0])
        
    except:
        continue

# ====================== PERFORMANCE METRICS ======================
def get_metrics(preds, actuals):
    auc = roc_auc_score(actuals, preds)
    preds_bin = [1 if p > 0.5 else 0 for p in preds]
    return {
        "AUC": round(auc, 3),
        "Accuracy": round(accuracy_score(actuals, preds_bin), 3),
        "Precision": round(precision_score(actuals, preds_bin, zero_division=0), 3),
        "Recall": round(recall_score(actuals, preds_bin, zero_division=0), 3)
    }

full_metrics = get_metrics(full_model_preds, actuals)
bench_metrics = get_metrics(benchmark_preds, actuals)

print("\n=== Backtest Results ===")
print(f"{'Metric':<12} {'Full Model':<12} {'UMich Benchmark':<15}")
print("-" * 40)
for metric in ["AUC", "Accuracy", "Precision", "Recall"]:
    print(f"{metric:<12} {full_metrics[metric]:<12} {bench_metrics[metric]:<15}")

# ====================== GENERATE DASHBOARD ======================
latest = df.iloc[-1]
latest_date = df.index[-1]
df_plot = df[df.index >= '2019-01-01']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df_plot[["Regime_Prob_Expansion", "Regime_Prob_Recession"]].plot(ax=axes[0, 0], title="Regime Probabilities (2019–2026)", linewidth=2)
axes[0, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.7)

df_plot["Recession_Prob"].plot(ax=axes[0, 1], title="4-Quarter Recession Probability", color="red", linewidth=2)
axes[0, 1].axhline(0.3, color="orange", linestyle="--", label="Warning Threshold")
axes[0, 1].legend()

ax3 = axes[1, 0]
df_plot["Bellwether_Index"].plot(ax=ax3, label="Bellwether Index", color="blue", linewidth=2)
df_plot["GDP_Growth"].plot(ax=ax3.twinx(), label="GDP Growth", color="green", alpha=0.7, linewidth=2)
ax3.set_title("Bellwether Index vs GDP Growth")
ax3.legend(loc="upper left")

latest_vals = df.loc[latest_date, [f"{v}_lag1" for v in predictors]]
sns.heatmap(latest_vals.to_frame().T, annot=True, cmap="RdYlGn", center=0, ax=axes[1, 1], cbar=True, fmt=".2f")
axes[1, 1].set_title("Latest Standardized Indicators")

plt.tight_layout()
plt.savefig("docs/latest_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# HTML Dashboard with Benchmark Comparison
html_content = f"""<!DOCTYPE html>
<html>
<head><title>Economic Regime Dashboard</title></head>
<body style="font-family: Arial; max-width: 1100px; margin: 40px auto; padding: 20px;">
    <h1>Economic Regime Dashboard</h1>
    <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
    
    <h2>Current Regime Probability</h2>
    <p>Expansion: <strong>{latest['Regime_Prob_Expansion']*100:.1f}%</strong></p>
    <p>Recession: <strong>{latest['Regime_Prob_Recession']*100:.1f}%</strong></p>
    
    <h2>4-Quarter Recession Probability</h2>
    <p style="font-size: 28px; color: red; font-weight: bold;">{latest['Recession_Prob']*100:.1f}%</p>
    
    <h2>Backtest Performance Comparison</h2>
    <table style="border-collapse: collapse; width: 70%; margin-top: 10px;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Metric</th>
            <th style="padding: 10px; border: 1px solid #ddd;">Full Model</th>
            <th style="padding: 10px; border: 1px solid #ddd;">UMich Sentiment (Benchmark)</th>
        </tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">AUC-ROC</td><td style="padding: 8px; border: 1px solid #ddd;">{full_metrics['AUC']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_metrics['AUC']}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">Accuracy</td><td style="padding: 8px; border: 1px solid #ddd;">{full_metrics['Accuracy']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_metrics['Accuracy']}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">Precision</td><td style="padding: 8px; border: 1px solid #ddd;">{full_metrics['Precision']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_metrics['Precision']}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">Recall</td><td style="padding: 8px; border: 1px solid #ddd;">{full_metrics['Recall']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_metrics['Recall']}</td></tr>
    </table>
    <p><small>Based on {len(full_model_preds)} out-of-sample predictions (Rolling 8-year windows)</small></p>
    
    <h2>Dashboard Visualization</h2>
    <img src="latest_dashboard.png" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
</body>
</html>"""

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\n✅ Dashboard generated successfully with benchmark comparison!")
