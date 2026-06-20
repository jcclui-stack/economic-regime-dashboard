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

print("=== Starting Economic Dashboard ===")
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

# ====================== MAIN MODEL ======================
print("Fitting Full Multi-Variable Model...")

endog = df["GDP_Growth"]
exog = df[[f"{v}_lag1" for v in predictors]]

ms_model = MarkovRegression(endog, k_regimes=2, exog=exog,
                            switching_variance=True, switching_exog=True)
ms_results = ms_model.fit(disp=False, maxiter=200)

df["Regime_Prob_Expansion"] = ms_results.smoothed_marginal_probabilities[0]
df["Regime_Prob_Recession"] = ms_results.smoothed_marginal_probabilities[1]
df["Regime_Prob_Recession_lag1"] = df["Regime_Prob_Recession"].shift(1)
df = df.dropna()

logit_exog = sm.add_constant(df[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
logit_model = Logit(df["Recession_Next4Q"], logit_exog)
logit_results = logit_model.fit(disp=False, maxiter=200)
df["Recession_Prob"] = logit_results.predict()

# ====================== ROBUST ROLLING BACKTEST ======================
print("\n=== Running Rolling Window Backtest ===")

window_size = 24
full_preds = []
bench_preds = []
actuals = []

for i in range(window_size, len(df) - 4):
    train = df.iloc[:i].copy()
    test = df.iloc[i:i+1].copy()
    
    try:
        if train["Recession_Next4Q"].nunique() < 2:
            continue
            
        endog_train = train["GDP_Growth"]
        exog_train = train[[f"{v}_lag1" for v in predictors]]
        
        ms_bt = MarkovRegression(endog_train, k_regimes=2, exog=exog_train,
                                 switching_variance=True, switching_exog=True)
        ms_bt_res = ms_bt.fit(disp=False, maxiter=100)
        
        train["Regime_Prob_Recession"] = ms_bt_res.smoothed_marginal_probabilities[1]
        train["Regime_Prob_Recession_lag1"] = train["Regime_Prob_Recession"].shift(1)
        train = train.dropna()
        
        if len(train) < 12:
            continue
            
        logit_exog_bt = sm.add_constant(train[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
        logit_bt = Logit(train["Recession_Next4Q"], logit_exog_bt)
        logit_bt_res = logit_bt.fit(disp=False, maxiter=100, method='lbfgs')
        
        test_exog = sm.add_constant(test[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
        full_p = logit_bt_res.predict(test_exog)[0]
        
        # Benchmark: UMich Sentiment
        bench_exog = sm.add_constant(train[["UMich_Sentiment_lag1"]])
        bench_model = Logit(train["Recession_Next4Q"], bench_exog)
        bench_res = bench_model.fit(disp=False, maxiter=100, method='lbfgs')
        
        test_bench = sm.add_constant(test[["UMich_Sentiment_lag1"]])
        bench_p = bench_res.predict(test_bench)[0]
        
        full_preds.append(full_p)
        bench_preds.append(bench_p)
        actuals.append(test["Recession_Next4Q"].values[0])
        
    except:
        continue

print(f"Successful out-of-sample predictions: {len(actuals)}")

def get_metrics(preds, actuals):
    if len(actuals) < 5:
        return {"AUC": 0.5, "Accuracy": 0.5, "Precision": 0.5, "Recall": 0.5}
    auc = roc_auc_score(actuals, preds)
    bin_preds = [1 if p > 0.5 else 0 for p in preds]
    return {
        "AUC": round(auc, 3),
        "Accuracy": round(accuracy_score(actuals, bin_preds), 3),
        "Precision": round(precision_score(actuals, bin_preds, zero_division=0), 3),
        "Recall": round(recall_score(actuals, bin_preds, zero_division=0), 3)
    }

full_m = get_metrics(full_preds, actuals)
bench_m = get_metrics(bench_preds, actuals)

print("\n=== Backtest Results ===")
print(f"{'Metric':<12} {'Full Model':<12} {'UMich Benchmark':<15}")
print("-" * 42)
for m in ["AUC", "Accuracy", "Precision", "Recall"]:
    print(f"{m:<12} {full_m[m]:<12} {bench_m[m]:<15}")

# ====================== PLOTS (from 2019) ======================
latest = df.iloc[-1]
latest_date = df.index[-1]
df_plot = df[df.index >= '2019-01-01']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df_plot[["Regime_Prob_Expansion", "Regime_Prob_Recession"]].plot(
    ax=axes[0, 0], title="Regime Probabilities (2019–2026)", linewidth=2)
axes[0, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.7)

df_plot["Recession_Prob"].plot(
    ax=axes[0, 1], title="4-Quarter Recession Probability", color="red", linewidth=2)
axes[0, 1].axhline(0.3, color="orange", linestyle="--", label="Warning (30%)")
axes[0, 1].legend()

ax3 = axes[1, 0]
df_plot["Bellwether_Index"].plot(ax=ax3, label="Bellwether Index", color="blue", linewidth=2)
df_plot["GDP_Growth"].plot(ax=ax3.twinx(), label="GDP Growth", color="green", alpha=0.7, linewidth=2)
ax3.set_title("Bellwether Index vs GDP Growth")
ax3.legend(loc="upper left")

latest_vals = df.loc[latest_date, [f"{v}_lag1" for v in predictors]]
sns.heatmap(latest_vals.to_frame().T, annot=True, cmap="RdYlGn", center=0, 
            ax=axes[1, 1], cbar=True, fmt=".2f")
axes[1, 1].set_title("Latest Standardized Indicators")

plt.tight_layout()
plt.savefig("docs/latest_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# ====================== HTML DASHBOARD ======================
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
    
    <h2>Backtest Performance (Rolling 6-year windows)</h2>
    <table style="border-collapse: collapse; width: 70%;">
        <tr style="background-color: #f2f2f2;">
            <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Full Model</th>
            <th style="padding: 8px; border: 1px solid #ddd;">UMich Benchmark</th>
        </tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">AUC-ROC</td><td style="padding: 8px; border: 1px solid #ddd;">{full_m['AUC']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_m['AUC']}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">Accuracy</td><td style="padding: 8px; border: 1px solid #ddd;">{full_m['Accuracy']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_m['Accuracy']}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">Precision</td><td style="padding: 8px; border: 1px solid #ddd;">{full_m['Precision']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_m['Precision']}</td></tr>
        <tr><td style="padding: 8px; border: 1px solid #ddd;">Recall</td><td style="padding: 8px; border: 1px solid #ddd;">{full_m['Recall']}</td><td style="padding: 8px; border: 1px solid #ddd;">{bench_m['Recall']}</td></tr>
    </table>
    
    <h2>Dashboard Visualization</h2>
    <img src="latest_dashboard.png" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
</body>
</html>"""

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Dashboard generated successfully!")
