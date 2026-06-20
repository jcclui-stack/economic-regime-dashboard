import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

print("=== Starting Economic Dashboard ===")
os.makedirs("dashboard", exist_ok=True)

# ====================== LOAD DATA ======================
macro = pd.read_csv("data/macro_data.csv", parse_dates=["date"], index_col="date")
corp = pd.read_csv("data/bellwether_corporate_index.csv", parse_dates=["date"], index_col="date")

# FIX: Handle overlapping columns
df = macro.join(corp, how="outer", lsuffix="_macro", rsuffix="_corp").dropna()
print(f"Data loaded successfully. Shape: {df.shape}")

# Create Bellwether Corporate Index (use the corp version if duplicated)
df["Bellwether_Index"] = df[[
    "WMT_Inv_Growth", "PG_Inv_Growth", "AAPL_Inv_Growth",
    "JPM_Inv_Growth", "CAT_Inv_Growth", "FDX_Inv_Growth",
    "TSLA_Inv_Days", "SPX_Guidance_Revision_corp", "Duke_CFO_Capex_corp"
]].mean(axis=1)

predictors = ["ISM_New_Orders", "Philly_Capex", "UMich_Sentiment", 
              "Bellwether_Index", "Duke_CFO_Capex_corp", "SPX_Guidance_Revision_corp"]

df[predictors] = (df[predictors] - df[predictors].mean()) / df[predictors].std()

for var in predictors + ["GDP_Growth"]:
    df[f"{var}_lag1"] = df[var].shift(1)

df = df.dropna()
print(f"Final data shape after processing: {df.shape}")

# ====================== REGIME-SWITCHING MODEL ======================
print("Fitting Markov Switching Model...")
endog = df["GDP_Growth"]
exog = df[[f"{v}_lag1" for v in predictors]]

ms_model = MarkovRegression(endog, k_regimes=2, exog=exog, 
                            switching_variance=True, switching_exog=True)
ms_results = ms_model.fit(disp=False)

df["Regime_Prob_Expansion"] = ms_results.smoothed_marginal_probabilities[0]
df["Regime_Prob_Recession"] = ms_results.smoothed_marginal_probabilities[1]

df["Regime_Prob_Recession_lag1"] = df["Regime_Prob_Recession"].shift(1)
df = df.dropna()

# ====================== LOGISTIC MODEL ======================
print("Fitting Logistic model...")
logit_exog = sm.add_constant(df[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
logit_model = Logit(df["Recession_Next4Q"], logit_exog)
logit_results = logit_model.fit(disp=False)
df["Recession_Prob"] = logit_results.predict()

# ====================== DASHBOARD ======================
latest = df.iloc[-1]
latest_date = df.index[-1]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
df[["Regime_Prob_Expansion", "Regime_Prob_Recession"]].plot(ax=axes[0, 0], title="Regime Probabilities")
df["Recession_Prob"].plot(ax=axes[0, 1], title="4Q Recession Probability", color="red")
df["Bellwether_Index"].plot(ax=axes[1, 0], label="Bellwether Index", color="blue")
df["GDP_Growth"].plot(ax=axes[1, 0].twinx(), label="GDP Growth", color="green", alpha=0.6)
axes[1, 0].set_title("Bellwether Index vs GDP Growth")

latest_vals = df.loc[latest_date, [f"{v}_lag1" for v in predictors]]
sns.heatmap(latest_vals.to_frame().T, annot=True, cmap="RdYlGn", center=0, ax=axes[1, 1], cbar=False)
axes[1, 1].set_title("Latest Standardized Indicators")

plt.tight_layout()
plt.savefig("dashboard/latest_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# Generate HTML
html = f"""<!DOCTYPE html>
<html><head><title>Economic Regime Dashboard</title></head>
<body style="font-family: Arial; max-width:1100px; margin:40px auto; padding:20px;">
<h1>Economic Regime Dashboard</h1>
<p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
<h2>Current Regime</h2>
<p>Expansion: <strong>{latest['Regime_Prob_Expansion']*100:.1f}%</strong></p>
<p>Recession: <strong>{latest['Regime_Prob_Recession']*100:.1f}%</strong></p>
<h2>4-Quarter Recession Probability</h2>
<p style="font-size:28px; color:red;"><strong>{latest['Recession_Prob']*100:.1f}%</strong></p>
<h2>Dashboard</h2>
<img src="latest_dashboard.png" style="max-width:100%; border:1px solid #ccc;">
</body></html>"""

with open("dashboard/index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Dashboard generated successfully!")
