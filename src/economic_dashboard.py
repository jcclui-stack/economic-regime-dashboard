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
os.makedirs("docs", exist_ok=True)

# ====================== LOAD DATA ======================
macro = pd.read_csv("data/macro_data.csv", parse_dates=["date"], index_col="date")
corp = pd.read_csv("data/bellwether_corporate_index.csv", parse_dates=["date"], index_col="date")

# Handle overlapping columns safely
df = macro.join(corp, how="outer", lsuffix="_macro", rsuffix="_corp").dropna()
print(f"Data loaded successfully. Shape: {df.shape}")

# Create Bellwether Corporate Index
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
print(f"Final data shape: {df.shape}")

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
print("Fitting Logistic Regression...")
logit_exog = sm.add_constant(df[[f"{v}_lag1" for v in predictors] + ["Regime_Prob_Recession_lag1"]])
logit_model = Logit(df["Recession_Next4Q"], logit_exog)
logit_results = logit_model.fit(disp=False)
df["Recession_Prob"] = logit_results.predict()

# ====================== GENERATE DASHBOARD ======================
latest = df.iloc[-1]
latest_date = df.index[-1]

print(f"Latest date: {latest_date}")

# Filter data from 2019 onwards for cleaner charts
df_plot = df[df.index >= '2019-01-01']

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Regime Probabilities
df_plot[["Regime_Prob_Expansion", "Regime_Prob_Recession"]].plot(
    ax=axes[0, 0], 
    title="Regime Probabilities (2019–2026)",
    linewidth=2
)
axes[0, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.7)
axes[0, 0].fill_between(df_plot.index, 0, 1, 
                        where=(df_plot["Regime_Prob_Recession"] > 0.5), 
                        color="red", alpha=0.15, label="Recession Regime Dominant")
axes[0, 0].legend()

# 2. 4-Quarter Recession Probability
df_plot["Recession_Prob"].plot(
    ax=axes[0, 1], 
    title="4-Quarter Recession Probability (2019–2026)", 
    color="red",
    linewidth=2
)
axes[0, 1].axhline(0.3, color="orange", linestyle="--", label="Warning Threshold (30%)")
axes[0, 1].axhline(0.5, color="darkred", linestyle="--", label="High Risk (50%)")
axes[0, 1].legend()
axes[0, 1].set_ylim(0, 1)

# 3. Bellwether Index vs GDP Growth
ax3 = axes[1, 0]
df_plot["Bellwether_Index"].plot(ax=ax3, label="Bellwether Index", color="blue", linewidth=2)
df_plot["GDP_Growth"].plot(ax=ax3.twinx(), label="GDP Growth (%)", color="green", alpha=0.7, linewidth=2)
ax3.set_title("Bellwether Corporate Index vs GDP Growth (2019–2026)")
ax3.legend(loc="upper left")

# 4. Latest Indicators Heatmap (still uses latest values)
latest_vals = df.loc[latest_date, [f"{v}_lag1" for v in predictors]]
sns.heatmap(latest_vals.to_frame().T, annot=True, cmap="RdYlGn", center=0, 
            ax=axes[1, 1], cbar=True, fmt=".2f")
axes[1, 1].set_title("Latest Standardized Indicators (Lagged)")

plt.tight_layout()
plt.savefig("docs/latest_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
# Generate HTML
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Economic Regime Dashboard</title>
</head>
<body style="font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 20px;">
    <h1>Economic Regime Dashboard</h1>
    <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
    
    <h2>Current Regime Probability</h2>
    <p>Expansion: <strong>{latest['Regime_Prob_Expansion']*100:.1f}%</strong></p>
    <p>Recession: <strong>{latest['Regime_Prob_Recession']*100:.1f}%</strong></p>
    
    <h2>4-Quarter Recession Probability</h2>
    <p style="font-size: 28px; color: red; font-weight: bold;">
        {latest['Recession_Prob']*100:.1f}%
    </p>
    
    <h2>Dashboard Visualization</h2>
    <img src="latest_dashboard.png" style="max-width: 100%; border: 1px solid #ccc; border-radius: 8px;">
</body>
</html>"""

with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Dashboard generated successfully!")
