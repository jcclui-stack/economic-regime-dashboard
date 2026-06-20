# economic-regime-dashboard
Regime-Switching VAR + Logistic Recession Probability Model Expanded Variable Set + 7 Bellwether Companies + Python Implementation + Dashboard
Model Specification
Two-regime Markov-switching framework (Expansion vs. Recession/Contraction regime) combined with a logistic model for recession probability.
Target variables:

Quarterly annualized real GDP growth (for the regime-switching regression)
Binary recession indicator (NBER-defined recession in the next 4 quarters) for the logistic model

Expanded predictor set (all lagged 1–2 quarters where appropriate):
Macro Survey Indicators

ISM Manufacturing New Orders (diffusion index)
Philadelphia Fed Future Capital Expenditures (or Business Conditions) diffusion index
University of Michigan Consumer Sentiment Index

Corporate Leading Indicators (newly expanded)

Bellwether Corporate Index (simple average of standardized scores across 7 companies):
Walmart, Procter & Gamble, Apple, JPMorgan Chase, Caterpillar, FedEx, Tesla
Components per company/quarter: YoY inventory growth (or days inventory for Tesla) + net guidance revision score (upward minus downward revisions in sales/EPS guidance)

Aggregate S&P 500 Guidance Revision Index (net % of companies raising guidance in the latest earnings season)
Duke CFO Survey – Capital Spending Plans (% of CFOs planning to increase capex)

Regime-switching component: Coefficients on the predictors are allowed to differ across Expansion and Recession regimes (Markov-switching regression). Transition probabilities between regimes are estimated.
Logistic component: Probability of recession in the next 4 quarters as a function of the same predictors + smoothed regime probability from the switching model.
This captures non-linear dynamics: e.g., rising inventories + weak guidance are much more negative in a high-recession-probability regime.
