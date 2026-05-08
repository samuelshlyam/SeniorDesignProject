import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# 1) Load data
# =========================
df = pd.read_csv("spx_returns_sentiment_merged.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

if "daily.returns" in df.columns:
    df = df.rename(columns={"daily.returns": "daily_returns"})

# =========================
# 2) Build features
# =========================
df["cum_log_price"] = df["daily_returns"].cumsum()
df["price_proxy"] = np.exp(df["cum_log_price"])

df["ret"] = df["daily_returns"]
df["ret_sq"] = df["daily_returns"] ** 2
df["abs_ret"] = df["daily_returns"].abs()

df["roll_mean_5"] = df["daily_returns"].rolling(5).mean()
df["roll_mean_21"] = df["daily_returns"].rolling(21).mean()

df["roll_vol_5"] = df["daily_returns"].rolling(5).std()
df["roll_vol_21"] = df["daily_returns"].rolling(21).std()

df["downside_21"] = (
    df["daily_returns"]
    .where(df["daily_returns"] < 0, 0.0)
    .pow(2)
    .rolling(21)
    .mean()
    .pow(0.5)
)

rolling_max_21 = df["price_proxy"].rolling(21).max()
df["drawdown_21"] = df["price_proxy"] / rolling_max_21 - 1.0

df["realized_var_21"] = df["daily_returns"].rolling(21).var()


sent = df["daily_sentiment"].dropna()
ranks = sent.rank(method="average").to_numpy()
probs = (ranks - 0.5) / len(sent)
probs = np.clip(probs, 1e-6, 1 - 1e-6)
z_vals = norm.ppf(probs)

aligned = df.loc[df["daily_sentiment"].notna()].copy()
aligned["z"] = z_vals
aligned["z_lag1"] = aligned["z"].shift(1)

feature_cols = [
    "ret",
    "ret_sq",
    "abs_ret",
    "roll_mean_5",
    "roll_mean_21",
    "roll_vol_5",
    "roll_vol_21",
    "downside_21",
    "drawdown_21",
    "realized_var_21",
]

model_df = aligned[["z", "z_lag1"] + feature_cols].dropna().copy()

# =========================
# 4) Fit regression
# =========================
X = model_df[["z_lag1"] + feature_cols].to_numpy()
y = model_df["z"].to_numpy()

reg = LinearRegression()
reg.fit(X, y)

y_hat = reg.predict(X)
resid = y - y_hat

coef_table = pd.DataFrame({
    "feature": ["intercept", "z_lag1"] + feature_cols,
    "coef": [reg.intercept_] + list(reg.coef_)
})

print("=== COEFFICIENTS ===")
print(coef_table.to_string(index=False))

print("\nLatent R^2:", r2_score(y, y_hat))
print("Residual std:", resid.std(ddof=1))
print("Observations used:", len(model_df))

# Save outputs 
coef_table.to_csv("latent_sentiment_regression_outputs.csv", index=False)

