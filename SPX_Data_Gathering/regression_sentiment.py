import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================
# 1) Load data
# =========================
df = pd.read_csv("spx_returns_sentiment_merged.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Rename returns column if needed
if "daily.returns" in df.columns:
    df = df.rename(columns={"daily.returns": "daily_returns"})

# =========================
# 2) Build market features
 # =========================

# Price proxy from cumulative log returns
df["cum_log_price"] = df["daily_returns"].cumsum()
df["price_proxy"] = np.exp(df["cum_log_price"])

# Return-based features
df["ret"] = df["daily_returns"]
df["ret_sq"] = df["daily_returns"] ** 2
df["abs_ret"] = df["daily_returns"].abs()

# Rolling mean features
df["roll_mean_5"] = df["daily_returns"].rolling(5).mean()
df["roll_mean_21"] = df["daily_returns"].rolling(21).mean()

# Rolling volatility features
df["roll_vol_5"] = df["daily_returns"].rolling(5).std()
df["roll_vol_21"] = df["daily_returns"].rolling(21).std()

# Downside volatility
df["downside_21"] = (
    df["daily_returns"]
    .where(df["daily_returns"] < 0, 0.0)
    .pow(2)
    .rolling(21)
    .mean()
    .pow(0.5)
)

# Drawdown over 21-day window
rolling_max_21 = df["price_proxy"].rolling(21).max()
df["drawdown_21"] = df["price_proxy"] / rolling_max_21 - 1.0

# Realized variance
df["realized_var_21"] = df["daily_returns"].rolling(21).var()

# =========================
# 3) Tanh sentiment transformation
# =========================

# Keep only rows where real sentiment exists
aligned = df.loc[df["daily_sentiment"].notna()].copy()

# Sentiment must be strictly inside (-1, 1)
# because arctanh(-1) and arctanh(1) are infinite
aligned["sentiment_clip"] = np.clip(
    aligned["daily_sentiment"],
    -0.999,
    0.999
)

# Transform sentiment from [-1, 1] to real line
# This is the dependent variable for regression
aligned["y_latent"] = np.arctanh(aligned["sentiment_clip"])

# Lagged transformed sentiment
aligned["y_lag1"] = aligned["y_latent"].shift(1)

# =========================
# 4) Select features
# =========================

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

model_df = aligned[
    ["date", "daily_sentiment", "sentiment_clip", "y_latent", "y_lag1"] + feature_cols
].dropna().copy()

# =========================
# 5) Fit tanh-bounded regression
# =========================

X = model_df[["y_lag1"] + feature_cols].to_numpy()
y = model_df["y_latent"].to_numpy()

reg = LinearRegression()
reg.fit(X, y)

# Predict in latent space
y_hat_latent = reg.predict(X)

# Convert predictions back to [-1, 1]
sentiment_hat = np.tanh(y_hat_latent)

# Actual sentiment in original bounded space
sentiment_actual = model_df["sentiment_clip"].to_numpy()

# Residuals in latent space
resid = y - y_hat_latent

# =========================
# 6) Output results
# =========================

coef_table = pd.DataFrame({
    "feature": ["intercept", "y_lag1"] + feature_cols,
    "coef": [reg.intercept_] + list(reg.coef_)
})

print("=== COEFFICIENTS ===")
print(coef_table.to_string(index=False))

print("\n=== MODEL PERFORMANCE ===")
print("Latent-space R^2:", r2_score(y, y_hat_latent))
print("Sentiment-space R^2:", r2_score(sentiment_actual, sentiment_hat))
print("Residual std in latent space:", resid.std(ddof=1))
print("Observations used:", len(model_df))

print("\n=== SENTIMENT RANGE CHECK ===")
print("Minimum predicted sentiment:", sentiment_hat.min())
print("Maximum predicted sentiment:", sentiment_hat.max())

# =========================
# 7) Save outputs
# =========================

coef_table.to_csv("tanh_sentiment_regression_coefficients.csv", index=False)

pred_df = model_df.copy()
pred_df["latent_hat"] = y_hat_latent
pred_df["sentiment_hat"] = sentiment_hat
pred_df["latent_resid"] = resid

pred_df.to_csv("tanh_sentiment_regression_fitted_values.csv", index=False)

print("\nSaved files:")
print("- tanh_sentiment_regression_coefficients.csv")
print("- tanh_sentiment_regression_fitted_values.csv")
