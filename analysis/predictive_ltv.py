"""
Predictive LTV — RavenStack

Goal: at ~90 days into an account's lifecycle, can we predict its eventual net
LTV using only signals that would have been available at that point? This is
the practical version of LTV modeling: score new customers early so marketing
spend / onboarding attention can be prioritized toward likely high-value
accounts, instead of waiting years to find out who was valuable.

Two things this script deliberately avoids, because they'd leak the answer:
- current_mrr / churn_flag / tenure -- these ARE (or are direct functions of)
  the outcome we're trying to predict before it has happened.
- any feature_usage or support_tickets rows after the 90-day cutoff.

Everything used as a feature is either fixed at signup (plan, seats, channel,
industry) or observed strictly within the first 90 days.

Outputs analysis/output/predictive_ltv_data.json for the dashboard.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA = "../dataset"
OUT = "output"
os.makedirs(OUT, exist_ok=True)

CUTOFF = pd.Timestamp("2024-12-31")
EARLY_WINDOW_DAYS = 90
ASSUMED_GROSS_MARGIN = 0.78
ASSUMED_SUPPORT_COST_PER_HOUR = 35

# ---------------------------------------------------------------- load ----
accounts = pd.read_csv(f"{DATA}/ravenstack_accounts.csv", parse_dates=["signup_date"])
subs = pd.read_csv(f"{DATA}/ravenstack_subscriptions.csv", parse_dates=["start_date", "end_date"])
usage = pd.read_csv(f"{DATA}/ravenstack_feature_usage.csv", parse_dates=["usage_date"])
tickets = pd.read_csv(f"{DATA}/ravenstack_support_tickets.csv", parse_dates=["submitted_at", "closed_at"])
churn = pd.read_csv(f"{DATA}/ravenstack_churn_events.csv", parse_dates=["churn_date"])

accounts["early_cutoff"] = accounts["signup_date"] + pd.Timedelta(days=EARLY_WINDOW_DAYS)

# =====================================================================
# TARGET: net LTV observed through CUTOFF (same definition as the main
# dashboard, recomputed here so this script is self-contained)
# =====================================================================
subs["end_eff"] = subs["end_date"].fillna(CUTOFF)
subs["months_active"] = ((subs["end_eff"] - subs["start_date"]).dt.days / 30.44).clip(lower=0.5)
subs["revenue_to_date"] = subs["mrr_amount"] * subs["months_active"]
realized_ltv = subs.groupby("account_id")["revenue_to_date"].sum().rename("realized_ltv")

ticket_cost = tickets.groupby("account_id")["resolution_time_hours"].sum() * ASSUMED_SUPPORT_COST_PER_HOUR
refund_cost = churn.groupby("account_id")["refund_amount_usd"].sum()

target = accounts[["account_id"]].merge(realized_ltv, on="account_id", how="left")
target["realized_ltv"] = target["realized_ltv"].fillna(0)
target = target.merge(ticket_cost.rename("support_cost"), on="account_id", how="left")
target = target.merge(refund_cost.rename("refund_cost"), on="account_id", how="left")
target[["support_cost", "refund_cost"]] = target[["support_cost", "refund_cost"]].fillna(0)
target["net_ltv"] = target["realized_ltv"] * ASSUMED_GROSS_MARGIN - target["support_cost"] - target["refund_cost"]

# =====================================================================
# FEATURES: only what's knowable within the first 90 days post-signup
# =====================================================================
# static, known at signup
feat = accounts[["account_id", "plan_tier", "referral_source", "industry", "seats", "is_trial", "signup_date", "early_cutoff"]].copy()

# first subscription terms (known at signup -- the deal that was struck)
first_sub = subs.sort_values("start_date").groupby("account_id").first()
feat = feat.merge(
    first_sub[["mrr_amount", "billing_frequency"]].rename(columns={"mrr_amount": "initial_mrr"}),
    on="account_id", how="left",
)

# early usage signal: feature_usage rows within 90 days of signup
usage_early = usage.merge(subs[["subscription_id", "account_id"]], on="subscription_id", how="left")
usage_early = usage_early.merge(feat[["account_id", "signup_date", "early_cutoff"]], on="account_id", how="left")
usage_early = usage_early[(usage_early["usage_date"] >= usage_early["signup_date"]) & (usage_early["usage_date"] <= usage_early["early_cutoff"])]
usage_agg = usage_early.groupby("account_id").agg(
    early_usage_events=("usage_id", "count"),
    early_usage_count=("usage_count", "sum"),
    early_usage_duration=("usage_duration_secs", "sum"),
    early_error_count=("error_count", "sum"),
    early_distinct_features=("feature_name", "nunique"),
    early_beta_usage=("is_beta_feature", "sum"),
)
feat = feat.merge(usage_agg, on="account_id", how="left")

# early support signal: tickets within 90 days of signup
tix_early = tickets.merge(feat[["account_id", "signup_date", "early_cutoff"]], on="account_id", how="left")
tix_early = tix_early[(tix_early["submitted_at"] >= tix_early["signup_date"]) & (tix_early["submitted_at"] <= tix_early["early_cutoff"])]
tix_agg = tix_early.groupby("account_id").agg(
    early_ticket_count=("ticket_id", "count"),
    early_avg_satisfaction=("satisfaction_score", "mean"),
    early_escalation_rate=("escalation_flag", "mean"),
)
feat = feat.merge(tix_agg, on="account_id", how="left")

num_fill_zero = [
    "early_usage_events", "early_usage_count", "early_usage_duration", "early_error_count",
    "early_distinct_features", "early_beta_usage", "early_ticket_count", "early_escalation_rate",
]
feat[num_fill_zero] = feat[num_fill_zero].fillna(0)
feat["early_avg_satisfaction"] = feat["early_avg_satisfaction"].fillna(feat["early_avg_satisfaction"].median())
feat["initial_mrr"] = feat["initial_mrr"].fillna(feat["initial_mrr"].median())
feat["billing_frequency"] = feat["billing_frequency"].fillna("monthly")

# =====================================================================
# Eligibility: only accounts whose 90-day window has fully elapsed by
# CUTOFF can be used for training (their target is "settled" and their
# features are complete). Everyone else is the accounts we'd actually
# SCORE with the trained model in production.
# =====================================================================
feat["is_mature"] = feat["early_cutoff"] <= CUTOFF
model_df = feat.merge(target[["account_id", "net_ltv"]], on="account_id", how="left")

train_pool = model_df[model_df["is_mature"]].copy()
score_pool = model_df[~model_df["is_mature"]].copy()

FEATURES_NUM = [
    "seats", "initial_mrr", "early_usage_events", "early_usage_count", "early_usage_duration",
    "early_error_count", "early_distinct_features", "early_beta_usage", "early_ticket_count",
    "early_avg_satisfaction", "early_escalation_rate",
]
FEATURES_CAT = ["plan_tier", "referral_source", "industry", "billing_frequency", "is_trial"]

X_train_pool = train_pool[FEATURES_NUM + FEATURES_CAT]
y_train_pool = train_pool["net_ltv"]

preprocess = ColumnTransformer([
    ("num", "passthrough", FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

# 5-fold cross-validation: at 500 rows, a single 75/25 split is small enough
# that "which rows happened to land in the test fold" swings R2 by more than
# the actual model signal does. CV averages that noise out.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "linear_regression": Pipeline([("prep", preprocess), ("model", LinearRegression())]),
    "random_forest": Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=8, random_state=42
    ))]),
}

results = {
    "baseline_mean": {
        "mae": round(float(np.mean(np.abs(y_train_pool - y_train_pool.mean()))), 2),
        "rmse": round(float(np.std(y_train_pool)), 2),
        "r2": 0.0,
        "r2_std": 0.0,
    }
}
for name, pipe in models.items():
    r2_scores = cross_val_score(pipe, X_train_pool, y_train_pool, cv=kf, scoring="r2")
    neg_mae = cross_val_score(pipe, X_train_pool, y_train_pool, cv=kf, scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(pipe, X_train_pool, y_train_pool, cv=kf, scoring="neg_root_mean_squared_error")
    results[name] = {
        "mae": round(float(-neg_mae.mean()), 2),
        "rmse": round(float(-neg_rmse.mean()), 2),
        "r2": round(float(r2_scores.mean()), 4),
        "r2_std": round(float(r2_scores.std()), 4),
    }

# out-of-fold predictions for every mature account -- an honest predicted-vs-
# actual view (each account is predicted by a model that never saw it train)
oof_pred = cross_val_predict(models["random_forest"], X_train_pool, y_train_pool, cv=kf)
test_pred_df = train_pool[["account_id", "referral_source"]].copy()
test_pred_df["actual_net_ltv"] = y_train_pool.values
test_pred_df["predicted_net_ltv"] = oof_pred

# refit the winning model (random forest) on the FULL mature pool for scoring
final_model = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=300, max_depth=5, min_samples_leaf=8, random_state=42
))])
final_model.fit(X_train_pool, y_train_pool)

# feature importances (mapped back to readable names)
ohe = final_model.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(FEATURES_CAT))
all_feature_names = FEATURES_NUM + cat_names
importances = final_model.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}).sort_values("importance", ascending=False)

# score the young (immature) accounts -- the actual production use case
score_pool = score_pool.copy()
if len(score_pool):
    score_pool["predicted_net_ltv"] = final_model.predict(score_pool[FEATURES_NUM + FEATURES_CAT])
else:
    score_pool["predicted_net_ltv"] = []

output = {
    "config": {
        "early_window_days": EARLY_WINDOW_DAYS,
        "train_accounts": int(len(X_train_pool)),
        "cv_folds": kf.get_n_splits(),
        "score_accounts_too_young_to_grade": int(len(score_pool)),
        "assumed_gross_margin": ASSUMED_GROSS_MARGIN,
    },
    "model_comparison": results,
    "feature_importance": imp_df.head(12).to_dict(orient="records"),
    "test_predictions": test_pred_df[["account_id", "referral_source", "actual_net_ltv", "predicted_net_ltv"]]
        .round(2).to_dict(orient="records"),
    "scored_young_accounts": score_pool[["account_id", "referral_source", "plan_tier", "signup_date", "predicted_net_ltv"]]
        .assign(signup_date=lambda d: d["signup_date"].dt.strftime("%Y-%m-%d"))
        .sort_values("predicted_net_ltv", ascending=False)
        .round(2).to_dict(orient="records"),
}

with open(f"{OUT}/predictive_ltv_data.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("MODEL COMPARISON (5-fold CV, net LTV in USD)")
for name, m in results.items():
    r2_std = m.get("r2_std", 0.0)
    print(f"  {name:20s} MAE={m['mae']:>10,.0f}  RMSE={m['rmse']:>10,.0f}  R2={m['r2']:.3f} (+/-{r2_std:.3f})")

print(f"\nTrain/CV pool (mature accounts): {len(X_train_pool)}")
print(f"Young accounts scored (no ground truth yet): {len(score_pool)}")

print("\nTOP FEATURE IMPORTANCES")
print(imp_df.head(12).to_string(index=False))

print("\nTOP 10 YOUNG ACCOUNTS BY PREDICTED NET LTV")
print(score_pool[["account_id", "referral_source", "plan_tier", "signup_date", "predicted_net_ltv"]]
      .sort_values("predicted_net_ltv", ascending=False).head(10).to_string(index=False))

print(f"\nWrote {OUT}/predictive_ltv_data.json")
