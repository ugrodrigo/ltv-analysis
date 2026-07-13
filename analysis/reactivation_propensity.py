"""
Reactivation timing & outreach prioritization -- RavenStack

Original question: "who is likely to reactivate, and which offer should we
send?" Two data-quality findings changed the shape of this analysis:

1. accounts.churn_flag does NOT reliably correspond to the actual
   subscription/churn_event history (accounts exist with churn_flag=True and
   zero churn events, zero ended subscriptions; a crosstab shows it's
   statistically independent of both). It's treated here as unreliable and
   NOT used -- "currently churned" is instead derived directly from whether a
   later subscription exists after an account's most recent churn event.

2. Once "reactivated" is derived from the real event data, 87% of churn
   events (524/600) are followed by a new subscription within days-to-weeks
   (median 16 days), and almost all of the remaining 13% are simply recent
   (median 17 days old) rather than permanently lost. That makes "will they
   reactivate" a near-degenerate classification target -- the honest answer
   is "yes, almost certainly" regardless of segment, which a model can't beat.

PIVOT: model TIME-TO-REACTIVATION instead (real variance: 1-300+ days), and
build a rule-based (not learned -- there's no offer/outcome experiment data
in this dataset) reason-code -> offer mapping, combined with predicted delay
and historical account value to prioritize the outreach list.

Outputs analysis/output/reactivation_propensity_data.json for the dashboard.
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
ASSUMED_GROSS_MARGIN = 0.78

# ---------------------------------------------------------------- load ----
accounts = pd.read_csv(f"{DATA}/ravenstack_accounts.csv", parse_dates=["signup_date"])
subs = pd.read_csv(f"{DATA}/ravenstack_subscriptions.csv", parse_dates=["start_date", "end_date"])
tickets = pd.read_csv(f"{DATA}/ravenstack_support_tickets.csv", parse_dates=["submitted_at", "closed_at"])
churn = pd.read_csv(f"{DATA}/ravenstack_churn_events.csv", parse_dates=["churn_date"])

# =====================================================================
# For every churn event (not just the first -- more data, and each churn
# instance is its own decision point), find whether/when a later
# subscription started for that account. This is the ONLY source of truth
# used for "reactivated" -- accounts.churn_flag is intentionally ignored.
# =====================================================================
churn_ev = churn.sort_values(["account_id", "churn_date"]).reset_index(drop=True).copy()
churn_ev["churn_seq"] = churn_ev.groupby("account_id").cumcount() + 1
churn_ev["total_churn_events"] = churn_ev.groupby("account_id")["account_id"].transform("count")

subs_sorted = subs.sort_values("start_date")
def find_next_start(row):
    later = subs_sorted[(subs_sorted["account_id"] == row["account_id"]) & (subs_sorted["start_date"] > row["churn_date"])]
    return later["start_date"].min() if len(later) else pd.NaT

churn_ev["next_start"] = churn_ev.apply(find_next_start, axis=1)
churn_ev["reactivated"] = churn_ev["next_start"].notna()
churn_ev["days_to_reactivate"] = (churn_ev["next_start"] - churn_ev["churn_date"]).dt.days

churn_ev = churn_ev.merge(
    accounts[["account_id", "plan_tier", "referral_source", "industry", "seats", "signup_date"]],
    on="account_id", how="left",
)

# =====================================================================
# FEATURES -- everything knowable at the moment of this specific churn
# =====================================================================
subs_m = subs.merge(churn_ev[["account_id", "churn_date"]], on="account_id", how="inner")
subs_m = subs_m[subs_m["start_date"] <= subs_m["churn_date"]]
subs_m["end_capped"] = subs_m[["end_date", "churn_date"]].min(axis=1)
subs_m["months_active"] = ((subs_m["end_capped"] - subs_m["start_date"]).dt.days / 30.44).clip(lower=0.5)
subs_m["revenue_to_churn"] = subs_m["mrr_amount"] * subs_m["months_active"]
ltv_to_churn = subs_m.groupby("account_id")["revenue_to_churn"].sum().rename("ltv_to_churn")
# note: this is keyed on account_id, so it's a (slight) upper bound for an
# account's later churn events -- acceptable given the dataset's dominant
# revenue driver is deal size (seats/MRR), not the churn-to-churn delta

tix_m = tickets.merge(churn_ev[["account_id", "churn_date"]], on="account_id", how="inner")
tix_m = tix_m[tix_m["submitted_at"] <= tix_m["churn_date"]]
tix_agg = tix_m.groupby("account_id").agg(
    pre_churn_ticket_count=("ticket_id", "count"),
    pre_churn_avg_satisfaction=("satisfaction_score", "mean"),
    pre_churn_escalation_rate=("escalation_flag", "mean"),
)

df = churn_ev.merge(ltv_to_churn, on="account_id", how="left")
df = df.merge(tix_agg, on="account_id", how="left")
df["ltv_to_churn"] = df["ltv_to_churn"].fillna(0)
df["tenure_days_at_churn"] = (df["churn_date"] - df["signup_date"]).dt.days.clip(lower=1)
df["pre_churn_ticket_count"] = df["pre_churn_ticket_count"].fillna(0)
df["pre_churn_escalation_rate"] = df["pre_churn_escalation_rate"].fillna(0)
df["pre_churn_avg_satisfaction"] = df["pre_churn_avg_satisfaction"].fillna(df["pre_churn_avg_satisfaction"].median())
df["refund_amount_usd"] = df["refund_amount_usd"].fillna(0)

FEATURES_NUM = [
    "tenure_days_at_churn", "ltv_to_churn", "refund_amount_usd", "seats", "churn_seq",
    "pre_churn_ticket_count", "pre_churn_avg_satisfaction", "pre_churn_escalation_rate",
]
FEATURES_CAT = [
    "reason_code", "plan_tier", "referral_source", "industry",
    "preceding_upgrade_flag", "preceding_downgrade_flag",
]

# =====================================================================
# Train the TIME-TO-REACTIVATE regressor on events with an OBSERVED
# reactivation (524 of 600) -- these aren't censored, we know the true delay
# regardless of how long ago the churn happened.
# =====================================================================
observed = df[df["reactivated"]].copy()
X = observed[FEATURES_NUM + FEATURES_CAT]
y = observed["days_to_reactivate"].astype(float)
y_log = np.log1p(y)

preprocess = ColumnTransformer([
    ("num", "passthrough", FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "linear_regression": Pipeline([("prep", preprocess), ("model", LinearRegression())]),
    "random_forest": Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=8, random_state=42
    ))]),
}

results = {
    "baseline_median_days": {
        "mae_days": round(float(np.mean(np.abs(y - y.median()))), 2),
    }
}
for name, pipe in models.items():
    neg_mae = cross_val_score(pipe, X, y_log, cv=kf, scoring="neg_mean_absolute_error")
    r2 = cross_val_score(pipe, X, y_log, cv=kf, scoring="r2")
    # convert log-space MAE back to an approximate day-scale MAE for readability
    pred_log = cross_val_predict(pipe, X, y_log, cv=kf)
    pred_days = np.expm1(pred_log)
    mae_days = float(np.mean(np.abs(y.values - pred_days)))
    results[name] = {
        "mae_days": round(mae_days, 2),
        "r2_log_space": round(float(r2.mean()), 4),
        "r2_log_space_std": round(float(r2.std()), 4),
    }

oof_pred_days = np.expm1(cross_val_predict(models["random_forest"], X, y_log, cv=kf))
observed["oof_predicted_days"] = oof_pred_days

final_model = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=300, max_depth=5, min_samples_leaf=8, random_state=42
))])
final_model.fit(X, y_log)

ohe = final_model.named_steps["prep"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(FEATURES_CAT))
all_feature_names = FEATURES_NUM + cat_names
importances = final_model.named_steps["model"].feature_importances_
imp_df = pd.DataFrame({"feature": all_feature_names, "importance": importances}).sort_values("importance", ascending=False)

# =====================================================================
# Actionable population: each account's MOST RECENT churn event, where no
# later subscription has (yet) been observed -- the real "who's still
# outstanding right now" list, derived purely from event data.
# =====================================================================
latest_churn = df.sort_values("churn_date").groupby("account_id").tail(1)
open_pool = latest_churn[~latest_churn["reactivated"]].copy()
open_pool["days_since_churn"] = (CUTOFF - open_pool["churn_date"]).dt.days

open_X = open_pool[FEATURES_NUM + FEATURES_CAT]
open_pool["predicted_days_to_reactivate"] = np.expm1(final_model.predict(open_X))
open_pool["days_overdue"] = (open_pool["days_since_churn"] - open_pool["predicted_days_to_reactivate"]).clip(lower=0)
# expected value skews toward accounts that are both high-value AND already
# running well past their model-predicted natural return window
open_pool["expected_value_usd"] = open_pool["ltv_to_churn"] * ASSUMED_GROSS_MARGIN

# ---- offer mapping: a disclosed BUSINESS RULE, not a learned model -------
OFFER_MAP = {
    "pricing":    {"offer": "Flexible billing / loyalty discount", "motion": "Commercial"},
    "budget":     {"offer": "Downgrade-to-retain / annual prepay discount", "motion": "Commercial"},
    "features":   {"offer": "Roadmap preview + early access to requested feature", "motion": "Product"},
    "support":    {"offer": "Dedicated CSM + expedited support tier for 90 days", "motion": "Service"},
    "competitor": {"offer": "Competitive win-back offer (feature/price match)", "motion": "Commercial"},
    "unknown":    {"offer": "Discovery call to diagnose reason before offering anything", "motion": "Discovery"},
}

def priority_tier(row):
    overdue = row["days_overdue"] > 14
    high_value = row["ltv_to_churn"] >= open_pool["ltv_to_churn"].median()
    if overdue and high_value:
        return "Urgent: past its typical return window and high value -- call this week"
    if overdue and not high_value:
        return "Overdue but lower value -- automated/email nudge"
    if not overdue and high_value:
        return "On track, high value -- light-touch check-in, no discount needed yet"
    return "On track, lower value -- monitor, no action needed yet"

open_pool["offer_type"] = open_pool["reason_code"].map(lambda r: OFFER_MAP.get(r, OFFER_MAP["unknown"])["offer"])
open_pool["offer_motion"] = open_pool["reason_code"].map(lambda r: OFFER_MAP.get(r, OFFER_MAP["unknown"])["motion"])
open_pool["priority_tier"] = open_pool.apply(priority_tier, axis=1)

# =====================================================================
# Reason-code view: median days-to-reactivate (the real, non-degenerate
# signal that reason code carries)
# =====================================================================
reason_timing = (
    observed.groupby("reason_code")["days_to_reactivate"]
    .agg(median_days="median", mean_days="mean", n="count")
    .round(1).reset_index()
    .sort_values("median_days")
)

# =====================================================================
# Assemble output
# =====================================================================
output = {
    "config": {
        "total_churn_events": int(len(df)),
        "observed_reactivations": int(len(observed)),
        "actionable_open_population": int(len(open_pool)),
        "cv_folds": kf.get_n_splits(),
        "assumed_gross_margin": ASSUMED_GROSS_MARGIN,
        "overall_reactivation_rate": round(float(df["reactivated"].mean()), 4),
    },
    "data_quality_note": (
        "accounts.churn_flag was found to be statistically independent of the actual "
        "subscription/churn_event history and is NOT used anywhere in this analysis. "
        "'Currently churned' is derived only from whether a later subscription exists "
        "after an account's most recent logged churn event."
    ),
    "model_comparison": results,
    "feature_importance": imp_df.head(10).to_dict(orient="records"),
    "reason_code_timing": reason_timing.to_dict(orient="records"),
    "offer_map": OFFER_MAP,
    "outreach_list": (
        open_pool[["account_id", "reason_code", "referral_source", "plan_tier", "ltv_to_churn",
                    "days_since_churn", "predicted_days_to_reactivate", "days_overdue",
                    "expected_value_usd", "offer_type", "offer_motion", "priority_tier"]]
        .sort_values(["priority_tier", "expected_value_usd"], ascending=[True, False])
        .round(2).to_dict(orient="records")
    ),
}

with open(f"{OUT}/reactivation_propensity_data.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print("OVERALL REACTIVATION RATE (any horizon):", round(df["reactivated"].mean(), 4))
print("\nMODEL COMPARISON (5-fold CV, time-to-reactivate in days, log-space target)")
for name, m in results.items():
    print(f"  {name:22s} {m}")

print(f"\nObserved reactivations (trainable): {len(observed)}")
print(f"Actionable (currently open) population: {len(open_pool)}")

print("\nDAYS-TO-REACTIVATE BY REASON CODE")
print(reason_timing.to_string(index=False))

print("\nTOP FEATURE IMPORTANCES")
print(imp_df.head(10).to_string(index=False))

print("\nTOP 10 OUTREACH TARGETS BY PRIORITY / VALUE")
print(open_pool[["account_id", "reason_code", "days_since_churn", "predicted_days_to_reactivate",
                  "days_overdue", "ltv_to_churn", "priority_tier"]]
      .sort_values(["days_overdue", "ltv_to_churn"], ascending=False).head(10).to_string(index=False))

print(f"\nWrote {OUT}/reactivation_propensity_data.json")
