"""
RavenStack LTV & Reactivation Analysis
Mirrors the scope of a Staff Marketing Analyst (Global Reactivations) role:
- Channel-level LTV and retention
- Cohort survival curves
- Reactivation impact on lifetime value
- Churn driver / support correlation
- KPI-tree style executive summary

Outputs a single JSON file (output/dashboard_data.json) consumed by the
HTML dashboard artifact, plus printed sanity-check tables.
"""
import json
import numpy as np
import pandas as pd

DATA = "../dataset"
OUT = "output"
import os
os.makedirs(OUT, exist_ok=True)

CUTOFF = pd.Timestamp("2024-12-31")  # observation date (max date in dataset)

# ---- disclosed assumptions for margin-adjusted (net) LTV -----------------
# The dataset has no COGS/hosting/salary fields, so a true contribution-margin
# LTV can't be derived from data alone. Two cost signals ARE real and in the
# data (refunds, support hours); everything else (hosting, infra, payment
# processing) is covered by an industry-benchmark SaaS gross margin assumption.
# These are assumptions, not measurements -- flagged explicitly in the output.
ASSUMED_GROSS_MARGIN = 0.78          # typical public SaaS benchmark (75-85%)
ASSUMED_SUPPORT_COST_PER_HOUR = 35   # fully-loaded support agent cost, USD/hr

# ---------------------------------------------------------------- load ----
accounts = pd.read_csv(f"{DATA}/ravenstack_accounts.csv", parse_dates=["signup_date"])
subs = pd.read_csv(f"{DATA}/ravenstack_subscriptions.csv", parse_dates=["start_date", "end_date"])
usage = pd.read_csv(f"{DATA}/ravenstack_feature_usage.csv", parse_dates=["usage_date"])
tickets = pd.read_csv(f"{DATA}/ravenstack_support_tickets.csv", parse_dates=["submitted_at", "closed_at"])
churn = pd.read_csv(f"{DATA}/ravenstack_churn_events.csv", parse_dates=["churn_date"])

# ---------------------------------------------------- revenue-to-date -----
subs["end_eff"] = subs["end_date"].fillna(CUTOFF)
subs["months_active"] = ((subs["end_eff"] - subs["start_date"]).dt.days / 30.44).clip(lower=0.5)
subs["revenue_to_date"] = subs["mrr_amount"] * subs["months_active"]

acct_revenue = subs.groupby("account_id")["revenue_to_date"].sum().rename("realized_ltv")
acct_active_mrr = (
    subs[subs["end_date"].isna()].groupby("account_id")["mrr_amount"].sum().rename("current_mrr")
)

acc = accounts.merge(acct_revenue, on="account_id", how="left").merge(
    acct_active_mrr, on="account_id", how="left"
)
acc["realized_ltv"] = acc["realized_ltv"].fillna(0)
acc["current_mrr"] = acc["current_mrr"].fillna(0)

# ------------------------------------------------- churn date / tenure ----
first_churn = churn.groupby("account_id")["churn_date"].min().rename("first_churn_date")
churn_event_counts = churn.groupby("account_id").size().rename("churn_event_count")
any_reactivation = churn.groupby("account_id")["is_reactivation"].any().rename("has_reactivated")

acc = acc.merge(first_churn, on="account_id", how="left")
acc = acc.merge(churn_event_counts, on="account_id", how="left")
acc = acc.merge(any_reactivation, on="account_id", how="left")
acc["churn_event_count"] = acc["churn_event_count"].fillna(0).astype(int)
acc["has_reactivated"] = acc["has_reactivated"].fillna(False)

acc["tenure_end"] = np.where(acc["churn_flag"], acc["first_churn_date"], CUTOFF)
acc["tenure_end"] = pd.to_datetime(acc["tenure_end"])
acc["tenure_days"] = (acc["tenure_end"] - acc["signup_date"]).dt.days.clip(lower=1)
acc["tenure_months"] = acc["tenure_days"] / 30.44

# segment: lifecycle status
def lifecycle(row):
    if row["churn_event_count"] >= 2 or row["has_reactivated"]:
        return "serial_churner_reactivated"
    if row["churn_flag"]:
        return "single_churn"
    return "never_churned"

acc["lifecycle_segment"] = acc.apply(lifecycle, axis=1)

# ---------------------------------------------- projected (formula) LTV ---
# classic SaaS LTV = ARPA / monthly churn rate, computed per segment
def projected_ltv(df):
    churned = df[df["churn_flag"]]
    active = df[~df["churn_flag"]]
    monthly_churn_rate = 1 / churned["tenure_months"].mean() if len(churned) and churned["tenure_months"].mean() > 0 else np.nan
    arpa = df["current_mrr"].replace(0, np.nan).mean()
    if arpa is np.nan or pd.isna(arpa) or pd.isna(monthly_churn_rate) or monthly_churn_rate == 0:
        return np.nan
    return arpa / monthly_churn_rate

# ------------------------------------------------------- support signal ---
ticket_agg = tickets.groupby("account_id").agg(
    ticket_count=("ticket_id", "count"),
    avg_satisfaction=("satisfaction_score", "mean"),
    escalation_rate=("escalation_flag", "mean"),
    avg_resolution_hrs=("resolution_time_hours", "mean"),
    total_resolution_hrs=("resolution_time_hours", "sum"),
)
acc = acc.merge(ticket_agg, on="account_id", how="left")
acc["total_resolution_hrs"] = acc["total_resolution_hrs"].fillna(0)

# ------------------------------------------ margin-adjusted (net) LTV -----
# net_ltv = realized revenue x assumed gross margin, minus REAL costs we can
# observe directly (refunds/credits + support cost-to-serve). Only the
# hosting/infra/payment-processing slice is an assumption (the margin %);
# refunds and support cost are actual figures from the dataset.
total_refund_by_acct = churn.groupby("account_id")["refund_amount_usd"].sum().rename("total_refund_usd")
acc = acc.merge(total_refund_by_acct, on="account_id", how="left")
acc["total_refund_usd"] = acc["total_refund_usd"].fillna(0)

acc["support_cost_usd"] = acc["total_resolution_hrs"] * ASSUMED_SUPPORT_COST_PER_HOUR
acc["gross_margin_usd"] = acc["realized_ltv"] * ASSUMED_GROSS_MARGIN
acc["net_ltv"] = acc["gross_margin_usd"] - acc["support_cost_usd"] - acc["total_refund_usd"]

# =====================================================================
# 1. Executive KPI summary
# =====================================================================
kpi = {
    "total_accounts": int(len(acc)),
    "active_accounts": int((~acc["churn_flag"]).sum()),
    "churned_accounts": int(acc["churn_flag"].sum()),
    "overall_churn_rate": round(acc["churn_flag"].mean(), 4),
    "total_realized_ltv": round(acc["realized_ltv"].sum(), 2),
    "avg_realized_ltv": round(acc["realized_ltv"].mean(), 2),
    "total_net_ltv": round(acc["net_ltv"].sum(), 2),
    "avg_net_ltv": round(acc["net_ltv"].mean(), 2),
    "total_support_cost": round(acc["support_cost_usd"].sum(), 2),
    "total_refund_cost": round(acc["total_refund_usd"].sum(), 2),
    "net_ltv_pct_of_gross": round(acc["net_ltv"].sum() / acc["realized_ltv"].sum(), 4),
    "current_total_mrr": round(acc["current_mrr"].sum(), 2),
    "current_total_arr": round(acc["current_mrr"].sum() * 12, 2),
    "reactivation_rate_among_churned": round(acc.loc[acc["churn_flag"], "has_reactivated"].mean(), 4)
    if acc["churn_flag"].sum() else 0,
    "accounts_with_multiple_churn_events": int((acc["churn_event_count"] >= 2).sum()),
    "avg_tenure_months_churned": round(acc.loc[acc["churn_flag"], "tenure_months"].mean(), 2),
    "avg_tenure_months_active": round(acc.loc[~acc["churn_flag"], "tenure_months"].mean(), 2),
}

# =====================================================================
# 2. LTV & retention by acquisition channel (referral_source)
# =====================================================================
channel_rows = []
for ch, g in acc.groupby("referral_source"):
    churned = g[g["churn_flag"]]
    monthly_churn_rate = (1 / churned["tenure_months"].mean()) if len(churned) and churned["tenure_months"].mean() > 0 else np.nan
    arpa = g.loc[g["current_mrr"] > 0, "current_mrr"].mean()
    proj_ltv = arpa / monthly_churn_rate if pd.notna(arpa) and pd.notna(monthly_churn_rate) and monthly_churn_rate > 0 else np.nan
    channel_rows.append({
        "channel": ch,
        "accounts": int(len(g)),
        "churn_rate": round(g["churn_flag"].mean(), 4),
        "reactivation_rate": round(g.loc[g["churn_flag"], "has_reactivated"].mean(), 4) if churned.shape[0] else 0,
        "avg_realized_ltv": round(g["realized_ltv"].mean(), 2),
        "avg_net_ltv": round(g["net_ltv"].mean(), 2),
        "avg_support_cost": round(g["support_cost_usd"].mean(), 2),
        "avg_current_mrr": round(arpa, 2) if pd.notna(arpa) else 0,
        "projected_ltv": round(proj_ltv, 2) if pd.notna(proj_ltv) else None,
        "projected_net_ltv": round(proj_ltv * ASSUMED_GROSS_MARGIN, 2) if pd.notna(proj_ltv) else None,
        "avg_tenure_months": round(g["tenure_months"].mean(), 2),
    })
channel_df = pd.DataFrame(channel_rows).sort_values("avg_realized_ltv", ascending=False)

# =====================================================================
# 3. LTV & churn by plan tier
# =====================================================================
plan_rows = []
for pt, g in acc.groupby("plan_tier"):
    plan_rows.append({
        "plan_tier": pt,
        "accounts": int(len(g)),
        "churn_rate": round(g["churn_flag"].mean(), 4),
        "avg_realized_ltv": round(g["realized_ltv"].mean(), 2),
        "avg_net_ltv": round(g["net_ltv"].mean(), 2),
        "avg_current_mrr": round(g.loc[g["current_mrr"] > 0, "current_mrr"].mean(), 2)
        if (g["current_mrr"] > 0).any() else 0,
    })
plan_df = pd.DataFrame(plan_rows)

# =====================================================================
# 4. Survival curve (retention %) by month-since-signup, overall + channel
# =====================================================================
max_month = 24

def survival_curve(df):
    out = []
    n0 = len(df)
    for m in range(0, max_month + 1):
        # at risk / retained if tenure_months >= m (i.e. still "alive" at month m)
        still = ((df["tenure_months"] >= m) | (~df["churn_flag"] & (df["tenure_months"] >= m))).sum()
        # correct retained definition: account survived to month m if tenure_months >= m
        retained = (df["tenure_months"] >= m).sum()
        out.append({"month": m, "retained_pct": round(retained / n0, 4) if n0 else 0})
    return out

survival_overall = survival_curve(acc)
survival_by_channel = {ch: survival_curve(g) for ch, g in acc.groupby("referral_source")}

# =====================================================================
# 5. Cohort LTV — signup month cohorts
# =====================================================================
acc["signup_month"] = acc["signup_date"].dt.to_period("M").astype(str)
cohort_rows = []
for cm, g in acc.groupby("signup_month"):
    cohort_rows.append({
        "cohort": cm,
        "accounts": int(len(g)),
        "avg_realized_ltv": round(g["realized_ltv"].mean(), 2),
        "churn_rate": round(g["churn_flag"].mean(), 4),
    })
cohort_df = pd.DataFrame(cohort_rows).sort_values("cohort")

# =====================================================================
# 6. Reactivation value analysis
# =====================================================================
lifecycle_rows = []
for seg, g in acc.groupby("lifecycle_segment"):
    lifecycle_rows.append({
        "segment": seg,
        "accounts": int(len(g)),
        "avg_realized_ltv": round(g["realized_ltv"].mean(), 2),
        "avg_net_ltv": round(g["net_ltv"].mean(), 2),
        "avg_tenure_months": round(g["tenure_months"].mean(), 2),
        "avg_current_mrr": round(g.loc[g["current_mrr"] > 0, "current_mrr"].mean(), 2)
        if (g["current_mrr"] > 0).any() else 0,
    })
lifecycle_df = pd.DataFrame(lifecycle_rows)

# how much currently-active revenue comes from accounts that churned at least
# once and came back (i.e. the tangible payoff of a reactivation program)
won_back_active = acc[(~acc["churn_flag"]) & (acc["churn_event_count"] > 0)]
reactivation_value = {
    "accounts_currently_active_after_winback": int(len(won_back_active)),
    "pct_of_active_base": round(len(won_back_active) / (~acc["churn_flag"]).sum(), 4),
    "current_mrr_from_winback_accounts": round(won_back_active["current_mrr"].sum(), 2),
    "pct_of_current_mrr": round(
        won_back_active["current_mrr"].sum() / acc.loc[~acc["churn_flag"], "current_mrr"].sum(), 4
    ),
}

# reactivation by channel: which channels reactivate best after churn
reactivation_by_channel = []
for ch, g in acc[acc["churn_flag"]].groupby("referral_source"):
    reactivation_by_channel.append({
        "channel": ch,
        "churned_accounts": int(len(g)),
        "reactivated": int(g["has_reactivated"].sum()),
        "reactivation_rate": round(g["has_reactivated"].mean(), 4),
    })
reactivation_by_channel_df = pd.DataFrame(reactivation_by_channel).sort_values("reactivation_rate", ascending=False)

# =====================================================================
# 7. Churn reason breakdown + LTV lost
# =====================================================================
churn_reason = churn.merge(acc[["account_id", "realized_ltv", "referral_source", "plan_tier"]], on="account_id", how="left")
reason_rows = []
for r, g in churn_reason.groupby("reason_code"):
    reason_rows.append({
        "reason_code": r,
        "events": int(len(g)),
        "pct_of_churn_events": round(len(g) / len(churn_reason), 4),
        "avg_account_ltv": round(g["realized_ltv"].mean(), 2),
        "preceding_downgrade_rate": round(g["preceding_downgrade_flag"].mean(), 4),
        "preceding_upgrade_rate": round(g["preceding_upgrade_flag"].mean(), 4),
        "avg_refund_usd": round(g["refund_amount_usd"].mean(), 2),
    })
reason_df = pd.DataFrame(reason_rows).sort_values("events", ascending=False)

# =====================================================================
# 8. Support experience vs churn
# =====================================================================
support_rows = []
for status, g in acc.groupby(acc["churn_flag"].map({True: "churned", False: "active"})):
    support_rows.append({
        "status": status,
        "avg_satisfaction": round(g["avg_satisfaction"].mean(), 2) if g["avg_satisfaction"].notna().any() else None,
        "avg_escalation_rate": round(g["escalation_rate"].mean(), 4) if g["escalation_rate"].notna().any() else None,
        "avg_ticket_count": round(g["ticket_count"].fillna(0).mean(), 2),
        "avg_resolution_hrs": round(g["avg_resolution_hrs"].mean(), 2) if g["avg_resolution_hrs"].notna().any() else None,
    })
support_df = pd.DataFrame(support_rows)

# =====================================================================
# 9. Industry x channel LTV matrix (for exec heatmap)
# =====================================================================
matrix = acc.pivot_table(index="industry", columns="referral_source", values="realized_ltv", aggfunc="mean").round(0)

# =====================================================================
# Assemble dashboard JSON
# =====================================================================
assumptions = {
    "assumed_gross_margin": ASSUMED_GROSS_MARGIN,
    "assumed_support_cost_per_hour_usd": ASSUMED_SUPPORT_COST_PER_HOUR,
    "note": (
        "Dataset has no COGS/hosting/salary fields. Net LTV = realized revenue x "
        "an assumed SaaS-benchmark gross margin, minus two REAL cost signals "
        "present in the data: support resolution hours (costed at an assumed "
        "$/hr) and refund/credit amounts (actual $ from churn_events). Everything "
        "except the margin % and the $/hr rate is measured, not assumed."
    ),
}

dashboard = {
    "kpi": kpi,
    "assumptions": assumptions,
    "channel": channel_df.to_dict(orient="records"),
    "plan_tier": plan_df.to_dict(orient="records"),
    "survival_overall": survival_overall,
    "survival_by_channel": survival_by_channel,
    "cohort": cohort_df.to_dict(orient="records"),
    "lifecycle_segment": lifecycle_df.to_dict(orient="records"),
    "reactivation_value": reactivation_value,
    "reactivation_by_channel": reactivation_by_channel_df.to_dict(orient="records"),
    "churn_reason": reason_df.to_dict(orient="records"),
    "support_vs_churn": support_df.to_dict(orient="records"),
    "industry_channel_matrix": {
        "industries": matrix.index.tolist(),
        "channels": matrix.columns.tolist(),
        "values": matrix.fillna(0).values.tolist(),
    },
}

with open(f"{OUT}/dashboard_data.json", "w") as f:
    json.dump(dashboard, f, indent=2, default=str)

# ------------------------------------------------------------- console ---
print("KPI SUMMARY")
for k, v in kpi.items():
    print(f"  {k}: {v}")
print("\nCHANNEL TABLE")
print(channel_df.to_string(index=False))
print("\nPLAN TIER TABLE")
print(plan_df.to_string(index=False))
print("\nLIFECYCLE SEGMENT (reactivation value)")
print(lifecycle_df.to_string(index=False))
print("\nREACTIVATION VALUE")
for k, v in reactivation_value.items():
    print(f"  {k}: {v}")
print("\nREACTIVATION RATE BY CHANNEL")
print(reactivation_by_channel_df.to_string(index=False))
print("\nCHURN REASON TABLE")
print(reason_df.to_string(index=False))
print("\nSUPPORT VS CHURN")
print(support_df.to_string(index=False))
print("\nASSUMPTIONS (net LTV)")
for k, v in assumptions.items():
    print(f"  {k}: {v}")
print(f"\nWrote {OUT}/dashboard_data.json")
