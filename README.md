# RavenStack — LTV & Reactivation Analysis

**Live dashboard: https://ugrodrigo.github.io/ltv-analysis/**

An end-to-end customer lifetime value analysis built as a working sample against the scope of a
Staff Marketing Analyst (Global Reactivations) role at a subscription-based B2C business —
KPI framing, channel/cohort LTV, retention curves, margin-adjusted net LTV, a predictive LTV
model, and a reactivation-timing / outreach-prioritization model.

The dataset is a synthetic B2B SaaS dataset (RavenStack), not the hiring company's real data —
see [Credits](#credits). The point of the project is the analysis, not the specific numbers.

## What's in the dashboard

- **Executive KPIs** — accounts, churn rate, realized/net LTV, ARR, and how much of the active
  book is made up of previously-churned-and-won-back accounts (71%).
- **Channel & cohort LTV** — realized vs. projected (ARPA ÷ churn-rate) LTV by acquisition channel,
  survival curves, and an industry × channel value heatmap.
- **Margin-adjusted (net) LTV** — gross LTV minus real cost signals in the data (support
  cost-to-serve, refunds) and a disclosed gross-margin assumption, since the dataset has no
  COGS/hosting fields.
- **Predictive LTV** — a model trained on an account's first 90 days (plan, seats, deal size,
  early usage/support signals — nothing that leaks the outcome) to predict eventual net LTV,
  evaluated with 5-fold cross-validation and scored against the newest accounts.
- **Reactivation timing & outreach** — a pivoted version of a "reactivation propensity" model.
  87% of churn events resolve into a new subscription within weeks regardless of segment, which
  makes "will they come back" undecidable from this data — so the model instead predicts *how
  long* reactivation takes, and flags accounts running past their expected return window,
  combined with a disclosed rule-based (not learned) offer mapping.

Every assumption used in the analysis (gross margin %, support cost/hr, offer-mapping rules) is
disclosed in the dashboard itself, not buried in a footnote — including a data-quality finding
that `accounts.churn_flag` doesn't reliably match the underlying subscription/churn history, so
it was excluded from every model.

## Repo structure

```
dataset/                       RavenStack synthetic SaaS dataset (5 CSVs) + its own README
analysis/
  ltv_analysis.py              Realized / projected / net LTV, cohorts, channels, reactivation value
  predictive_ltv.py            90-day early-signal LTV prediction model
  reactivation_propensity.py   Reactivation-timing model + outreach prioritization
  output/                      Generated JSON + the dashboard HTML (build artifacts)
docs/
  index.html                   Published copy of the dashboard (GitHub Pages source)
.github/workflows/pages.yml    Deploys docs/ to GitHub Pages on push to main
requirements.txt
LICENSE
```

## Reproducing the analysis

```bash
pip install -r requirements.txt
cd analysis
python ltv_analysis.py             # -> output/dashboard_data.json
python predictive_ltv.py           # -> output/predictive_ltv_data.json
python reactivation_propensity.py  # -> output/reactivation_propensity_data.json
```

The three scripts are independent (each loads its own data from `dataset/`) and can be run in
any order. `output/ltv_dashboard.html` embeds the JSON outputs directly and is self-contained —
open it in a browser with no server needed. After regenerating the JSON, copy it to
`docs/index.html` to update the published site.

## Methodology notes

- **Realized LTV** = Σ(mrr_amount × active months) per account, through the dataset's last
  observed date (2024-12-31).
- **Projected LTV** = ARPA ÷ monthly churn rate — the standard SaaS shorthand, not a cohort-fit
  survival model.
- **Net LTV** = realized LTV × an assumed 78% gross margin, minus actual refund/credit amounts
  and support cost-to-serve (ticket hours × an assumed $35/hr). The margin % and hourly rate are
  disclosed industry-benchmark assumptions; everything else is measured from the data.
- **Predictive LTV** uses only signals knowable in an account's first 90 days and is validated
  with 5-fold cross-validation rather than a single train/test split, which is too noisy to trust
  at this sample size (500 accounts).
- **Reactivation timing** is derived entirely from subscription start/end dates and churn events
  — not from `accounts.churn_flag`, which was found to be statistically independent of the actual
  transactional history during this analysis.

## Credits

Dataset: **RavenStack: Synthetic SaaS Dataset**, created by **River @ Rivalytics**, fully
synthetic and distributed under an MIT-like license requiring attribution. See
[`dataset/README.md`](dataset/README.md) for the full schema and license terms. The dataset is
used here for portfolio/educational purposes only.

## License

Code in this repository (`analysis/*.py`, `docs/index.html`) is MIT licensed — see
[LICENSE](LICENSE). The dataset in `dataset/` carries its own separate license and attribution
requirement (above) and is not covered by this repo's LICENSE.
