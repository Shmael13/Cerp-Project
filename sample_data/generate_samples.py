"""
Generate rich sample Excel files that showcase every chart type in CERP Visualizer.
Run once: python generate_samples.py
"""
import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)
OUT = Path(__file__).parent


# ── 1. SaaS Company Metrics ───────────────────────────────────────────────────
# Monthly data: good for Line, Area, Bar, KPI, Distribution, Scatter, Bullet
months = pd.date_range("2022-01", periods=36, freq="MS")
regions = ["EMEA", "Americas", "APAC", "LATAM"]
saas_rows = []
for m in months:
    base = 1 + (m.year - 2022) * 0.25 + (m.month - 1) * 0.02
    for r in regions:
        mult = {"EMEA": 1.2, "Americas": 1.8, "APAC": 0.9, "LATAM": 0.5}[r]
        mrr      = round(rng.normal(120_000 * mult * base, 8_000), 0)
        churn    = round(rng.normal(mrr * 0.04, mrr * 0.005), 0)
        new_mrr  = round(rng.normal(mrr * 0.08, mrr * 0.01), 0)
        target   = round(mrr * rng.uniform(0.95, 1.10), 0)
        cac      = round(rng.normal(450 * mult, 40), 2)
        ltv      = round(rng.normal(3200 * mult * base, 200), 2)
        nps      = round(rng.normal(42, 8), 1)
        support  = int(rng.integers(20, 90))
        saas_rows.append({
            "Date": m, "Region": r,
            "MRR ($)": max(mrr, 0), "Churned MRR ($)": -abs(churn),
            "New MRR ($)": new_mrr, "MRR Target ($)": target,
            "CAC ($)": cac, "LTV ($)": ltv, "NPS": nps,
            "Support Tickets": support,
        })

saas_df = pd.DataFrame(saas_rows)

# Monthly totals sheet
saas_monthly = saas_df.groupby("Date").agg({
    "MRR ($)": "sum", "Churned MRR ($)": "sum",
    "New MRR ($)": "sum", "MRR Target ($)": "sum",
    "Support Tickets": "sum", "NPS": "mean", "CAC ($)": "mean", "LTV ($)": "mean",
}).reset_index()


# ── 2. Retail Sales ───────────────────────────────────────────────────────────
# Good for Bar, Treemap, Pie/Donut, Heatmap, Funnel, Waterfall
categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Beauty", "Toys", "Books", "Food"]
sub_cats = {
    "Electronics": ["Phones", "Laptops", "Tablets", "Accessories", "TVs"],
    "Clothing":    ["Men's", "Women's", "Kids'", "Footwear", "Accessories"],
    "Home & Garden": ["Furniture", "Decor", "Tools", "Plants", "Bedding"],
    "Sports":      ["Gym", "Outdoor", "Team Sports", "Water Sports", "Cycling"],
    "Beauty":      ["Skincare", "Makeup", "Haircare", "Fragrance", "Wellness"],
    "Toys":        ["Action", "Educational", "Outdoor", "Board Games", "Arts"],
    "Books":       ["Fiction", "Non-fiction", "Children's", "Academic", "Comics"],
    "Food":        ["Snacks", "Beverages", "Organic", "Frozen", "Bakery"],
}
retail_rows = []
for cat in categories:
    for sub in sub_cats[cat]:
        for month in pd.date_range("2023-01", periods=12, freq="MS"):
            seasonal = 1 + 0.3 * np.sin((month.month - 3) * np.pi / 6)
            cat_mult = {"Electronics": 3.2, "Clothing": 1.8, "Home & Garden": 1.2,
                        "Sports": 1.1, "Beauty": 1.4, "Toys": 0.9, "Books": 0.7, "Food": 2.1}[cat]
            revenue  = round(rng.normal(15_000 * cat_mult * seasonal, 2_000), 0)
            cogs     = round(revenue * rng.uniform(0.45, 0.65), 0)
            returns  = round(revenue * rng.uniform(0.02, 0.12), 0)
            units    = int(revenue / rng.uniform(18, 120))
            retail_rows.append({
                "Date": month, "Category": cat, "Sub-Category": sub,
                "Revenue ($)": max(revenue, 0), "COGS ($)": cogs,
                "Gross Profit ($)": max(revenue - cogs, 0),
                "Returns ($)": returns, "Units Sold": units,
            })

retail_df = pd.DataFrame(retail_rows)

# Category summary (no dates) — best for Treemap, Pie, Bar
retail_cat = retail_df.groupby(["Category", "Sub-Category"]).agg({
    "Revenue ($)": "sum", "COGS ($)": "sum",
    "Gross Profit ($)": "sum", "Units Sold": "sum", "Returns ($)": "sum",
}).reset_index()
retail_cat["Margin (%)"] = (retail_cat["Gross Profit ($)"] / retail_cat["Revenue ($)"] * 100).round(1)


# ── 3. Pipeline & Funnel ─────────────────────────────────────────────────────
# Good for Funnel, Sankey, Waterfall, Bar
stages = ["Leads", "Qualified", "Demo", "Proposal", "Negotiation", "Closed Won"]
quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"]
funnel_rows = []
for q in quarters:
    leads = int(rng.integers(800, 1400))
    conv  = [1.0, rng.uniform(0.35, 0.50), rng.uniform(0.18, 0.28),
             rng.uniform(0.10, 0.17), rng.uniform(0.06, 0.11), rng.uniform(0.03, 0.07)]
    for stage, rate in zip(stages, conv):
        value = int(leads * rate)
        deal_val = round(value * rng.uniform(4_500, 9_500), 0)
        funnel_rows.append({
            "Quarter": q, "Stage": stage,
            "Count": value, "Pipeline Value ($)": deal_val,
            "Conversion Rate (%)": round(rate * 100, 1),
        })

funnel_df = pd.DataFrame(funnel_rows)

# P&L Waterfall — signed values per line item
pnl_rows = [
    {"Line Item": "Revenue",           "Amount ($)":  4_820_000, "Category": "Income"},
    {"Line Item": "COGS",              "Amount ($)": -1_930_000, "Category": "Cost"},
    {"Line Item": "Gross Profit",      "Amount ($)":  2_890_000, "Category": "Subtotal"},
    {"Line Item": "Sales & Marketing", "Amount ($)":   -780_000, "Category": "OpEx"},
    {"Line Item": "R&D",               "Amount ($)":   -620_000, "Category": "OpEx"},
    {"Line Item": "G&A",               "Amount ($)":   -310_000, "Category": "OpEx"},
    {"Line Item": "EBITDA",            "Amount ($)":  1_180_000, "Category": "Subtotal"},
    {"Line Item": "Depreciation",      "Amount ($)":   -145_000, "Category": "Non-cash"},
    {"Line Item": "Interest",          "Amount ($)":    -68_000, "Category": "Finance"},
    {"Line Item": "Tax",               "Amount ($)":   -241_000, "Category": "Tax"},
    {"Line Item": "Net Income",        "Amount ($)":    726_000, "Category": "Net"},
]
pnl_df = pd.DataFrame(pnl_rows)


# ── 4. HR & Workforce ────────────────────────────────────────────────────────
# Good for Distribution, Heatmap, Scatter, Tornado
departments  = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Product", "Operations"]
locations    = ["London", "New York", "Singapore", "Berlin", "Toronto"]
hr_rows = []
for _ in range(420):
    dept     = rng.choice(departments)
    loc      = rng.choice(locations)
    tenure   = round(float(rng.exponential(3.5)), 1)
    salary   = round(rng.normal(
        {"Engineering": 95, "Sales": 80, "Marketing": 72, "Finance": 85,
         "HR": 62, "Product": 90, "Operations": 65}[dept] * 1000, 12_000
    ), 0)
    perf     = round(float(rng.normal(3.4, 0.7)), 1)
    age      = int(rng.integers(22, 60))
    hrs_wk   = round(float(rng.normal(42, 5)), 1)
    training = int(rng.integers(0, 80))
    hr_rows.append({
        "Department": dept, "Location": loc,
        "Tenure (years)": max(tenure, 0.1), "Salary ($k)": max(salary / 1000, 35),
        "Performance Score": min(max(perf, 1.0), 5.0), "Age": age,
        "Hours/Week": max(hrs_wk, 30.0), "Training Hours": training,
    })

hr_df = pd.DataFrame(hr_rows)


# ── 5. Operations & Logistics ────────────────────────────────────────────────
# Good for Heatmap (warehouse × day), Scatter, Bullet, KPI
warehouses = ["WH-North", "WH-South", "WH-East", "WH-West", "WH-Central"]
days = pd.date_range("2023-01-02", periods=260, freq="B")  # business days
ops_rows = []
for d in days:
    for wh in warehouses:
        wh_base = {"WH-North": 920, "WH-South": 760, "WH-East": 1100,
                   "WH-West": 830, "WH-Central": 1350}[wh]
        orders     = int(rng.normal(wh_base, wh_base * 0.08))
        shipped    = int(orders * rng.uniform(0.88, 0.99))
        on_time    = int(shipped * rng.uniform(0.90, 0.99))
        cost       = round(orders * rng.uniform(2.8, 4.2), 2)
        target_ord = int(wh_base * 1.05)
        ops_rows.append({
            "Date": d, "Warehouse": wh,
            "Orders": orders, "Shipped": shipped, "On-Time": on_time,
            "Shipping Cost ($)": cost, "Orders Target": target_ord,
            "Fill Rate (%)": round(shipped / orders * 100, 1),
            "On-Time Rate (%)": round(on_time / max(shipped, 1) * 100, 1),
        })

ops_df = pd.DataFrame(ops_rows)

# Aggregated by warehouse for bullet/KPI
ops_summary = ops_df.groupby("Warehouse").agg({
    "Orders": "sum", "Shipped": "sum", "On-Time": "sum",
    "Shipping Cost ($)": "sum", "Orders Target": "sum",
    "Fill Rate (%)": "mean", "On-Time Rate (%)": "mean",
}).reset_index()


# ── Write to Excel ────────────────────────────────────────────────────────────
files = {
    "saas_metrics.xlsx": {
        "Monthly by Region": saas_df,
        "Monthly Totals": saas_monthly,
    },
    "retail_sales.xlsx": {
        "Daily Sales": retail_df,
        "Category Summary": retail_cat,
    },
    "sales_pipeline.xlsx": {
        "Funnel by Quarter": funnel_df,
        "P&L Waterfall": pnl_df,
    },
    "hr_workforce.xlsx": {
        "Employees": hr_df,
    },
    "operations_logistics.xlsx": {
        "Daily Operations": ops_df,
        "Warehouse Summary": ops_summary,
    },
}

for fname, sheets in files.items():
    path = OUT / fname
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for sheet_name, df in sheets.items():
            df.to_excel(w, sheet_name=sheet_name, index=False)
    rows = sum(len(d) for d in sheets.values())
    print(f"  {fname}  ({rows:,} rows across {len(sheets)} sheet(s))")

print("\nDone. All files written to", OUT)
