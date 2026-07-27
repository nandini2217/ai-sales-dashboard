"""
metrics.py
Shared business-metric calculations and AI prompt logic for the sales
insight pipeline. Both insight_generator.py (console + file output) and
autoreport.py (adds scheduled email delivery) import from here so the
metric definitions only live in one place.
"""
import pandas as pd


def load_data(path: str = "data/superstore_cleaned.csv") -> pd.DataFrame:
    """Load the cleaned Superstore dataset produced by
    data/notebooks/01_data_cleaning.ipynb."""
    return pd.read_csv(path)


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute the core business metrics that drive the AI insight prompt.

    Returns a plain dict so it's easy to test, log, or template into an
    email without touching pandas objects downstream.
    """
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    profit_margin = (total_profit / total_sales) * 100
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer Name"].nunique()

    region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    top_region = region_sales.index[0]
    top_region_sales = region_sales.iloc[0]

    loss_subs = df.groupby("Sub-Category")["Profit"].sum()
    loss_subs = loss_subs[loss_subs < 0].sort_values()
    loss_list = ", ".join(loss_subs.index.tolist()) if not loss_subs.empty else "None"

    cat_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
    best_category = cat_profit.index[0]
    worst_category = cat_profit.index[-1]

    monthly_sales = df.groupby(["Order Year", "Order Month"])["Sales"].sum()
    best_period = monthly_sales.idxmax()

    # Discount vs. profitability: the single strongest driver of loss in
    # this dataset (see NOTES / README for the full finding).
    df = df.copy()
    df["Profit Margin"] = df["Profit"] / df["Sales"]
    discount_margin_corr = df["Discount"].corr(df["Profit Margin"])
    heavy_discount_orders = df[df["Discount"] >= 0.3]
    heavy_discount_loss = heavy_discount_orders["Profit"].sum()
    total_loss = df.loc[df["Profit"] < 0, "Profit"].sum()
    heavy_discount_share_of_loss = (
        heavy_discount_loss / total_loss if total_loss != 0 else 0.0
    )

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "profit_margin": profit_margin,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "top_region": top_region,
        "top_region_sales": top_region_sales,
        "loss_list": loss_list,
        "best_category": best_category,
        "worst_category": worst_category,
        "best_period": best_period,
        "discount_margin_corr": discount_margin_corr,
        "heavy_discount_loss": heavy_discount_loss,
        "heavy_discount_share_of_loss": heavy_discount_share_of_loss,
    }


def format_metrics_summary(m: dict) -> str:
    """Human-readable console summary of the computed metrics."""
    lines = [
        "=== Business Metrics Calculated ===",
        f"Total Sales: ${m['total_sales']:,.2f}",
        f"Total Profit: ${m['total_profit']:,.2f}",
        f"Profit Margin: {m['profit_margin']:.2f}%",
        f"Total Orders: {m['total_orders']:,}",
        f"Total Customers: {m['total_customers']:,}",
        f"Top Region: {m['top_region']} (${m['top_region_sales']:,.2f})",
        f"Best Category: {m['best_category']}",
        f"Worst Category: {m['worst_category']}",
        f"Loss Making Sub-Categories: {m['loss_list']}",
        f"Best Period: {m['best_period'][0]}-{int(m['best_period'][1]):02d}",
        f"Discount-vs-Margin Correlation: {m['discount_margin_corr']:.2f}",
        f"Loss from Orders Discounted >=30%: ${m['heavy_discount_loss']:,.2f} "
        f"({m['heavy_discount_share_of_loss']:.0%} of all losses)",
    ]
    return "\n".join(lines)


def build_prompt(m: dict) -> str:
    """Build the LLM prompt from computed metrics. Kept in one place so the
    prompt and the numbers it's fed can't drift out of sync between scripts."""
    return f"""
You are a senior business analyst. Based on the following real sales data,
write a professional business insight report with clear recommendations.

BUSINESS METRICS:
- Total Sales: ${m['total_sales']:,.2f}
- Total Profit: ${m['total_profit']:,.2f}
- Profit Margin: {m['profit_margin']:.2f}%
- Total Orders: {m['total_orders']:,}
- Total Customers: {m['total_customers']:,}
- Top Performing Region: {m['top_region']} (${m['top_region_sales']:,.2f} in sales)
- Best Category: {m['best_category']}
- Worst Category: {m['worst_category']}
- Loss Making Sub-Categories: {m['loss_list']}
- Best Sales Period: Year {m['best_period'][0]}, Month {int(m['best_period'][1])}
- Correlation between Discount and Profit Margin: {m['discount_margin_corr']:.2f}
- Losses from orders discounted 30%+: ${m['heavy_discount_loss']:,.2f} ({m['heavy_discount_share_of_loss']:.0%} of total losses)

Write a report with these 4 sections:
1. Executive Summary (2-3 sentences overview)
2. Key Findings (3-4 bullet points of most important insights)
3. Risk Areas (focus on loss-making sub-categories and the discount pattern)
4. Recommendations (3 clear, actionable steps management should take)

Keep the tone professional, concise and data-driven.
"""


def generate_ai_report(client, m: dict) -> str:
    """Call the LLM with the metrics-driven prompt and return the report text."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": build_prompt(m)}],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content
