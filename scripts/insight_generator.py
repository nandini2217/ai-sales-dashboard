import os
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

#Load API key from .env file
load_dotenv()

#connect to Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#Load the cleaned dataset
df=pd.read_csv('data/superstore_cleaned.csv')

#Calculate business metrics
total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
profit_margin = (total_profit / total_sales)*100
total_orders = df['Order ID'].nunique()
total_customers = df['Customer Name'].nunique()

#Sales by region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
top_region = region_sales.index[0]
top_region_sales = region_sales.iloc[0]

#Loss making sub-categories
loss_subs = df.groupby('Sub-Category')['Profit'].sum()
loss_subs = loss_subs[loss_subs < 0].sort_values()
loss_list = ', '.join(loss_subs.index.tolist())

#Best and worst category
category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
best_category = category_profit.index[0]
worst_category = category_profit.index[-1]

#Monthly trend - best month
monthly_sales = df.groupby(['Order Year', 'Order Month'])['Sales'].sum()
best_period = monthly_sales.idxmax()

print("=== Business Metrics Calculated ===")
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Profit Margin: {profit_margin:.2f}%")
print(f"Total Orders: {total_orders}")
print(f"Top Region: {top_region}")
print(f"Loss Making Sub-Categories: {loss_list}")
print(f"Best Category: {best_category}")
print(f"Worst Category: {worst_category}")
print(f"Best Period: {best_period[0]}-{best_period[1]:02d}")
print("n/Metrics ready ! Sending to AI...\n")

# Build the prompt with real business data
prompt = f"""
You are a senior business analyst. Based on the following real sales data, 
write a professional business insight report with clear recommendations.

BUSINESS METRICS:
- Total Sales: ${total_sales:,.2f}
- Total Profit: ${total_profit:,.2f}
- Profit Margin: {profit_margin:.2f}%
- Total Orders: {total_orders:,}
- Total Customers: {total_customers:,}
- Top Performing Region: {top_region} (${top_region_sales:,.2f} in sales)
- Best Category: {best_category}
- Worst Category: {worst_category}
- Loss Making Sub-Categories: {loss_list}
- Best Sales Period: Year {best_period[0]}, Month {best_period[1]}

Write a report with these 4 sections:
1. Executive Summary (2-3 sentences overview)
2. Key Findings (3-4 bullet points of most important insights)
3. Risk Areas (focus on loss-making sub-categories and what it means)
4. Recommendations (3 clear, actionable steps management should take)

Keep the tone professional, concise and data-driven.
"""

# Send to Groq AI
print("Generating AI insights...\n")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,
    max_tokens=1000
)

# Extract the AI response
insight_report = response.choices[0].message.content

print("=== AI GENERATED BUSINESS INSIGHT REPORT ===\n")
print(insight_report)

# Save the report to output folder
with open('output/insight_report.txt', 'w') as f:
    f.write("AI GENERATED BUSINESS INSIGHT REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(insight_report)

print("\n\nReport saved to output/insight_report.txt!")