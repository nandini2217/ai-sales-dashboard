"""
insight_generator.py
Generates a one-off AI business insight report from the cleaned sales data
and saves it to output/insight_report.txt. For the scheduled version that
also emails the report, see autoreport.py.
"""
import os

from dotenv import load_dotenv
from groq import Groq

from metrics import compute_metrics, format_metrics_summary, generate_ai_report, load_data

load_dotenv()


def main():
    df = load_data()
    m = compute_metrics(df)
    print(format_metrics_summary(m))

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("\nGenerating AI insights...\n")
    report = generate_ai_report(client, m)

    print("=== AI GENERATED BUSINESS INSIGHT REPORT ===\n")
    print(report)

    with open("output/insight_report.txt", "w") as f:
        f.write("AI GENERATED BUSINESS INSIGHT REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    print("\nReport saved to output/insight_report.txt!")


if __name__ == "__main__":
    main()
