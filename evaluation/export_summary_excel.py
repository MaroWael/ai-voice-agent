"""
Banque Misr RAG Evaluation Summary Statistics Exporter
=====================================================

This script reads the evaluation results from `reports/rag_evaluation_results.json`
and generates a multi-tab Excel spreadsheet (`reports/rag_evaluation_summary.xlsx`)
containing summary statistics across all categories, verdicts, question types, and overall performance.

Does NOT modify existing code or result files.
"""

import json
from pathlib import Path
import pandas as pd

RESULTS_JSON = Path(__file__).parent.parent / "reports" / "rag_evaluation_results.json"
EXCEL_OUTPUT = Path(__file__).parent.parent / "reports" / "rag_evaluation_summary.xlsx"


def generate_excel_summary():
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"Results file not found at: {RESULTS_JSON}")

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # 1. Overall Key Performance Indicators (KPIs)
    # ---------------------------------------------------------
    total_q = len(df)
    successful_calls = (df["api_status"] == "success").sum()
    failed_calls = (df["api_status"] != "success").sum()
    avg_score = df["score"].mean()
    avg_sim = df["similarity_pct"].mean()
    avg_latency = df[df["api_status"] == "success"]["api_latency_ms"].mean()

    expected_refusals = df[df["expected_refusal"] == True]
    correct_refusals = expected_refusals[expected_refusals["is_refusal"] == True]
    oos_hallucinations = expected_refusals[expected_refusals["is_refusal"] == False]

    kpi_data = {
        "Metric": [
            "Total Questions",
            "Successful API Calls",
            "Failed / Timeout Calls",
            "Average Score (/10)",
            "Average Similarity (%)",
            "Average Latency (ms)",
            "Pass Rate (Score >= 5.0)",
            "Out-of-Scope Questions",
            "Correct OOS Refusals",
            "OOS Hallucinations",
        ],
        "Value": [
            total_q,
            successful_calls,
            failed_calls,
            round(avg_score, 2),
            f"{avg_sim:.1f}%",
            round(avg_latency, 2) if not pd.isna(avg_latency) else 0,
            f"{(df['score'] >= 5.0).mean() * 100:.1f}%",
            len(expected_refusals),
            len(correct_refusals),
            len(oos_hallucinations),
        ],
    }
    df_kpi = pd.DataFrame(kpi_data)

    # ---------------------------------------------------------
    # 2. Category Statistics
    # ---------------------------------------------------------
    category_rows = []
    for cat, group in df.groupby("category"):
        cat_total = len(group)
        cat_avg_score = group["score"].mean()
        cat_min_score = group["score"].min()
        cat_max_score = group["score"].max()
        cat_avg_sim = group["similarity_pct"].mean()
        cat_avg_latency = group["api_latency_ms"].mean()
        pass_count = (group["score"] >= 5.0).sum()
        fail_count = (group["score"] < 5.0).sum()
        pass_rate = (pass_count / cat_total) * 100

        category_rows.append({
            "Category": cat,
            "Total Questions": cat_total,
            "Avg Score (/10)": round(cat_avg_score, 2),
            "Min Score": cat_min_score,
            "Max Score": cat_max_score,
            "Avg Similarity (%)": round(cat_avg_sim, 1),
            "Avg Latency (ms)": round(cat_avg_latency, 1),
            "Passed (>=5.0)": pass_count,
            "Failed (<5.0)": fail_count,
            "Pass Rate (%)": round(pass_rate, 1),
        })
    df_category = pd.DataFrame(category_rows).sort_values(by="Category")

    # ---------------------------------------------------------
    # 3. Verdict Distribution Statistics
    # ---------------------------------------------------------
    verdict_order = ["Excellent", "Good", "Acceptable", "Poor", "Fail"]
    verdict_rows = []
    for v in verdict_order:
        group = df[df["verdict"] == v]
        count = len(group)
        pct = (count / total_q) * 100 if total_q > 0 else 0
        v_avg_score = group["score"].mean() if count > 0 else 0
        v_avg_sim = group["similarity_pct"].mean() if count > 0 else 0

        verdict_rows.append({
            "Verdict": v,
            "Count": count,
            "Percentage (%)": round(pct, 1),
            "Avg Score (/10)": round(v_avg_score, 2),
            "Avg Similarity (%)": round(v_avg_sim, 1),
        })
    df_verdict = pd.DataFrame(verdict_rows)

    # ---------------------------------------------------------
    # 4. Question Type Statistics
    # ---------------------------------------------------------
    type_rows = []
    for q_type, group in df.groupby("type"):
        t_total = len(group)
        t_avg_score = group["score"].mean()
        t_avg_sim = group["similarity_pct"].mean()
        t_pass_rate = (group["score"] >= 5.0).mean() * 100

        type_rows.append({
            "Question Type": q_type,
            "Total Questions": t_total,
            "Avg Score (/10)": round(t_avg_score, 2),
            "Avg Similarity (%)": round(t_avg_sim, 1),
            "Pass Rate (%)": round(t_pass_rate, 1),
        })
    df_type = pd.DataFrame(type_rows).sort_values(by="Question Type")

    # ---------------------------------------------------------
    # 5. Full Detailed Export
    # ---------------------------------------------------------
    df_detail = df[[
        "id", "category", "type", "question", "ground_truth",
        "retrieved_answer", "api_status", "api_latency_ms",
        "similarity_pct", "score", "verdict", "is_refusal",
        "expected_refusal", "explanation"
    ]].copy()
    df_detail.rename(columns={
        "id": "ID",
        "category": "Category",
        "type": "Type",
        "question": "Question",
        "ground_truth": "Ground Truth",
        "retrieved_answer": "Retrieved Answer",
        "api_status": "API Status",
        "api_latency_ms": "Latency (ms)",
        "similarity_pct": "Similarity (%)",
        "score": "Score",
        "verdict": "Verdict",
        "is_refusal": "Is Refusal",
        "expected_refusal": "Expected Refusal",
        "explanation": "Explanation",
    }, inplace=True)

    # ---------------------------------------------------------
    # Write to Excel with openpyxl
    # ---------------------------------------------------------
    EXCEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
        df_kpi.to_excel(writer, sheet_name="KPI Summary", index=False)
        df_category.to_excel(writer, sheet_name="Category Stats", index=False)
        df_verdict.to_excel(writer, sheet_name="Verdict Distribution", index=False)
        df_type.to_excel(writer, sheet_name="Question Type Stats", index=False)
        df_detail.to_excel(writer, sheet_name="All Evaluation Details", index=False)

    print(f"Summary statistics Excel sheet successfully generated at:\n{EXCEL_OUTPUT.resolve()}")


if __name__ == "__main__":
    generate_excel_summary()
