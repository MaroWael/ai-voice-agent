"""
Banque Misr RAG Evaluation Pipeline
=====================================

Step 1  - 50-question benchmark dataset (grounded in actual knowledge base scope)
Step 2  - Ground-truth answers (reference answers derived from known KB content)
Step 3  - Call the LIVE FastAPI RAG endpoint for every question
Step 4  - LLM-as-a-Judge evaluation (Groq llama-3.3-70b-versatile)
Step 5  - Markdown report with statistics derived from ACTUAL results

CRITICAL:
  - Retrieved answers come EXCLUSIVELY from the real RAG API.
  - No simulated retrieval. No fabricated answers. No estimated scores.
  - Every metric is calculated from actual API responses.

Usage:
    python -m evaluation.rag_evaluator

Environment requirements:
  - FastAPI RAG server running at http://localhost:8000
  - .env configured (GROQ_API_KEY, QDRANT_HOST, etc.)
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# -- Environment ---------------------------------------------------------------
load_dotenv(Path(__file__).parent.parent / ".env")

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# -- Constants -----------------------------------------------------------------
RAG_API_BASE  = "http://localhost:8000"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
REPORT_PATH   = Path(__file__).parent.parent / "reports" / "rag_evaluation_report.md"
RESULTS_JSON  = Path(__file__).parent.parent / "reports" / "rag_evaluation_results.json"

# ==============================================================================
# STEP 1 & 2 -- 50-Question Benchmark Dataset with Ground Truth
# ==============================================================================
# All questions are grounded in the actual data files in the /data directory.
# Coverage: credit cards (fees, benefits, limits, installments, international usage).
# Out-of-scope questions (Q48-Q50) test the refusal capability.
# ==============================================================================

BENCHMARK_QUESTIONS = [
    # -- CREDIT CARD FEES (Direct) --
    {
        "id": "Q01", "category": "Card Fees", "type": "direct",
        "question": "What are the issuance and renewal fees for the Classic Credit Card?",
        "ground_truth": "The Classic Credit Card has an issuance fee of EGP 250 and a renewal fee of EGP 250.",
    },
    {
        "id": "Q02", "category": "Card Fees", "type": "direct",
        "question": "What is the issuance fee for the Platinum Visa or MasterCard credit card?",
        "ground_truth": "The issuance fee for the Platinum Visa or MasterCard Credit Card is EGP 500.",
    },
    {
        "id": "Q03", "category": "Card Fees", "type": "direct",
        "question": "What is the renewal fee for the Titanium Credit Card?",
        "ground_truth": "The renewal fee for the Titanium Credit Card is EGP 750.",
    },
    {
        "id": "Q04", "category": "Card Fees", "type": "direct",
        "question": "What is the penalty for exceeding the credit limit on Banque Misr credit cards?",
        "ground_truth": "The penalty for exceeding the credit limit on Banque Misr credit cards is EGP 75.",
    },
    {
        "id": "Q05", "category": "Card Fees", "type": "direct",
        "question": "What is the monthly interest rate on Banque Misr credit cards?",
        "ground_truth": "The interest rate on Banque Misr credit cards is 4% per month.",
    },
    {
        "id": "Q06", "category": "Card Fees", "type": "direct",
        "question": "How much is the late payment penalty for Banque Misr credit cards?",
        "ground_truth": "The late payment penalty (penalty for delay) on Banque Misr credit cards is EGP 75.",
    },
    {
        "id": "Q07", "category": "Card Fees", "type": "direct",
        "question": "What is the fee for replacing a lost or damaged Classic Credit Card?",
        "ground_truth": "The reissuance fee for a lost or damaged Classic Credit Card is EGP 100.",
    },
    {
        "id": "Q08", "category": "Card Fees", "type": "comparative",
        "question": "What is the supplementary card fee for the Classic Credit Card versus the Platinum Credit Card?",
        "ground_truth": "For the Classic Credit Card, the supplementary card issuance and renewal fee is EGP 100. For the Platinum Credit Card, it is EGP 300.",
    },
    # -- CREDIT CARD BENEFITS (Direct) --
    {
        "id": "Q09", "category": "Card Benefits", "type": "direct",
        "question": "What is the grace period for the Classic Credit Card?",
        "ground_truth": "The Classic Credit Card offers the longest grace period of up to 56 days.",
    },
    {
        "id": "Q10", "category": "Card Benefits", "type": "direct",
        "question": "What is the minimum payment percentage for the Classic Credit Card?",
        "ground_truth": "The Classic Credit Card offers the lowest payment limit at 5% of the monthly usages.",
    },
    {
        "id": "Q11", "category": "Card Benefits", "type": "direct",
        "question": "Does the Platinum Visa Credit Card offer airport lounge access?",
        "ground_truth": (
            "Yes. The Platinum Visa Credit Card offers 6 free accesses to airport lounges worldwide "
            "via the Visa Airport Companion App, provided an international transaction of at least "
            "USD 5 (or EGP equivalent) via POS or e-commerce is made within 90 days prior to travel."
        ),
    },
    {
        "id": "Q12", "category": "Card Benefits", "type": "direct",
        "question": "What rewards points does the Platinum Credit Card offer?",
        "ground_truth": (
            "Platinum Credit Card holders earn 3 points per EGP spent locally and internationally, "
            "and receive 25,000 welcome points upon spending EGP 2,500 within 3 months from the issue date."
        ),
    },
    {
        "id": "Q13", "category": "Card Benefits", "type": "direct",
        "question": "What is the contactless payment limit without PIN inside Egypt for Banque Misr credit cards?",
        "ground_truth": (
            "The contactless purchase limit without PIN inside Egypt is EGP 600 per transaction "
            "with a maximum of 15 consecutive transactions."
        ),
    },
    {
        "id": "Q14", "category": "Card Benefits", "type": "direct",
        "question": "What interest-free installment period is available through Banque Misr partner merchants?",
        "ground_truth": "Customers can enjoy interest-free installments of up to 60 months for purchases at partnered Banque Misr merchants.",
    },
    {
        "id": "Q15", "category": "Card Benefits", "type": "direct",
        "question": "Does Banque Misr send SMS notifications after each credit card transaction?",
        "ground_truth": "Yes, Banque Misr credit cards offer a free SMS service after each card transaction.",
    },
    # -- CREDIT CARD USAGE LIMITS (Direct) --
    {
        "id": "Q16", "category": "Card Limits", "type": "direct",
        "question": "What is the daily cash withdrawal limit from local ATMs for Banque Misr credit cards?",
        "ground_truth": "The local cash withdrawal limit via ATM for Banque Misr credit cards is EGP 30,000 daily.",
    },
    {
        "id": "Q17", "category": "Card Limits", "type": "direct",
        "question": "What is the monthly international purchase limit for the Classic Credit Card if I have not informed the bank of travel?",
        "ground_truth": "If the customer has not informed the bank of travel, the international monthly purchase limit for the Classic Credit Card is EGP 10,000 per month with a limit of 8 times daily.",
    },
    {
        "id": "Q18", "category": "Card Limits", "type": "direct",
        "question": "What is the maximum online purchase limit inside Egypt per month for the Platinum Credit Card?",
        "ground_truth": (
            "The limit of online purchasing inside Egypt for the Platinum Credit Card is within the card credit limits "
            "with a maximum of EGP 400,000 per month and a limit of 12 times daily."
        ),
    },
    {
        "id": "Q19", "category": "Card Limits", "type": "direct",
        "question": "How much can I withdraw at a Banque Misr branch POS using my credit card?",
        "ground_truth": "The POS cash withdrawal limit from BM branches is a maximum of EGP 250,000 daily.",
    },
    {
        "id": "Q20", "category": "Card Limits", "type": "direct",
        "question": "What is the international cash withdrawal limit if I have notified the bank of my travel?",
        "ground_truth": "If the customer has informed the bank of travel, the international cash withdrawal limit is EGP 3,000 per month.",
    },
    # -- CREDIT LIMITS --
    {
        "id": "Q21", "category": "Credit Limits", "type": "direct",
        "question": "What is the credit limit range for the Classic Credit Card?",
        "ground_truth": "The Classic Credit Card offers a credit limit starting from EGP 2,000 up to less than EGP 3,000.",
    },
    {
        "id": "Q22", "category": "Credit Limits", "type": "direct",
        "question": "What is the credit limit range for the Platinum Visa or MasterCard credit card?",
        "ground_truth": "The Platinum Visa or MasterCard Credit Card offers a credit limit starting from EGP 25,000 up to less than EGP 100,000.",
    },
    {
        "id": "Q23", "category": "Credit Limits", "type": "comparative",
        "question": "What is the credit limit difference between the Classic Credit Card and the Platinum Credit Card?",
        "ground_truth": (
            "The Classic Credit Card has a credit limit from EGP 2,000 to less than EGP 3,000. "
            "The Platinum Credit Card has a credit limit from EGP 25,000 to less than EGP 100,000. "
            "The Platinum Card offers significantly higher limits."
        ),
    },
    # -- INSTALLMENTS --
    {
        "id": "Q24", "category": "Installments", "type": "direct",
        "question": "What is the maximum installment tenor available for credit card purchases and cash withdrawals?",
        "ground_truth": "Installment payment for card purchase transactions and cash withdrawals is available for up to 36 months with a special interest rate.",
    },
    {
        "id": "Q25", "category": "Installments", "type": "direct",
        "question": "What is the monthly interest rate for a 12-month installment plan on the Classic Credit Card?",
        "ground_truth": "For a 12-month installment tenor on the Classic Credit Card, the interest rate is 2.73% per month.",
    },
    {
        "id": "Q26", "category": "Installments", "type": "direct",
        "question": "What are the early repayment fees if I pay off my credit card installment early?",
        "ground_truth": "Early repayment fees of 4% are applied if the customer pays off the value of their installment early.",
    },
    {
        "id": "Q27", "category": "Installments", "type": "direct",
        "question": "What is the interest rate for a 6-month installment plan on Banque Misr credit cards?",
        "ground_truth": "For a 6-month installment tenor, the monthly interest rate is 2.77%.",
    },
    {
        "id": "Q28", "category": "Installments", "type": "direct",
        "question": "What is the interest rate for a 36-month installment plan on Banque Misr credit cards?",
        "ground_truth": "For a 36-month installment tenor, the monthly interest rate is 2.56%.",
    },
    # -- CARD COMPARISON (Comparative) --
    {
        "id": "Q29", "category": "Card Comparison", "type": "comparative",
        "question": "What is the difference in issuance fees between the Classic Credit Card and the Platinum Credit Card?",
        "ground_truth": "The Classic Credit Card issuance fee is EGP 250, while the Platinum Credit Card issuance fee is EGP 500. The difference is EGP 250.",
    },
    {
        "id": "Q30", "category": "Card Comparison", "type": "comparative",
        "question": "Which credit card offers purchase protection insurance: the Classic or the Platinum Visa?",
        "ground_truth": (
            "The Platinum Visa Credit Card offers purchase protection service, protecting purchases against "
            "theft and damage for up to 365 days from the purchase date. The Classic Credit Card does not offer this benefit."
        ),
    },
    {
        "id": "Q31", "category": "Card Comparison", "type": "comparative",
        "question": "Does the Classic Credit Card offer airport lounge access like the Platinum Card?",
        "ground_truth": "No. Airport lounge access is a benefit of the Platinum Credit Card (and other premium cards), not the Classic Credit Card.",
    },
    # -- FOLLOW-UP QUESTIONS --
    {
        "id": "Q32", "category": "Card Fees", "type": "follow-up",
        "question": "Is the 4% monthly interest rate the same for all Banque Misr credit cards?",
        "ground_truth": "Yes, the interest rate of 4% per month is applied consistently across Banque Misr credit cards as documented in the knowledge base.",
    },
    {
        "id": "Q33", "category": "Installments", "type": "follow-up",
        "question": "If I use the 3-month installment plan, what is the monthly interest rate compared to the 36-month plan?",
        "ground_truth": "The 3-month installment plan has a monthly interest rate of 2.81%, while the 36-month plan has a lower rate of 2.56%.",
    },
    {
        "id": "Q34", "category": "Card Benefits", "type": "follow-up",
        "question": "Can I use 100% of my credit limit for cash withdrawals with the Classic Credit Card?",
        "ground_truth": "Yes, the Classic Credit Card allows the use of 100% of the card credit limit in cash withdrawals.",
    },
    # -- MULTI-TURN QUESTIONS --
    {
        "id": "Q35", "category": "Card Benefits", "type": "multi-turn",
        "question": "What are the main benefits of the Classic Credit Card for a new customer?",
        "ground_truth": (
            "Classic Credit Card benefits include: issuance with in-kind warranty or personal guarantee, "
            "local and international usage, 100% cash withdrawal of credit limit, supplementary cards, "
            "internet usage, 56-day grace period, 5% minimum payment, payment via ATM/branch/internet, "
            "contactless technology, free SMS notifications, installments up to 36 months, and "
            "interest-free installments up to 60 months at partner merchants."
        ),
    },
    {
        "id": "Q36", "category": "Card Benefits", "type": "multi-turn",
        "question": "How does the Platinum Visa Credit Card differ from the Classic in terms of travel benefits?",
        "ground_truth": (
            "The Platinum Visa Credit Card offers travel-specific benefits that the Classic Card does not have: "
            "6 free airport lounge accesses via Visa Airport Companion App (requires USD 5+ international transaction "
            "within 90 days), medical and legal services during travel, and purchase protection for 365 days. "
            "The Classic Card has no travel lounge access or purchase protection."
        ),
    },
    {
        "id": "Q37", "category": "Card Limits", "type": "multi-turn",
        "question": "I travel frequently internationally. What is the monthly international purchase limit on the Platinum Card if I notify the bank before travel?",
        "ground_truth": "If you inform the bank of your travel, the international purchase limit (E-commerce + POS) for the Platinum Credit Card is EGP 300,000 per month.",
    },
    # -- ARABIC QUESTIONS (Bilingual) --
    {
        "id": "Q38", "category": "Card Fees", "type": "direct",
        "question": "ما هي رسوم اصدار بطاقة الكريديت الكلاسيك من بنك مصر؟",
        "ground_truth": "رسوم اصدار بطاقة الكريديت الكلاسيك من بنك مصر هي 250 جنيه مصري.",
    },
    {
        "id": "Q39", "category": "Card Fees", "type": "direct",
        "question": "ما هي رسوم اصدار بطاقة الفيزا البلاتينية في بنك مصر؟",
        "ground_truth": "رسوم اصدار بطاقة الفيزا بلاتينيوم او ماستركارد بلاتينيوم من بنك مصر هي 500 جنيه مصري.",
    },
    {
        "id": "Q40", "category": "Card Benefits", "type": "direct",
        "question": "ما هي فترة السماح لبطاقة الكريديت الكلاسيك من بنك مصر؟",
        "ground_truth": "تتيح بطاقة الكريديت الكلاسيك فترة سماح تصل الى 56 يوما.",
    },
    {
        "id": "Q41", "category": "Card Limits", "type": "direct",
        "question": "ما هو الحد اليومي للسحب النقدي من ماكينات ATM بالبطاقة الائتمانية في بنك مصر؟",
        "ground_truth": "الحد اليومي للسحب النقدي من ماكينات ATM داخل مصر بالبطاقة الائتمانية هو 30,000 جنيه مصري.",
    },
    {
        "id": "Q42", "category": "Installments", "type": "direct",
        "question": "ما هي رسوم السداد المبكر لخطة التقسيط على البطاقة الائتمانية في بنك مصر؟",
        "ground_truth": "رسوم السداد المبكر لخطة التقسيط على البطاقات الائتمانية في بنك مصر هي 4% من قيمة القسط.",
    },
    {
        "id": "Q43", "category": "Card Fees", "type": "direct",
        "question": "ما هو معدل الفائدة الشهري على بطاقات الائتمان في بنك مصر؟",
        "ground_truth": "معدل الفائدة الشهري على بطاقات الائتمان في بنك مصر هو 4% شهريا.",
    },
    # -- ATM / CASH WITHDRAWAL FEES --
    {
        "id": "Q44", "category": "ATM", "type": "direct",
        "question": "Is balance inquiry at BM ATMs free for credit card holders?",
        "ground_truth": "Yes, balance inquiry via BM ATMs is free for Banque Misr credit card holders.",
    },
    {
        "id": "Q45", "category": "ATM", "type": "direct",
        "question": "What is the fee for balance inquiry at other banks' ATMs inside Egypt?",
        "ground_truth": "The fee for balance inquiry via other banks' ATMs inside Egypt is EGP 1 per transaction.",
    },
    {
        "id": "Q46", "category": "ATM", "type": "direct",
        "question": "What is the cash withdrawal fee at ATMs outside Egypt for Banque Misr credit cards?",
        "ground_truth": "Cash withdrawals outside Egypt incur a fee of 3% of the transaction's value plus EGP 50 per transaction.",
    },
    {
        "id": "Q47", "category": "ATM", "type": "direct",
        "question": "What is the domestic cash withdrawal fee at BM ATMs for credit cards?",
        "ground_truth": "Cash withdrawals through BM ATMs and POS within Egypt incur a fee of 2% of the amount withdrawn with a minimum of EGP 15.",
    },
    # -- CARD INTERNATIONAL USAGE --
    {
        "id": "Q48", "category": "Card Features", "type": "direct",
        "question": "When can I start using my Banque Misr credit card internationally after issuance?",
        "ground_truth": "International usage is available after 2 months from the date of credit card issuance.",
    },
    # -- OUT-OF-SCOPE QUESTIONS (Testing refusal behavior) --
    {
        "id": "Q49", "category": "Out-of-Scope", "type": "direct",
        "question": "What are the interest rates for personal loans at Banque Misr?",
        "ground_truth": "EXPECTED REFUSAL: The RAG system should return a refusal message because the knowledge base does not contain information about personal loans.",
    },
    {
        "id": "Q50", "category": "Out-of-Scope", "type": "direct",
        "question": "How do I open a savings account at Banque Misr?",
        "ground_truth": "EXPECTED REFUSAL: The RAG system should return a refusal message because the knowledge base does not contain information about savings accounts.",
    },
]


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class EvalResult:
    """Complete evaluation result for one question."""
    id: str
    category: str
    type: str
    question: str
    ground_truth: str
    retrieved_answer: str
    api_status: str             # "success" | "error" | "timeout"
    api_latency_ms: float
    similarity_pct: float       # 0.0-100.0
    score: float                # 0.0-10.0
    verdict: str                # Excellent | Good | Acceptable | Poor | Fail
    explanation: str
    is_refusal: bool            # True if system returned a refusal message
    expected_refusal: bool      # True if question is out-of-scope


# ==============================================================================
# STEP 3 -- Call the Real RAG API
# ==============================================================================

async def call_rag_api(
    question: str,
    client: httpx.AsyncClient,
) -> tuple[str, str, float]:
    """
    Call the live FastAPI /eval/query endpoint.
    Returns (retrieved_answer, status, latency_ms).
    """
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{RAG_API_BASE}/eval/query",
            json={"question": question},
            timeout=90.0,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "")
            return answer, "success", latency_ms
        else:
            return f"[API ERROR {response.status_code}]: {response.text[:300]}", "error", latency_ms

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return "[TIMEOUT]: RAG API did not respond within 90 seconds", "timeout", latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return f"[EXCEPTION]: {exc}", "error", latency_ms


# ==============================================================================
# STEP 4 -- LLM-as-a-Judge
# ==============================================================================

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial RAG evaluation judge for a Banque Misr banking customer service assistant.

Your task is to compare a Ground Truth answer against a Retrieved Answer produced by a live RAG system.

Evaluate along these 6 dimensions:
1. Semantic Correctness  - Does the retrieved answer convey the correct information?
2. Completeness          - Does it cover all key facts from the ground truth?
3. Relevance             - Is the answer focused on what was asked?
4. Grounding             - Is the answer grounded in retrievable banking facts (not hallucinated)?
5. Hallucinations        - Does the answer contain invented facts, wrong numbers, or false claims?
6. Missing Information   - What key information from the ground truth is absent?

Special rules:
- If the Ground Truth starts with "EXPECTED REFUSAL:", the question is out-of-scope.
  In this case: if the Retrieved Answer is a polite refusal ("I don't have enough information", etc.),
  assign Score=10 (perfect behavior). If the system hallucinated an answer instead of refusing, assign Score=0-2.
- Be especially strict about numeric values (fees, percentages, limits).
  A wrong number (e.g., EGP 300 instead of EGP 250) = major deduction (-2 to -4 points).
- A partially correct answer is NOT excellent.

Return ONLY valid JSON with no markdown fences, no extra text. Use this exact format:
{"similarity_pct": <float 0-100>, "score": <float 0-10>, "verdict": "<Excellent|Good|Acceptable|Poor|Fail>", "explanation": "<2-4 sentences citing specific correct/incorrect facts>"}

Verdict thresholds:
  Excellent  : score >= 9.0
  Good       : score >= 7.0
  Acceptable : score >= 5.0
  Poor       : score >= 3.0
  Fail       : score <  3.0"""


async def llm_judge(
    question: str,
    ground_truth: str,
    retrieved_answer: str,
    client: httpx.AsyncClient,
) -> tuple[float, float, str, str]:
    """
    Call Groq LLM-as-a-Judge.
    Returns (similarity_pct, score, verdict, explanation).
    """
    user_prompt = (
        f"Question: {question}\n\n"
        f"Ground Truth Answer:\n{ground_truth}\n\n"
        f"Retrieved Answer (from live RAG system):\n{retrieved_answer}\n\n"
        f"Evaluate the Retrieved Answer against the Ground Truth. Return JSON only."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }

    try:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        data = json.loads(content)
        sim    = float(data.get("similarity_pct", 0.0))
        score  = float(data.get("score", 0.0))
        verdict     = data.get("verdict", "Fail")
        explanation = data.get("explanation", "No explanation provided.")
        return sim, score, verdict, explanation

    except json.JSONDecodeError as exc:
        logger.warning("Judge returned non-JSON content: %s", exc)
        return 0.0, 0.0, "Fail", f"Judge returned unparseable content: {exc}"
    except Exception as exc:
        logger.error("Judge API error for question: %s", exc)
        return 0.0, 0.0, "Fail", f"Judge call failed: {exc}"


# ==============================================================================
# Refusal Detection
# ==============================================================================

REFUSAL_PHRASES = [
    "لا أملك معلومات",
    "لا يمكنني",
    "لا تتوفر",
    "I don't have enough information",
    "not have enough",
    "insufficient",
    "outside the scope",
    "cannot answer",
    "no information",
    "not covered",
]


def is_refusal_response(answer: str) -> bool:
    """Detect if the RAG system returned a refusal message."""
    answer_lower = answer.lower()
    return any(phrase.lower() in answer_lower for phrase in REFUSAL_PHRASES)


# ==============================================================================
# STEP 5 -- Markdown Report Generator
# ==============================================================================

def generate_markdown_report(results: list[EvalResult], run_timestamp: str) -> str:
    """Generate the full markdown evaluation report from actual results."""
    total             = len(results)
    successful_calls  = sum(1 for r in results if r.api_status == "success")
    failed_calls      = total - successful_calls
    avg_score         = sum(r.score for r in results) / total if total else 0.0
    avg_similarity    = sum(r.similarity_pct for r in results) / total if total else 0.0
    avg_latency       = (
        sum(r.api_latency_ms for r in results if r.api_status == "success")
        / max(successful_calls, 1)
    )

    verdict_counts = {"Excellent": 0, "Good": 0, "Acceptable": 0, "Poor": 0, "Fail": 0}
    for r in results:
        v = r.verdict if r.verdict in verdict_counts else "Fail"
        verdict_counts[v] += 1

    expected_refusals      = [r for r in results if r.expected_refusal]
    correct_refusals       = [r for r in expected_refusals if r.is_refusal]
    oos_hallucinations     = [r for r in expected_refusals if not r.is_refusal]

    categories: dict[str, list[EvalResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    failures = [r for r in results if r.score < 5.0]

    def _cell(text: str, max_len: int = 200) -> str:
        t = text.replace("|", "\\|").replace("\n", " ").strip()
        return (t[:max_len] + "...") if len(t) > max_len else t

    lines: list[str] = []
    lines += [
        "# Banque Misr RAG Evaluation Report",
        "",
        f"**Generated:** {run_timestamp}  ",
        f"**Model Under Test:** bge-m3 Embeddings + Qdrant + Two-Pass Retrieval + Groq llama-3.3-70b-versatile  ",
        f"**Judge Model:** Groq llama-3.3-70b-versatile (LLM-as-a-Judge)  ",
        f"**Total Questions:** {total}  ",
        f"**Successful API Calls:** {successful_calls} / {total}  ",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report evaluates the **live** Banque Misr RAG Customer Service pipeline.",
        "Every Retrieved Answer in this report was produced by calling the real FastAPI endpoint",
        "(`POST http://localhost:8000/eval/query`), which routes through the full pipeline:",
        "",
        "```",
        "Question",
        "  --> FastAPI /eval/query",
        "  --> QueryNormalizer (cleans + normalizes the query)",
        "  --> bge-m3 Embeddings (dense vector encoding)",
        "  --> Qdrant (vector similarity search)",
        "  --> Two-Pass Retrieval + UnknownAnswerDetector",
        "  --> ContextBuilder + PromptBuilder",
        "  --> Groq llama-3.3-70b-versatile (LLM generation)",
        "  --> Returned Answer",
        "```",
        "",
        "No answers were simulated, estimated, or generated by the evaluator.",
        "If the API returned an incorrect answer, it is scored accordingly.",
        "If the API hallucinated, it is detected and penalized.",
        "If retrieval failed (INSUFFICIENT_CONTEXT), the refusal message is what is scored.",
        "",
        "**Knowledge Base Scope:** The RAG system's knowledge base contains documentation",
        "for Banque Misr credit card products only:",
        "Classic, Gold, Platinum, Titanium, Visa Infinite, Visa Infinite Private,",
        "Visa Signature, World, World Elite, Al-Araby, Asatha, and Secured cards.",
        "Questions about personal loans, savings accounts, and certificates are **out-of-scope**",
        "and should trigger a polite refusal.",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Average Score** | **{avg_score:.2f} / 10** |",
        f"| **Average Similarity** | **{avg_similarity:.1f}%** |",
        f"| **Average Latency** | **{avg_latency:.0f} ms** |",
        f"| Successful API Calls | {successful_calls} / {total} |",
        f"| Failed / Timeout | {failed_calls} |",
        f"| Correct Refusals (out-of-scope) | {len(correct_refusals)} / {len(expected_refusals)} |",
        f"| OOS Hallucinations | {len(oos_hallucinations)} |",
        "",
        "### Verdict Distribution",
        "",
        "| Verdict | Count | Percentage |",
        "|---------|-------|------------|",
    ]

    for verdict, count in verdict_counts.items():
        pct = (count / total * 100) if total else 0.0
        lines.append(f"| {verdict} | {count} | {pct:.1f}% |")

    lines += [
        "",
        "### Score by Category",
        "",
        "| Category | Questions | Avg Score | Avg Similarity |",
        "|----------|-----------|-----------|----------------|",
    ]
    for cat, cat_results in sorted(categories.items()):
        cat_avg_score = sum(r.score for r in cat_results) / len(cat_results)
        cat_avg_sim   = sum(r.similarity_pct for r in cat_results) / len(cat_results)
        lines.append(f"| {cat} | {len(cat_results)} | {cat_avg_score:.2f} | {cat_avg_sim:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## Detailed Results",
        "",
        "| # | Category | Type | Question | Ground Truth | Retrieved Answer | Sim% | Score | Verdict | Explanation |",
        "|---|----------|------|----------|--------------|-----------------|------|-------|---------|-------------|",
    ]

    for r in results:
        status_icon = "" if r.api_status == "success" else "[FAIL] "
        lines.append(
            f"| {r.id} | {r.category} | {r.type} "
            f"| {_cell(r.question, 100)} "
            f"| {_cell(r.ground_truth, 130)} "
            f"| {status_icon}{_cell(r.retrieved_answer, 180)} "
            f"| {r.similarity_pct:.0f}% "
            f"| {r.score:.1f}/10 "
            f"| **{r.verdict}** "
            f"| {_cell(r.explanation, 220)} |"
        )

    # Failure Analysis
    lines += [
        "",
        "---",
        "",
        "## Failure Analysis",
        "",
        f"Questions scoring below 5.0/10: **{len(failures)}** out of {total}",
        "",
    ]

    if not failures:
        lines.append("No questions scored below 5.0/10. The RAG system performed acceptably or better on all questions.")
    else:
        lines.append("### Failed Questions (Score < 5.0/10)")
        lines.append("")
        for r in failures:
            lines += [
                f"#### {r.id} [{r.verdict}] -- {r.question}",
                f"- **Score:** {r.score:.1f}/10 | **Similarity:** {r.similarity_pct:.0f}%",
                f"- **Ground Truth:** {r.ground_truth}",
                f"- **Retrieved Answer:** {r.retrieved_answer}",
                f"- **Explanation:** {r.explanation}",
                "",
            ]

        lines += ["### Failure Patterns", ""]

        retrieval_fails = [r for r in failures if r.api_status != "success"]
        oos_fails       = [r for r in failures if r.expected_refusal and not r.is_refusal]
        halluc_fails    = [r for r in failures if not r.expected_refusal and r.score < 3.0]
        partial_fails   = [r for r in failures if r not in retrieval_fails + oos_fails + halluc_fails]

        if retrieval_fails:
            lines += [
                f"**Pattern 1 - API/Retrieval Failures ({len(retrieval_fails)} cases):**",
                "The following questions received no answer due to API errors or timeouts:",
            ]
            for r in retrieval_fails:
                lines.append(f"- `{r.id}`: {r.question} -> `{r.retrieved_answer[:120]}`")
            lines.append("")

        if oos_fails:
            lines += [
                f"**Pattern 2 - Out-of-Scope Hallucination ({len(oos_fails)} cases):**",
                "The system attempted to answer questions that should have been refused.",
                "This indicates the retrieval thresholds may be too permissive.",
            ]
            for r in oos_fails:
                lines.append(f"- `{r.id}`: {r.question}")
            lines.append("")

        if halluc_fails:
            lines += [
                f"**Pattern 3 - Severe Hallucination / Wrong Facts ({len(halluc_fails)} cases):**",
                "The system produced answers with factually incorrect information.",
            ]
            for r in halluc_fails:
                lines.append(f"- `{r.id}`: {r.question} -- *{r.explanation[:180]}*")
            lines.append("")

        if partial_fails:
            lines += [
                f"**Pattern 4 - Incomplete Answers ({len(partial_fails)} cases):**",
                "The system returned partially correct but incomplete responses.",
            ]
            for r in partial_fails:
                lines.append(f"- `{r.id}`: {r.question} -- *{r.explanation[:180]}*")
            lines.append("")

    # Recommendations
    lines += [
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "> Every recommendation below is supported by at least one actual failed evaluation.",
        "> No recommendation is made without a corresponding failure case in this report.",
        "",
    ]

    if not failures:
        lines.append("The RAG system scored acceptably or better on all questions. No critical improvements required at this time.")
    else:
        rec_idx = 1
        if oos_fails:
            lines += [
                f"**{rec_idx}. Strengthen Out-of-Scope Detection**",
                f"  - Evidence: {len(oos_fails)} out-of-scope question(s) received hallucinated answers (see Pattern 2).",
                "  - Action: Increase `UNKNOWN_DETECTOR_MIN_SCORE` from 0.58 to 0.65.",
                "  - Action: Add topic boundary guards (allowlist of known product types).",
                "",
            ]
            rec_idx += 1

        if halluc_fails:
            lines += [
                f"**{rec_idx}. Reduce Hallucination in LLM Generation**",
                f"  - Evidence: {len(halluc_fails)} answer(s) contained fabricated or incorrect numerical facts (see Pattern 3).",
                "  - Action: Set `GROQ_TEMPERATURE=0.0` and reduce `GROQ_MAX_TOKENS` to 350.",
                "  - Action: Add a post-generation numeric validation step.",
                "",
            ]
            rec_idx += 1

        if partial_fails:
            lines += [
                f"**{rec_idx}. Improve Context Completeness**",
                f"  - Evidence: {len(partial_fails)} answer(s) were incomplete (see Pattern 4).",
                "  - Action: Increase `RAG_TOP_K` from 5 to 7 to retrieve more supporting chunks.",
                "  - Action: Review knowledge base chunking strategy to avoid splitting fee tables across chunks.",
                "",
            ]
            rec_idx += 1

        if retrieval_fails:
            lines += [
                f"**{rec_idx}. Improve API Reliability**",
                f"  - Evidence: {len(retrieval_fails)} question(s) received no answer due to API/network failures.",
                "  - Action: Add retry logic (3 attempts, exponential backoff) to the evaluation client.",
                "  - Action: Add Qdrant health check to FastAPI startup sequence.",
                "",
            ]

    lines += [
        "",
        "---",
        "",
        "*Report generated by the Banque Misr RAG Evaluation Pipeline.*",
        f"*Run timestamp: {run_timestamp}*",
    ]

    return "\n".join(lines)


# ==============================================================================
# Main Evaluation Runner
# ==============================================================================

async def run_evaluation() -> None:
    """Execute the complete 5-step evaluation pipeline."""

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 70)
    logger.info("Banque Misr RAG Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info("RAG API Base  : %s", RAG_API_BASE)
    logger.info("Judge Model   : %s", GROQ_MODEL)
    logger.info("Total Cases   : %d", len(BENCHMARK_QUESTIONS))
    logger.info("=" * 70)

    # Preflight check
    async with httpx.AsyncClient() as probe:
        try:
            r = await probe.get(f"{RAG_API_BASE}/", timeout=5.0)
            logger.info("FastAPI server is UP -- status: %s", r.json().get("status", "unknown"))
        except Exception as exc:
            logger.error("Cannot reach RAG API at %s: %s", RAG_API_BASE, exc)
            logger.error("Please start the FastAPI server and retry.")
            sys.exit(1)

        # Check /eval/query endpoint
        try:
            test_r = await probe.post(
                f"{RAG_API_BASE}/eval/query",
                json={"question": "ping"},
                timeout=30.0,
            )
            if test_r.status_code == 404:
                logger.error(
                    "The /eval/query endpoint does not exist.\n"
                    "Add it to app/main.py by running the server with the updated main.py."
                )
                sys.exit(1)
            else:
                logger.info("/eval/query endpoint verified (status %d)", test_r.status_code)
        except httpx.TimeoutException:
            logger.warning("/eval/query preflight timed out -- will proceed anyway.")

    results: list[EvalResult] = []

    async with httpx.AsyncClient() as client:
        for idx, case in enumerate(BENCHMARK_QUESTIONS, 1):
            q_id          = case["id"]
            question      = case["question"]
            ground_truth  = case["ground_truth"]
            expected_ref  = ground_truth.startswith("EXPECTED REFUSAL:")

            logger.info(
                "[%02d/%02d] %s | %s",
                idx, len(BENCHMARK_QUESTIONS), q_id, question[:80]
            )

            # STEP 3: Call the real RAG API
            retrieved_answer, api_status, latency_ms = await call_rag_api(question, client)

            if api_status != "success":
                logger.warning("  API FAILED (%s): %s", api_status, retrieved_answer[:120])
                results.append(EvalResult(
                    id=q_id,
                    category=case["category"],
                    type=case["type"],
                    question=question,
                    ground_truth=ground_truth,
                    retrieved_answer=retrieved_answer,
                    api_status=api_status,
                    api_latency_ms=latency_ms,
                    similarity_pct=0.0,
                    score=0.0,
                    verdict="Fail",
                    explanation=f"API call failed ({api_status}): {retrieved_answer[:200]}",
                    is_refusal=False,
                    expected_refusal=expected_ref,
                ))
                continue

            is_ref = is_refusal_response(retrieved_answer)
            logger.info("  Latency: %.0f ms | Refusal: %s", latency_ms, is_ref)
            logger.info("  Answer : %s", retrieved_answer[:120])

            # STEP 4: LLM-as-a-Judge
            similarity_pct, score, verdict, explanation = await llm_judge(
                question, ground_truth, retrieved_answer, client
            )
            logger.info(
                "  Judge  : Score=%.1f/10 | Sim=%.0f%% | Verdict=%s",
                score, similarity_pct, verdict
            )

            results.append(EvalResult(
                id=q_id,
                category=case["category"],
                type=case["type"],
                question=question,
                ground_truth=ground_truth,
                retrieved_answer=retrieved_answer,
                api_status=api_status,
                api_latency_ms=latency_ms,
                similarity_pct=similarity_pct,
                score=score,
                verdict=verdict,
                explanation=explanation,
                is_refusal=is_ref,
                expected_refusal=expected_ref,
            ))

            # Small pause to avoid rate-limit on judge calls
            await asyncio.sleep(0.4)

    # STEP 5: Generate reports
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE -- Generating reports...")
    logger.info("=" * 70)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save raw JSON results
    json_data = [
        {
            "id":               r.id,
            "category":         r.category,
            "type":             r.type,
            "question":         r.question,
            "ground_truth":     r.ground_truth,
            "retrieved_answer": r.retrieved_answer,
            "api_status":       r.api_status,
            "api_latency_ms":   round(r.api_latency_ms, 2),
            "similarity_pct":   round(r.similarity_pct, 2),
            "score":            round(r.score, 2),
            "verdict":          r.verdict,
            "explanation":      r.explanation,
            "is_refusal":       r.is_refusal,
            "expected_refusal": r.expected_refusal,
        }
        for r in results
    ]
    with RESULTS_JSON.open("w", encoding="utf-8") as fh:
        json.dump(json_data, fh, ensure_ascii=False, indent=2)
    logger.info("JSON results -> %s", RESULTS_JSON)

    # Generate and save markdown report
    report_md = generate_markdown_report(results, run_timestamp)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        fh.write(report_md)
    logger.info("Markdown report -> %s", REPORT_PATH)

    # Console summary
    total     = len(results)
    avg_score = sum(r.score for r in results) / total if total else 0.0
    avg_sim   = sum(r.similarity_pct for r in results) / total if total else 0.0
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("Total questions     : %d", total)
    logger.info("Average Score       : %.2f / 10", avg_score)
    logger.info("Average Similarity  : %.1f%%", avg_sim)
    logger.info("Verdict breakdown:")
    for v in ["Excellent", "Good", "Acceptable", "Poor", "Fail"]:
        count = sum(1 for r in results if r.verdict == v)
        logger.info("  %-12s : %d (%.1f%%)", v, count, count / total * 100 if total else 0)
    logger.info("=" * 70)


def main() -> None:
    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
