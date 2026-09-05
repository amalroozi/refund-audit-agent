#!/usr/bin/env python3
"""
reconcile.py
------------
Rule-based refund reconciliation pipeline.

Reads the two "real" datasets produced by generate_synthetic_data.py:

    data/crm_refund_promises.csv   -- what support/CRM promised the customer
    data/gateway_refunds.csv       -- what actually cleared in the gateway

and, for every CRM promise, decides whether it is:

    clean_match  -- refunded amount ~= promised amount, inside the window
    partial      -- refund happened but for meaningfully less than promised
    delayed      -- full amount refunded, but after the promised window
    no_refund    -- no gateway transaction exists for that order at all

Classification itself is pure rule-based logic on parsed amounts/dates --
no LLM calls. Once labels are final, ai_triage.py makes ONE additional
pass over just the exceptions (partial/delayed/no_refund) to assign an AI
severity judgment (explained/needs_review/critical) alongside the label --
it can read the label but can never change it. A *separate* scoring step
then checks the pipeline's deterministic labels against
data/ground_truth.csv purely as an external accuracy check; neither the AI
triage nor the scoring logic feeds back into the classifier itself.

Usage:
    python reconcile.py
"""

import csv
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import ai_triage

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
CRM_PATH = DATA_DIR / "crm_refund_promises.csv"
GATEWAY_PATH = DATA_DIR / "gateway_refunds.csv"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.csv"

# A refund is considered "matching" the promised amount if it's within
# this relative tolerance OR this absolute rupee tolerance (whichever is
# larger) -- this absorbs rounding/formatting noise without masking a real
# partial-refund deduction (which, in this dataset, is a ~3% haircut).
AMOUNT_REL_TOLERANCE = 0.01   # 1%
AMOUNT_ABS_TOLERANCE = 1.00   # ₹1

# Date formats we expect to see across the two files (mixed on purpose).
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"]

CURRENCY = "₹"  # ₹


# --------------------------------------------------------------------------
# Step 1-3: robust parsing helpers
# --------------------------------------------------------------------------

def parse_amount(raw: str) -> float:
    """Parse amount strings like '1999.00', '1,999.00', 'INR 1,999.00'
    into a plain float, stripping currency prefixes/symbols and thousand
    separators."""
    if raw is None:
        return 0.0
    cleaned = raw.strip()
    cleaned = re.sub(r"(?i)^(inr|rs\.?|₹)\s*", "", cleaned)  # currency prefix
    cleaned = cleaned.replace(",", "").replace("₹", "").strip()
    return float(cleaned)


def parse_date(raw: str) -> date:
    """Parse a date string that may be in any of DATE_FORMATS."""
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {raw!r}")


def extract_order_number(order_id: str) -> str:
    """Extract the numeric portion of an order_id regardless of the
    surrounding format/prefix, e.g. 'ORD100027' and 'order_100027' both
    -> '100027'. This is the real join key between the two systems, since
    the two systems format order_id completely differently."""
    match = re.search(r"\d+", order_id)
    if not match:
        raise ValueError(f"No numeric order id found in: {order_id!r}")
    return match.group()  # kept as string to preserve leading zeros if any


def parse_window(window_str: str):
    """'5-8' -> (5, 8) as (window_start_days, window_end_days)."""
    start_str, end_str = window_str.split("-")
    return int(start_str), int(end_str)


# --------------------------------------------------------------------------
# Load both CSVs
# --------------------------------------------------------------------------

def load_crm_promises(path: Path):
    """Returns a list of dicts, one per CRM promise, with fields parsed
    into usable types (floats/dates) plus the extracted join key."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            window_start, window_end = parse_window(row["promised_window_days"])
            rows.append({
                "promise_id": row["promise_id"],
                "customer_id": row["customer_id"],
                "order_id": row["order_id"],
                "order_num": extract_order_number(row["order_id"]),
                "promised_amount": parse_amount(row["promised_amount"]),
                "promise_date": parse_date(row["promise_date"]),
                "window_start_days": window_start,
                "window_end_days": window_end,
                "agent_notes": row["agent_notes"],
            })
    return rows


def load_gateway_refunds(path: Path):
    """Returns a dict: order_num -> aggregated gateway info.

    In this dataset there is at most one gateway row per order, but the
    aggregation below is written defensively in case an order ever has
    multiple gateway transactions (e.g. a retried payout): amounts are
    summed and the latest processed date is used, since that's the date
    the customer's money was actually fully settled.
    """
    by_order = defaultdict(lambda: {
        "refund_txn_ids": [],
        "total_refunded": 0.0,
        "latest_processed_date": None,
        "gateway_references": [],
    })
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            order_num = extract_order_number(row["order_id"])
            amount = parse_amount(row["refunded_amount"])
            processed_date = parse_date(row["refund_processed_date"])

            bucket = by_order[order_num]
            bucket["refund_txn_ids"].append(row["refund_txn_id"])
            bucket["total_refunded"] += amount
            bucket["gateway_references"].append(row["gateway_reference"])
            if bucket["latest_processed_date"] is None or processed_date > bucket["latest_processed_date"]:
                bucket["latest_processed_date"] = processed_date

    return dict(by_order)


# --------------------------------------------------------------------------
# Step 5: human-readable reason strings for anything that isn't a clean match
# --------------------------------------------------------------------------

def reason_partial(promised_amount, refunded_amount):
    diff = round(promised_amount - refunded_amount, 2)
    pct = (diff / promised_amount * 100) if promised_amount else 0.0
    return (
        f"Promised {CURRENCY}{promised_amount:.0f}, gateway shows {CURRENCY}{refunded_amount:.0f} "
        f"— {CURRENCY}{diff:.0f} short (~{pct:.1f}%), no matching fee schedule provided"
    )


def reason_delayed(refunded_amount, promise_date, window_end_days, processed_date):
    deadline = promise_date + timedelta(days=window_end_days)
    late_days = (processed_date - deadline).days
    return (
        f"Refund of {CURRENCY}{refunded_amount:.0f} processed on {processed_date.isoformat()}, "
        f"{late_days} day(s) past the promised {window_end_days}-day deadline ({deadline.isoformat()})"
    )


def reason_no_refund(promised_amount, promise_date, order_id):
    return (
        f"CRM promised {CURRENCY}{promised_amount:.0f} refund on {promise_date.isoformat()} "
        f"for order {order_id}, but no matching gateway transaction was found"
    )


# --------------------------------------------------------------------------
# Step 4: rule-based classifier
# --------------------------------------------------------------------------

def amounts_match(promised: float, refunded: float) -> bool:
    tolerance = max(AMOUNT_ABS_TOLERANCE, promised * AMOUNT_REL_TOLERANCE)
    return abs(promised - refunded) <= tolerance


def classify_case(crm_row, gateway_by_order):
    """Applies the four classification rules, in priority order:
    no matching gateway row -> no_refund; amount short -> partial;
    amount OK but late -> delayed; otherwise -> clean_match.
    Returns a result dict with the label, reason (None for clean_match),
    and the numbers used, for reporting.
    """
    match = gateway_by_order.get(crm_row["order_num"])

    result = {
        "promise_id": crm_row["promise_id"],
        "order_id": crm_row["order_id"],
        "promised_amount": crm_row["promised_amount"],
        "refunded_amount": None,
        "reason": None,
        # Carried through for ai_triage.py's prompts and the report --
        # the classifier itself doesn't need these beyond this point.
        "agent_notes": crm_row["agent_notes"],
        "promise_date": crm_row["promise_date"],
        "refund_processed_date": None,
        "window_start_days": crm_row["window_start_days"],
        "window_end_days": crm_row["window_end_days"],
    }

    # Rule 1: no gateway transaction at all -> true exception
    if match is None:
        result["label"] = "no_refund"
        result["reason"] = reason_no_refund(
            crm_row["promised_amount"], crm_row["promise_date"], crm_row["order_id"]
        )
        return result

    refunded_amount = match["total_refunded"]
    processed_date = match["latest_processed_date"]
    result["refunded_amount"] = refunded_amount
    result["refund_processed_date"] = processed_date

    # Rule 2: refunded meaningfully less than promised -> partial
    if refunded_amount < crm_row["promised_amount"] and not amounts_match(
        crm_row["promised_amount"], refunded_amount
    ):
        result["label"] = "partial"
        result["reason"] = reason_partial(crm_row["promised_amount"], refunded_amount)
        return result

    # Rule 3: amount is fine, but it arrived after the promised window -> delayed
    deadline = crm_row["promise_date"] + timedelta(days=crm_row["window_end_days"])
    if processed_date > deadline:
        result["label"] = "delayed"
        result["reason"] = reason_delayed(
            refunded_amount, crm_row["promise_date"], crm_row["window_end_days"], processed_date
        )
        return result

    # Rule 4: amount matches (or exceeds) and it's within the window -> clean
    result["label"] = "clean_match"
    return result


def run_pipeline():
    crm_rows = load_crm_promises(CRM_PATH)
    gateway_by_order = load_gateway_refunds(GATEWAY_PATH)
    return [classify_case(row, gateway_by_order) for row in crm_rows]


# --------------------------------------------------------------------------
# Step 6: scoring against ground_truth.csv
# --------------------------------------------------------------------------
# This is intentionally a separate function that only ever *reads* the
# pipeline's output -- it plays no part in how classify_case() decides
# labels. It exists purely to answer "how good is the agent?".

def load_ground_truth(path: Path):
    truth = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            truth[row["promise_id"]] = row["label"]
    return truth


def score_against_ground_truth(results):
    truth = load_ground_truth(GROUND_TRUTH_PATH)
    labels = ["clean_match", "partial", "delayed", "no_refund"]

    confusion = defaultdict(lambda: defaultdict(int))  # confusion[true][predicted]
    correct = 0
    total = 0
    mismatches = []

    for r in results:
        true_label = truth.get(r["promise_id"])
        if true_label is None:
            continue  # shouldn't happen, but don't crash scoring on it
        predicted_label = r["label"]
        confusion[true_label][predicted_label] += 1
        total += 1
        if predicted_label == true_label:
            correct += 1
        else:
            mismatches.append((r["promise_id"], r["order_id"], true_label, predicted_label))

    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "confusion": confusion,
        "mismatches": mismatches,
        "labels": labels,
    }


# --------------------------------------------------------------------------
# Step 7: reporting
# --------------------------------------------------------------------------

def print_report(results, score, triage_summary):
    total = len(results)
    counts = defaultdict(int)
    for r in results:
        counts[r["label"]] += 1

    print("=" * 72)
    print("REFUND RECONCILIATION REPORT")
    print("=" * 72)
    print(f"Total CRM-promised refund cases: {total}\n")

    clean = counts.get("clean_match", 0)
    print(f"Clean match rate: {clean}/{total} ({clean / total * 100:.1f}%)\n")

    print("Breakdown by category:")
    for label in ["clean_match", "partial", "delayed", "no_refund"]:
        n = counts.get(label, 0)
        print(f"  {label:<12} {n:>3}  ({n / total * 100:5.1f}%)")
    print()

    exceptions = [r for r in results if r["label"] != "clean_match"]
    print(f"Exception list ({len(exceptions)} cases needing attention), grouped by AI severity:")
    print("-" * 72)
    # Severity is the AI's own judgment call, layered on top of the fixed
    # deterministic label -- it never changes which label a case got.
    # Ordered most- to least-urgent for a human triaging the list top-down.
    severity_order = ["critical", "needs_review", "explained"]
    for severity in severity_order:
        cases = [r for r in exceptions if r.get("severity") == severity]
        if not cases:
            continue
        print(f"\n  -- {severity.upper()} ({len(cases)}) --")
        for r in sorted(cases, key=lambda x: (x["label"], x["order_id"])):
            fallback_tag = " [AI FALLBACK]" if r.get("ai_fallback") else ""
            confidence = r.get("confidence")
            conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
            print(f"  [{r['label'].upper():<11}] {r['promise_id']} ({r['order_id']}){fallback_tag}")
            print(f"      rule-based reason: {r['reason']}")
            print(f"      AI justification (confidence {conf_str}): {r['justification']}")
    print()

    print(
        f"{triage_summary['total_exceptions']} exceptions: "
        f"{triage_summary['critical']} critical, "
        f"{triage_summary['explained']} explained, "
        f"{triage_summary['needs_review']} need review"
    )
    print(
        f"AI triage fallbacks: {triage_summary['fallback_count']}/{triage_summary['total_exceptions']} "
        f"exceptions defaulted to needs_review (no note logged, API failure, bad JSON, or low confidence)."
    )
    print()

    print("=" * 72)
    print("ACCURACY VS GROUND TRUTH (external scoring, not used by the agent)")
    print("=" * 72)
    print(f"Overall accuracy: {score['correct']}/{score['total']} ({score['accuracy'] * 100:.1f}%)\n")

    print("Confusion matrix (rows = true label, columns = predicted label):")
    header = " " * 14 + "".join(f"{l:>13}" for l in score["labels"])
    print(header)
    for true_label in score["labels"]:
        row = score["confusion"].get(true_label, {})
        line = f"{true_label:<14}" + "".join(f"{row.get(pred, 0):>13}" for pred in score["labels"])
        print(line)
    print()

    if score["mismatches"]:
        print(f"Misclassified cases ({len(score['mismatches'])}):")
        for promise_id, order_id, true_label, predicted_label in score["mismatches"]:
            print(f"  {promise_id} ({order_id}): true={true_label} predicted={predicted_label}")
    else:
        print("No misclassifications \U0001F389")
    print()


def main():
    results = run_pipeline()

    # AI triage runs strictly after classification, on the exceptions only,
    # and only ever adds severity/justification fields -- see ai_triage.py.
    triage_summary = ai_triage.apply_ai_triage(results)

    # Ground-truth scoring reads only the deterministic `label` field above;
    # it never sees or is affected by the AI severities.
    score = score_against_ground_truth(results)

    print_report(results, score, triage_summary)


if __name__ == "__main__":
    main()
