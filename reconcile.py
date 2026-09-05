#!/usr/bin/env python3
"""
reconcile.py
------------
Rule-based refund reconciliation pipeline.

Reads two datasets -- a CRM/support-side "promises" file and a payment
gateway "actuals" file -- and, for every case, decides which of these it
is:

    clean_match      -- refunded amount ~= promised amount, inside the window
    partial          -- refund happened but for meaningfully less than promised
    over_refund      -- refund happened but for meaningfully more than promised
    delayed          -- full amount refunded, but after the promised window
    no_refund        -- CRM promised a refund; no gateway transaction exists
    double_refund     -- more than one gateway transaction for the same order
    orphan_refund    -- a gateway transaction exists with NO CRM promise at all
    unknown_pattern  -- the case doesn't cleanly fit any of the above; the
                        classifier abstains instead of forcing a label

Classification itself is pure rule-based logic on parsed amounts/dates --
no LLM calls, and it is bidirectional: it checks CRM promises against the
gateway (partial/delayed/no_refund/double_refund/unknown_pattern) AND scans
the gateway for order_ids the CRM side never mentioned at all
(orphan_refund). Once labels are final, ai_triage.py makes ONE additional
pass over just the exceptions (everything except clean_match) to assign an
AI severity judgment (explained/needs_review/critical) alongside the
label -- it can read the label but can never change it. A *separate*
scoring step then checks the pipeline's deterministic labels against a
ground_truth.csv purely as an external accuracy check; neither the AI
triage nor the scoring logic feeds back into the classifier itself.

Usage:
    python reconcile.py                    # main synthetic dataset
    python reconcile.py --dataset holdout  # adversarial held-out batch
    python reconcile.py --dataset both     # both, scored separately
"""

import argparse
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

DATASETS = {
    "main": {
        "display_name": "MAIN SYNTHETIC DATASET",
        "crm": DATA_DIR / "crm_refund_promises.csv",
        "gateway": DATA_DIR / "gateway_refunds.csv",
        "ground_truth": DATA_DIR / "ground_truth.csv",
    },
    "holdout": {
        "display_name": "HOLD-OUT ADVERSARIAL BATCH",
        "crm": DATA_DIR / "holdout_crm.csv",
        "gateway": DATA_DIR / "holdout_gateway.csv",
        "ground_truth": DATA_DIR / "holdout_ground_truth.csv",
    },
}

# A refund is considered "matching" the promised amount if it's within
# this relative tolerance OR this absolute rupee tolerance (whichever is
# larger) -- this absorbs rounding/formatting noise without masking a real
# partial-refund deduction (which, in this dataset, is a ~3% haircut).
AMOUNT_REL_TOLERANCE = 0.01   # 1%
AMOUNT_ABS_TOLERANCE = 1.00   # ₹1

# Beyond AMOUNT_REL_TOLERANCE but within this ceiling, a discrepancy is too
# large to be rounding/formatting noise but too small to confidently call a
# deliberate deduction or overpayment -- the classifier abstains
# (unknown_pattern) rather than guessing which one it is. Anything beyond
# this ceiling is a clear partial/over_refund call.
AMBIGUOUS_REL_CEILING = 0.02  # 2%

# Date formats we expect to see across the two files (mixed on purpose).
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"]

CURRENCY = "₹"  # ₹

# All possible deterministic labels, and the order they're reported in.
# clean_match is never an "exception" so it's handled separately in the
# report; LABEL_RANK controls how exceptions are ordered within the report
# (higher-risk categories first) -- orphan/double/over deliberately rank
# above delayed, since money moving with no record or for the wrong amount
# is a bigger problem than money that simply arrived late.
ALL_LABELS = [
    "clean_match", "partial", "over_refund", "delayed", "no_refund",
    "double_refund", "orphan_refund", "unknown_pattern",
]
LABEL_RANK = {
    "orphan_refund": 0,
    "double_refund": 1,
    "over_refund": 2,
    "no_refund": 3,
    "unknown_pattern": 4,
    "partial": 5,
    "delayed": 6,
}


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
    """Returns a dict: order_num -> list of individual gateway transaction
    rows, parsed into usable types.

    Rows are kept SEPARATE, never aggregated/summed -- summing silently
    would hide a double_refund (two payouts for one order) behind what
    looks like a single correct one. classify_case() is what decides how
    to interpret more than one row for an order.
    """
    by_order = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            order_num = extract_order_number(row["order_id"])
            by_order[order_num].append({
                "refund_txn_id": row["refund_txn_id"],
                "order_id": row["order_id"],
                "refunded_amount": parse_amount(row["refunded_amount"]),
                "refund_processed_date": parse_date(row["refund_processed_date"]),
                "gateway_reference": row["gateway_reference"],
            })
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


def reason_over_refund(promised_amount, refunded_amount):
    diff = round(refunded_amount - promised_amount, 2)
    pct = (diff / promised_amount * 100) if promised_amount else 0.0
    return (
        f"Promised {CURRENCY}{promised_amount:.0f}, gateway shows {CURRENCY}{refunded_amount:.0f} "
        f"— {CURRENCY}{diff:.0f} over (~{pct:.1f}%), refunded more than was promised"
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


def reason_double_refund(order_id, rows, total):
    amounts = ", ".join(f"{CURRENCY}{r['refunded_amount']:.0f}" for r in rows)
    return (
        f"{len(rows)} gateway transactions found for order {order_id} ({amounts}), "
        f"totaling {CURRENCY}{total:.0f} — expected exactly one refund per promise, not summed automatically"
    )


def reason_orphan_refund(order_id, rows, total):
    amounts = ", ".join(f"{CURRENCY}{r['refunded_amount']:.0f}" for r in rows)
    latest = max(r["refund_processed_date"] for r in rows)
    return (
        f"Gateway shows {len(rows)} refund(s) ({amounts}, totaling {CURRENCY}{total:.0f}) on "
        f"{latest.isoformat()} for order {order_id}, but no CRM promise exists for this order at all"
    )


def reason_unknown_pattern_amount(promised_amount, refunded_amount):
    diff = promised_amount - refunded_amount
    pct = abs(diff) / promised_amount * 100 if promised_amount else 0.0
    direction = "short" if diff > 0 else "over"
    return (
        f"Promised {CURRENCY}{promised_amount:.0f}, gateway shows {CURRENCY}{refunded_amount:.0f} "
        f"— {CURRENCY}{abs(diff):.0f} {direction} (~{pct:.1f}%): too large to be rounding/formatting "
        f"noise, too small to confidently call a deliberate deduction or overpayment -- abstaining "
        f"rather than forcing a label"
    )


def reason_unknown_pattern_timeline(promise_date, processed_date):
    return (
        f"Refund processed date ({processed_date.isoformat()}) is before the promise date "
        f"({promise_date.isoformat()}) — the timeline is internally inconsistent, so timeliness "
        f"cannot be determined; abstaining rather than forcing a label"
    )


# --------------------------------------------------------------------------
# Step 4: rule-based classifier
# --------------------------------------------------------------------------

def amounts_match(promised: float, refunded: float) -> bool:
    tolerance = max(AMOUNT_ABS_TOLERANCE, promised * AMOUNT_REL_TOLERANCE)
    return abs(promised - refunded) <= tolerance


def amount_relationship(promised: float, refunded: float) -> str:
    """Classifies how `refunded` relates to `promised`:
      match            -- within tolerance, effectively the same amount
      ambiguous_short/
      ambiguous_over   -- outside tolerance but only slightly (within
                          AMBIGUOUS_REL_CEILING) -- too small a gap to
                          confidently call a deliberate deduction or
                          overpayment, too big to be rounding noise
      short / over     -- clearly and meaningfully different
    """
    if amounts_match(promised, refunded):
        return "match"
    diff = promised - refunded  # positive => short, negative => over
    rel = abs(diff) / promised if promised else float("inf")
    if rel <= AMBIGUOUS_REL_CEILING:
        return "ambiguous_short" if diff > 0 else "ambiguous_over"
    return "short" if diff > 0 else "over"


def classify_case(crm_row, gateway_by_order):
    """Applies the classification rules, in priority order:

      0 gateway rows                            -> no_refund
      >1 gateway rows for the order              -> double_refund (never silently summed)
      1 row, but dated before the promise itself -> unknown_pattern (timeline is broken)
      1 row, amount ambiguously off (1%-2%)      -> unknown_pattern (abstain, don't guess)
      1 row, amount clearly short                -> partial
      1 row, amount clearly over                 -> over_refund
      1 row, amount matches, but late            -> delayed
      1 row, amount matches, on time             -> clean_match

    Returns a result dict with the label, reason (None for clean_match),
    and the numbers used, for reporting. unknown_pattern is an explicit,
    logged abstention -- not a crash, and not a forced guess.
    """
    rows = gateway_by_order.get(crm_row["order_num"], [])

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
    if not rows:
        result["label"] = "no_refund"
        result["reason"] = reason_no_refund(
            crm_row["promised_amount"], crm_row["promise_date"], crm_row["order_id"]
        )
        return result

    # Rule 2: more than one gateway transaction for this order -> flag it
    # instead of silently summing them into what would look like one
    # correct (or over-)refund.
    if len(rows) > 1:
        total = sum(r["refunded_amount"] for r in rows)
        latest_date = max(r["refund_processed_date"] for r in rows)
        result["refunded_amount"] = total
        result["refund_processed_date"] = latest_date
        result["label"] = "double_refund"
        result["reason"] = reason_double_refund(crm_row["order_id"], rows, total)
        return result

    row = rows[0]
    refunded_amount = row["refunded_amount"]
    processed_date = row["refund_processed_date"]
    result["refunded_amount"] = refunded_amount
    result["refund_processed_date"] = processed_date

    # Rule 3: the refund predates the promise -- the timeline itself is
    # broken, so don't force a delayed/clean verdict on top of bad data.
    if processed_date < crm_row["promise_date"]:
        result["label"] = "unknown_pattern"
        result["reason"] = reason_unknown_pattern_timeline(crm_row["promise_date"], processed_date)
        return result

    relationship = amount_relationship(crm_row["promised_amount"], refunded_amount)

    # Rule 4: amount is off by more than noise but less than a confident
    # deduction/overpayment call -- abstain instead of guessing.
    if relationship in ("ambiguous_short", "ambiguous_over"):
        result["label"] = "unknown_pattern"
        result["reason"] = reason_unknown_pattern_amount(crm_row["promised_amount"], refunded_amount)
        return result

    # Rule 5: refunded meaningfully less than promised -> partial
    if relationship == "short":
        result["label"] = "partial"
        result["reason"] = reason_partial(crm_row["promised_amount"], refunded_amount)
        return result

    # Rule 6: refunded meaningfully more than promised -> over_refund
    if relationship == "over":
        result["label"] = "over_refund"
        result["reason"] = reason_over_refund(crm_row["promised_amount"], refunded_amount)
        return result

    # relationship == "match" from here on.
    # Rule 7: amount is fine, but it arrived after the promised window -> delayed
    deadline = crm_row["promise_date"] + timedelta(days=crm_row["window_end_days"])
    if processed_date > deadline:
        result["label"] = "delayed"
        result["reason"] = reason_delayed(
            refunded_amount, crm_row["promise_date"], crm_row["window_end_days"], processed_date
        )
        return result

    # Rule 8: amount matches and it's within the window -> clean
    result["label"] = "clean_match"
    return result


def build_orphan_results(crm_rows, gateway_by_order):
    """Bidirectional check: scans the GATEWAY side for order_nums that have
    no corresponding CRM promise at all -- money that moved with no
    approval record behind it. These have no promise_id, so each gets a
    synthetic one (ORPHAN-<order_num>); ground_truth.csv is written to use
    the exact same convention so scoring can look these cases up like any
    other.
    """
    promised_order_nums = {row["order_num"] for row in crm_rows}
    orphans = []
    for order_num, rows in gateway_by_order.items():
        if order_num in promised_order_nums:
            continue
        total = sum(r["refunded_amount"] for r in rows)
        latest_date = max(r["refund_processed_date"] for r in rows)
        orphans.append({
            "promise_id": f"ORPHAN-{order_num}",
            "order_id": rows[0]["order_id"],
            "promised_amount": None,
            "refunded_amount": total,
            "reason": reason_orphan_refund(rows[0]["order_id"], rows, total),
            "agent_notes": "",  # no CRM record exists, so there's no note to have logged
            "promise_date": None,
            "refund_processed_date": latest_date,
            "window_start_days": None,
            "window_end_days": None,
            "label": "orphan_refund",
        })
    return orphans


def run_pipeline(crm_path: Path, gateway_path: Path):
    crm_rows = load_crm_promises(crm_path)
    gateway_by_order = load_gateway_refunds(gateway_path)
    results = [classify_case(row, gateway_by_order) for row in crm_rows]
    results.extend(build_orphan_results(crm_rows, gateway_by_order))
    return results


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


def score_against_ground_truth(results, ground_truth_path: Path):
    truth = load_ground_truth(ground_truth_path)

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
        "labels": ALL_LABELS,
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
    n_orphan = counts.get("orphan_refund", 0)
    n_crm = total - n_orphan
    print(f"Total cases reviewed: {total} ({n_crm} CRM-promised + {n_orphan} gateway-only orphans)\n")

    clean = counts.get("clean_match", 0)
    print(f"Clean match rate: {clean}/{total} ({clean / total * 100:.1f}%)\n")

    print("Breakdown by category:")
    for label in ALL_LABELS:
        n = counts.get(label, 0)
        print(f"  {label:<15} {n:>3}  ({n / total * 100:5.1f}%)")
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
        # Within a severity tier, order by how serious the *label itself*
        # is (orphan/double/over rank above delayed) -- see LABEL_RANK.
        for r in sorted(cases, key=lambda x: (LABEL_RANK.get(x["label"], 99), x["order_id"])):
            fallback_tag = " [AI FALLBACK]" if r.get("ai_fallback") else ""
            confidence = r.get("confidence")
            conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
            print(f"  [{r['label'].upper():<14}] {r['promise_id']} ({r['order_id']}){fallback_tag}")
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
    # Column width must clear the longest label ("unknown_pattern", 15
    # chars) with room to spare, or long headers run together with no gap.
    col_width = max(len(l) for l in score["labels"]) + 2
    header = " " * 16 + "".join(f"{l:>{col_width}}" for l in score["labels"])
    print(header)
    for true_label in score["labels"]:
        if true_label not in score["confusion"]:
            continue  # skip label rows that never occur in this dataset's ground truth
        row = score["confusion"][true_label]
        line = f"{true_label:<16}" + "".join(f"{row.get(pred, 0):>{col_width}}" for pred in score["labels"])
        print(line)
    print()

    if score["mismatches"]:
        print(f"Misclassified cases ({len(score['mismatches'])}):")
        for promise_id, order_id, true_label, predicted_label in score["mismatches"]:
            print(f"  {promise_id} ({order_id}): true={true_label} predicted={predicted_label}")
    else:
        print("No misclassifications \U0001F389")
    print()


def run_dataset(name: str):
    cfg = DATASETS[name]
    results = run_pipeline(cfg["crm"], cfg["gateway"])

    # AI triage runs strictly after classification, on every exception
    # (everything but clean_match), and only ever adds severity/
    # justification fields -- see ai_triage.py.
    triage_summary = ai_triage.apply_ai_triage(results)

    # Ground-truth scoring reads only the deterministic `label` field
    # above; it never sees or is affected by the AI severities.
    score = score_against_ground_truth(results, cfg["ground_truth"])

    print("#" * 72)
    print(f"# DATASET: {cfg['display_name']}")
    print("#" * 72 + "\n")
    print_report(results, score, triage_summary)

    return score


def main():
    parser = argparse.ArgumentParser(description="Refund reconciliation pipeline")
    parser.add_argument(
        "--dataset", choices=["main", "holdout", "both"], default="main",
        help="Which batch to run: the main synthetic dataset, the adversarial "
             "hold-out batch, or both (scored separately). Default: main.",
    )
    args = parser.parse_args()

    names = ["main", "holdout"] if args.dataset == "both" else [args.dataset]
    scores = {name: run_dataset(name) for name in names}

    if len(names) > 1:
        print("=" * 72)
        print("ACCURACY SUMMARY (each dataset scored independently)")
        print("=" * 72)
        for name in names:
            s = scores[name]
            print(f"  {DATASETS[name]['display_name']:<30} {s['correct']}/{s['total']} ({s['accuracy'] * 100:.1f}%)")
        print()


if __name__ == "__main__":
    main()
