#!/usr/bin/env python3
"""
generate_synthetic_data.py
---------------------------
Generates synthetic data for a refund-reconciliation hackathon project.

It simulates two independent, real-world systems that a fintech company
would actually have to reconcile against each other:

  1. crm_refund_promises.csv  -> what a support/CRM agent PROMISED the
     customer (approved refund, amount, expected turnaround window).

  2. gateway_refunds.csv      -> what ACTUALLY cleared in the payment
     gateway (the source of truth for money movement).

Both files are joined by `order_id`, but -- exactly like real systems built
by different teams -- each side uses its own ID format, its own date
format, and its own way of writing currency. On top of that, ~40% of the
underlying refund cases are deliberately "broken" in one of three realistic
ways (partial refund, delayed refund, refund never happened) so that a
reconciliation agent has something non-trivial to detect.

A third file, ground_truth.csv, records the TRUE label for every single
case (clean_match / partial / delayed / no_refund) plus the true refunded
amount. This file is intentionally kept OUT of the two "real" datasets --
it exists purely so you can score your own agent's precision/recall later.

Usage:
    python generate_synthetic_data.py

Output:
    ./data/crm_refund_promises.csv
    ./data/gateway_refunds.csv
    ./data/ground_truth.csv
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 42                      # reproducible runs -> same data every time
random.seed(SEED)

N_CASES = 65                   # total underlying refund cases to simulate
TODAY = date(2026, 9, 5)       # fixed "today" so the dataset is stable

# Case-mix counts (chosen to land inside the ranges the spec asked for):
#   clean_match : 38/65 = 58.5%   (target ~55-60%)
#   partial     : 10/65 = 15.4%   (target ~15%)
#   delayed     :  7/65 = 10.8%   (target ~10%)
#   no_refund   : 10/65 = 15.4%   (target ~15%)
N_CLEAN, N_PARTIAL, N_DELAYED, N_NO_REFUND = 38, 10, 7, 10
assert N_CLEAN + N_PARTIAL + N_DELAYED + N_NO_REFUND == N_CASES

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# A consistent, plausible "fee" applied to every partial-refund case.
# Modeling something like a payment-gateway processing fee that gets
# netted off before the customer sees the money -- NOT random noise.
PARTIAL_REFUND_FEE_RATE = 0.03  # gateway keeps 3%, customer gets the rest

AGENT_NOTE_TEMPLATES = [
    "Refund approved due to defective item",
    "Customer reported item not delivered, refund approved",
    "Approved refund - order cancelled before shipment",
    "Wrong item received, refund approved as goodwill",
    "Refund approved per return policy after inspection",
    "Duplicate charge, refund approved by billing team",
    "Approved partial refund per manager discretion",
    "Customer escalation - refund approved to close ticket",
    "Refund approved, item damaged in transit",
    "Approved refund - service not rendered as promised",
]

CUSTOMER_PREFIXES = ["CUST", "USR", "CID"]


# --------------------------------------------------------------------------
# Small formatting helpers -- these are what inject the "messy real world"
# noise: different ID schemes, different date formats, different currency
# string formats between the two systems.
# --------------------------------------------------------------------------

def crm_order_id(n: int) -> str:
    """CRM's own order numbering: zero-padded, e.g. ORD100027."""
    return f"ORD{100000 + n:06d}"


def gateway_order_id(n: int) -> str:
    """Gateway's numbering for the SAME order: different prefix, no padding,
    lowercase -- a totally different formatting convention, which is the
    norm when two different vendors/systems reference the same entity."""
    return f"order_{100000 + n}"


def make_promise_id(n: int) -> str:
    return f"PRM-{200000 + n}"


def make_refund_txn_id(n: int) -> str:
    return f"rtxn_{300000 + n:06d}"


def make_gateway_reference(rng: random.Random) -> str:
    """Gateway-internal reference number -- unrelated ID scheme to
    promise_id on purpose (different systems rarely share an ID space)."""
    return "GWREF" + "".join(rng.choice("0123456789ABCDEF") for _ in range(8))


def random_customer_id(rng: random.Random) -> str:
    return f"{rng.choice(CUSTOMER_PREFIXES)}{rng.randint(10000, 99999)}"


def format_date_noisy(d: date, rng: random.Random) -> str:
    """Most rows use ISO format, but a slice of rows use a different (still
    valid, still human-plausible) date format -- e.g. exports coming out of
    different tools/locales that were pasted into the same CSV over time."""
    style = rng.random()
    if style < 0.70:
        return d.strftime("%Y-%m-%d")            # 2026-07-14
    elif style < 0.88:
        return d.strftime("%d/%m/%Y")             # 14/07/2026
    else:
        return d.strftime("%d-%b-%Y")             # 14-Jul-2026


def format_amount_noisy(amount: float, rng: random.Random) -> str:
    """Most amounts are plain decimals, but some carry thousand separators
    or a currency prefix -- typical of exports from different finance
    tools / manual copy-paste into a CRM notes/amount field."""
    style = rng.random()
    if style < 0.65:
        return f"{amount:.2f}"                    # 1999.00
    elif style < 0.85:
        return f"{amount:,.2f}"                    # 1,999.00
    else:
        return f"INR {amount:,.2f}"                 # INR 1,999.00


def random_amount(rng: random.Random) -> float:
    """A plausible e-commerce refund amount (not uniformly random cents)."""
    base = rng.choice([199, 299, 499, 799, 999, 1499, 1999, 2499, 2999,
                        3499, 4999, 5999, 7999, 9999, 12999, 14999])
    jitter = rng.choice([0, 0, 0, 0.5, 0.99, 1.0])  # some round, some not
    return round(base + jitter, 2)


def parse_window(rng: random.Random):
    """Returns (window_str, window_end_days) e.g. ("5-7", 7)."""
    start = rng.choice([3, 5, 7])
    end = start + rng.choice([2, 3, 4])
    return f"{start}-{end}", end


# --------------------------------------------------------------------------
# Build the underlying cases
# --------------------------------------------------------------------------

def build_cases():
    rng = random.Random(SEED)
    labels = (
        ["clean_match"] * N_CLEAN
        + ["partial"] * N_PARTIAL
        + ["delayed"] * N_DELAYED
        + ["no_refund"] * N_NO_REFUND
    )
    rng.shuffle(labels)

    cases = []
    for i, label in enumerate(labels, start=1):
        promised_amount = random_amount(rng)
        window_str, window_end = parse_window(rng)

        # Promise made sometime in the last 20-90 days so that even
        # "delayed" refunds have had time to land before TODAY.
        days_ago = rng.randint(20, 90)
        promise_date = TODAY - timedelta(days=days_ago)

        agent_notes = rng.choice(AGENT_NOTE_TEMPLATES)
        # ~12% of promises have no notes logged -- agents forget/skip this.
        if rng.random() < 0.12:
            agent_notes = ""

        case = {
            "case_no": i,
            "customer_id": random_customer_id(rng),
            "crm_order_id": crm_order_id(i),
            "gateway_order_id": gateway_order_id(i),
            "promise_id": make_promise_id(i),
            "refund_txn_id": make_refund_txn_id(i),
            "gateway_reference": make_gateway_reference(rng),
            "promised_amount": promised_amount,
            "promise_date": promise_date,
            "window_str": window_str,
            "window_end_days": window_end,
            "agent_notes": agent_notes,
            "label": label,
        }

        # Decide the refund outcome based on the label
        if label == "clean_match":
            refunded_amount = promised_amount
            processed_offset = rng.randint(1, window_end)  # inside window
            has_gateway_row = True

        elif label == "partial":
            fee = round(promised_amount * PARTIAL_REFUND_FEE_RATE, 2)
            refunded_amount = round(promised_amount - fee, 2)
            processed_offset = rng.randint(1, window_end)  # inside window
            has_gateway_row = True

        elif label == "delayed":
            refunded_amount = promised_amount
            # well past the promised window
            processed_offset = window_end + rng.randint(10, 25)
            has_gateway_row = True

        else:  # no_refund
            refunded_amount = None
            processed_offset = None
            has_gateway_row = False

        case["has_gateway_row"] = has_gateway_row
        case["refunded_amount"] = refunded_amount
        if has_gateway_row:
            case["refund_processed_date"] = promise_date + timedelta(days=processed_offset)
        else:
            case["refund_processed_date"] = None

        cases.append(case)

    return cases


# --------------------------------------------------------------------------
# Write the three CSVs
# --------------------------------------------------------------------------

def write_crm_csv(cases, rng):
    path = DATA_DIR / "crm_refund_promises.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "promise_id", "customer_id", "order_id", "promised_amount",
            "promise_date", "promised_window_days", "agent_notes",
        ])
        for c in cases:
            writer.writerow([
                c["promise_id"],
                c["customer_id"],
                c["crm_order_id"],
                format_amount_noisy(c["promised_amount"], rng),
                format_date_noisy(c["promise_date"], rng),
                c["window_str"],
                c["agent_notes"],
            ])
    print(f"wrote {path} ({len(cases)} rows)")


def write_gateway_csv(cases, rng):
    path = DATA_DIR / "gateway_refunds.csv"
    rows = [c for c in cases if c["has_gateway_row"]]
    # Gateway transactions don't arrive in promise order -- shuffle for realism.
    rows = rows[:]
    rng.shuffle(rows)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "refund_txn_id", "order_id", "refunded_amount",
            "refund_processed_date", "gateway_reference",
        ])
        for c in rows:
            writer.writerow([
                c["refund_txn_id"],
                c["gateway_order_id"],
                format_amount_noisy(c["refunded_amount"], rng),
                format_date_noisy(c["refund_processed_date"], rng),
                c["gateway_reference"],
            ])
    print(f"wrote {path} ({len(rows)} rows)")


def write_ground_truth_csv(cases):
    path = DATA_DIR / "ground_truth.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_no", "promise_id", "refund_txn_id",
            "crm_order_id", "gateway_order_id",
            "label", "promised_amount", "true_refunded_amount",
            "promise_date", "window_end_days", "refund_processed_date",
        ])
        for c in cases:
            writer.writerow([
                c["case_no"],
                c["promise_id"],
                c["refund_txn_id"] if c["has_gateway_row"] else "",
                c["crm_order_id"],
                c["gateway_order_id"],
                c["label"],
                f"{c['promised_amount']:.2f}",
                f"{c['refunded_amount']:.2f}" if c["refunded_amount"] is not None else "",
                c["promise_date"].isoformat(),
                c["window_end_days"],
                c["refund_processed_date"].isoformat() if c["refund_processed_date"] else "",
            ])
    print(f"wrote {path} ({len(cases)} rows)")


def main():
    cases = build_cases()
    fmt_rng = random.Random(SEED + 1)  # separate RNG stream just for formatting noise
    write_crm_csv(cases, fmt_rng)
    write_gateway_csv(cases, fmt_rng)
    write_ground_truth_csv(cases)

    print(
        "\nDone. "
        f"{N_CLEAN} clean_match, {N_PARTIAL} partial, "
        f"{N_DELAYED} delayed, {N_NO_REFUND} no_refund "
        f"({N_CASES} total cases)."
    )


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------
# What each mismatch type simulates, and why it's in here
# --------------------------------------------------------------------------
#
# 1. Different order_id formats between files (ORD100027 vs order_100027)
#    -> Simulates the CRM/support tool and the payment gateway being two
#       completely separate systems (built by different teams/vendors) that
#       both reference the same order but were never given a shared ID
#       schema. A naive `merge on order_id` will silently join nothing;
#       your agent needs to normalize/extract the identifying number (or
#       use a mapping table) before it can reconcile anything at all. This
#       is extremely common in real fintech stacks (CRM vs PSP vs ledger).
#
# 2. Currency formatting noise (1999.00 / 1,999.00 / INR 1,999.00)
#    -> Simulates amount fields exported from different tools/locales, or
#       typed by hand into a CRM note field. Forces the agent to parse
#       amounts robustly (strip currency symbols/commas) before comparing
#       values instead of assuming a clean float column.
#
# 3. Date formatting noise (2026-07-14 / 14/07/2026 / 14-Jul-2026)
#    -> Simulates rows coming from different export batches / regional
#       settings over the life of the CRM and gateway systems. Forces the
#       agent to parse dates flexibly before it can check whether a refund
#       landed inside the promised window.
#
# 4. Missing agent_notes (~12% blank)
#    -> Simulates support agents who forget to log a reason, or a required
#       field that wasn't actually enforced. Tests that the agent's
#       reasoning doesn't silently depend on notes always being present.
#
# 5. partial: refunded_amount < promised_amount (consistent 3% haircut)
#    -> Simulates a real, explainable discrepancy (e.g. a payment gateway
#       processing/convenience fee netted off before payout) rather than
#       random noise. A good agent should detect "refund happened, but for
#       less than promised" as its own category, distinct from "no refund
#       at all", and ideally infer/flag the consistent deduction pattern.
#
# 6. delayed: gateway refund lands 10-25 days after the promised window end
#    -> Simulates real operational delays (bank processing lag, manual
#       approval backlog, retried failed payouts). The refund DID happen
#       and for the full amount, but breaches the promised SLA -- a
#       reconciliation agent should flag this as an SLA violation, not a
#       clean match and not a missing refund.
#
# 7. no_refund: promise_id exists in CRM with zero matching gateway rows
#    -> Simulates the highest-severity failure mode: a customer was told
#       money is coming, but the payment was never actually initiated or
#       silently failed. These are the true "exceptions" the agent must
#       catch, since from the customer's perspective the promise was never
#       honored and no automatic system (other than reconciliation) would
#       ever surface this on its own.
#
# ground_truth.csv is deliberately excluded from both "real" files above --
# it exists only so you can compute your own agent's precision/recall
# against the labels it should have produced.
