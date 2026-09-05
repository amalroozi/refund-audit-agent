#!/usr/bin/env python3
"""
generate_holdout_data.py
-------------------------
Generates a small, hand-crafted ADVERSARIAL batch that exercises patterns
the original generate_synthetic_data.py never produced: double refunds,
gateway-only "orphan" refunds with no CRM promise behind them, over-refunds,
refunds that cross a calendar month boundary, and a few genuinely ambiguous
cases. This is a held-out regression/stress test for reconcile.py's
classifier -- it should NOT need special-casing to pass; if it does, that's
a bug the main synthetic dataset was too easy to expose.

Output:
    ./data/holdout_crm.csv           -- same schema as crm_refund_promises.csv
    ./data/holdout_gateway.csv       -- same schema as gateway_refunds.csv
    ./data/holdout_ground_truth.csv  -- same schema as ground_truth.csv

Run reconcile.py --dataset holdout (or --dataset both) to score against it.
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 7
rng = random.Random(SEED)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Base numbering kept well clear of the main dataset's 100000-165000 range,
# and given its own letter prefix, so the two batches never collide even
# if their CSVs were ever accidentally concatenated.
BASE = 600000

AGENT_NOTES_POOL = [
    "Refund approved due to defective item",
    "Approved refund - order cancelled before shipment",
    "Customer escalation - refund approved to close ticket",
    "Refund approved per return policy after inspection",
    "",  # some cases still carry no note, same as the main dataset
]


def format_amount_noisy(amount: float) -> str:
    style = rng.random()
    if style < 0.6:
        return f"{amount:.2f}"
    elif style < 0.85:
        return f"{amount:,.2f}"
    else:
        return f"INR {amount:,.2f}"


def format_date_noisy(d: date) -> str:
    style = rng.random()
    if style < 0.7:
        return d.strftime("%Y-%m-%d")
    elif style < 0.88:
        return d.strftime("%d/%m/%Y")
    else:
        return d.strftime("%d-%b-%Y")


class CaseBuilder:
    """Accumulates rows for the three output CSVs as cases are emitted."""

    def __init__(self):
        self.crm_rows = []
        self.gateway_rows = []
        self.ground_truth_rows = []
        self.h = BASE
        self.case_no = 0

    def _next_ids(self):
        self.h += 1
        crm_order_id = f"HORD{self.h:06d}"
        gateway_order_id = f"horder_{self.h}"
        order_num = str(self.h)  # what extract_order_number() will recover from either id
        promise_id = f"HPRM-{self.h}"
        return order_num, crm_order_id, gateway_order_id, promise_id

    def emit(self, promised_amount, promise_date, window_str, agent_notes,
              gateway_txns, true_label, true_refunded_amount):
        """gateway_txns: list of (amount, processed_date) tuples; empty for no_refund.
        Adds one CRM promise row, zero-or-more gateway rows, and one
        ground_truth row using the SAME promise_id/order_id join keys
        reconcile.py will independently derive -- ground truth is recorded
        here, not inferred from the classifier's own output."""
        self.case_no += 1
        order_num, crm_order_id, gateway_order_id, promise_id = self._next_ids()
        customer_id = f"HCUST{rng.randint(10000, 99999)}"

        self.crm_rows.append({
            "promise_id": promise_id,
            "customer_id": customer_id,
            "order_id": crm_order_id,
            "promised_amount": format_amount_noisy(promised_amount),
            "promise_date": format_date_noisy(promise_date),
            "promised_window_days": window_str,
            "agent_notes": agent_notes,
        })

        refund_txn_ids = []
        latest_date = None
        for i, (amount, processed_date) in enumerate(gateway_txns):
            txn_id = f"hrtxn_{self.h}_{i}"
            refund_txn_ids.append(txn_id)
            gw_ref = "HGWREF" + "".join(rng.choice("0123456789ABCDEF") for _ in range(8))
            self.gateway_rows.append({
                "refund_txn_id": txn_id,
                "order_id": gateway_order_id,
                "refunded_amount": format_amount_noisy(amount),
                "refund_processed_date": format_date_noisy(processed_date),
                "gateway_reference": gw_ref,
            })
            if latest_date is None or processed_date > latest_date:
                latest_date = processed_date

        window_end_days = int(window_str.split("-")[1])
        self.ground_truth_rows.append({
            "case_no": self.case_no,
            "promise_id": promise_id,
            "refund_txn_id": ";".join(refund_txn_ids),
            "crm_order_id": crm_order_id,
            "gateway_order_id": gateway_order_id,
            "label": true_label,
            "promised_amount": f"{promised_amount:.2f}",
            "true_refunded_amount": f"{true_refunded_amount:.2f}" if true_refunded_amount is not None else "",
            "promise_date": promise_date.isoformat(),
            "window_end_days": window_end_days,
            "refund_processed_date": latest_date.isoformat() if latest_date else "",
        })

    def emit_orphan(self, gateway_txns, true_label="orphan_refund"):
        """A gateway-only case: NO CRM promise row at all. reconcile.py
        discovers these via its bidirectional scan and must assign them the
        same synthetic 'ORPHAN-<order_num>' key used here, so scoring can
        look them up like any other case."""
        self.h += 1
        gateway_order_id = f"horder_{self.h}"
        order_num = str(self.h)

        refund_txn_ids = []
        total = 0.0
        latest_date = None
        for i, (amount, processed_date) in enumerate(gateway_txns):
            txn_id = f"hrtxn_{self.h}_{i}"
            refund_txn_ids.append(txn_id)
            gw_ref = "HGWREF" + "".join(rng.choice("0123456789ABCDEF") for _ in range(8))
            self.gateway_rows.append({
                "refund_txn_id": txn_id,
                "order_id": gateway_order_id,
                "refunded_amount": format_amount_noisy(amount),
                "refund_processed_date": format_date_noisy(processed_date),
                "gateway_reference": gw_ref,
            })
            total += amount
            if latest_date is None or processed_date > latest_date:
                latest_date = processed_date

        self.case_no += 1
        self.ground_truth_rows.append({
            "case_no": self.case_no,
            "promise_id": f"ORPHAN-{order_num}",
            "refund_txn_id": ";".join(refund_txn_ids),
            "crm_order_id": "",
            "gateway_order_id": gateway_order_id,
            "label": true_label,
            "promised_amount": "",
            "true_refunded_amount": f"{total:.2f}",
            "promise_date": "",
            "window_end_days": "",
            "refund_processed_date": latest_date.isoformat() if latest_date else "",
        })


b = CaseBuilder()

# --------------------------------------------------------------------------
# 1) Baseline cases (5 clean_match, 3 partial, 3 no_refund) -- a control
#    group proving the classifier still handles ordinary cases correctly
#    even inside a batch designed to stress it.
# --------------------------------------------------------------------------

for amount in [999.0, 1999.0, 2999.0, 4999.0, 7999.0]:
    pdate = date(2026, 7, 10)
    b.emit(amount, pdate, "5-7", rng.choice(AGENT_NOTES_POOL),
           [(amount, pdate + timedelta(days=3))], "clean_match", amount)

for amount in [1999.0, 4999.0, 9999.0]:
    pdate = date(2026, 7, 12)
    refunded = round(amount * 0.97, 2)  # same 3% fee pattern as the main dataset
    b.emit(amount, pdate, "5-8", "Approved partial refund per manager discretion",
           [(refunded, pdate + timedelta(days=4))], "partial", refunded)

for amount in [799.0, 2499.0, 5999.0]:
    pdate = date(2026, 7, 8)
    b.emit(amount, pdate, "5-7", rng.choice(AGENT_NOTES_POOL), [], "no_refund", None)

# --------------------------------------------------------------------------
# 2) double_refund (6) -- more than one gateway transaction for the same
#    order. Half look like an accidental full duplicate payout; half look
#    like the promised amount split into two partial payouts. Either way
#    the classifier must flag both rows rather than silently summing them
#    into what would otherwise look like a clean (or over) match.
# --------------------------------------------------------------------------

for amount in [1499.0, 3499.0, 6999.0]:  # duplicate full payout
    pdate = date(2026, 6, 20)
    b.emit(amount, pdate, "5-7", "Refund approved due to defective item",
           [(amount, pdate + timedelta(days=2)), (amount, pdate + timedelta(days=9))],
           "double_refund", amount * 2)

for amount in [2000.0, 5000.0, 9000.0]:  # split into two partial payouts
    pdate = date(2026, 6, 22)
    half = round(amount / 2, 2)
    b.emit(amount, pdate, "5-7", "Approved refund - order cancelled before shipment",
           [(half, pdate + timedelta(days=3)), (half, pdate + timedelta(days=4))],
           "double_refund", half * 2)

# --------------------------------------------------------------------------
# 3) over_refund (6) -- gateway paid out MORE than CRM promised, clearly
#    beyond the ambiguous band (>2%): agent/gateway error refunding extra
#    (e.g. shipping fee, or the item twice).
# --------------------------------------------------------------------------

OVER_REFUND_PAIRS = [
    (999.0, 1099.0),    # +10%
    (1999.0, 2299.0),   # +15%
    (2999.0, 3299.0),   # +10%
    (4999.0, 5999.0),   # +20%
    (999.0, 1998.0),    # refunded the item twice
    (7999.0, 8399.0),   # +5%
]
for promised, refunded in OVER_REFUND_PAIRS:
    pdate = date(2026, 6, 25)
    b.emit(promised, pdate, "5-7", rng.choice(AGENT_NOTES_POOL),
           [(refunded, pdate + timedelta(days=3))], "over_refund", refunded)

# --------------------------------------------------------------------------
# 4) orphan_refund (6) -- gateway paid out with NO CRM promise behind it at
#    all. This is what reconcile.py's bidirectional scan (over gateway
#    order_ids CRM never mentions) must catch.
# --------------------------------------------------------------------------

for amount in [499.0, 1299.0, 2599.0, 3999.0, 6499.0, 8999.0]:
    pdate = date(2026, 7, 1) + timedelta(days=rng.randint(0, 40))
    b.emit_orphan([(amount, pdate)])

# --------------------------------------------------------------------------
# 5) Month-boundary crossings (4 clean_match, 4 delayed) -- the promise is
#    made right before month-end so the refund (whether on time or late)
#    lands in the following calendar month. Pure day-count arithmetic
#    (promise_date + timedelta(days=N)) is correct across month/year
#    boundaries, but this is exactly the kind of edge a hand-rolled date
#    parser could get wrong -- so it's tested explicitly rather than
#    trusted implicitly.
# --------------------------------------------------------------------------

MONTH_BOUNDARY_CLEAN = [
    (date(2026, 5, 30), "5-7", 3),   # May -> June, within window
    (date(2026, 6, 29), "3-5", 4),   # June -> July
    (date(2026, 7, 31), "5-7", 5),   # July -> August
    (date(2026, 2, 26), "5-7", 6),   # February -> March
]
for pdate, window, offset in MONTH_BOUNDARY_CLEAN:
    amount = rng.choice([999.0, 1999.0, 2999.0, 4999.0])
    b.emit(amount, pdate, window, rng.choice(AGENT_NOTES_POOL),
           [(amount, pdate + timedelta(days=offset))], "clean_match", amount)

MONTH_BOUNDARY_DELAYED = [
    (date(2026, 4, 28), "3-5", 7),    # April -> May, 2 days late
    (date(2026, 5, 29), "5-7", 12),   # May -> June, 5 days late
    (date(2026, 6, 27), "3-5", 9),    # June -> July, 4 days late
    (date(2026, 3, 30), "5-7", 14),   # March -> April, 7 days late
]
for pdate, window, offset in MONTH_BOUNDARY_DELAYED:
    amount = rng.choice([799.0, 2499.0, 3999.0, 5999.0])
    b.emit(amount, pdate, window, rng.choice(AGENT_NOTES_POOL),
           [(amount, pdate + timedelta(days=offset))], "delayed", amount)

# --------------------------------------------------------------------------
# 6) unknown_pattern (3) -- genuinely ambiguous cases: the refunded amount
#    is off by more than rounding/formatting noise but less than the
#    smallest deliberate deduction pattern anywhere else in the data
#    (the 3% fee). A classifier that is forced to pick clean/partial/over
#    here is guessing; it should abstain instead.
# --------------------------------------------------------------------------

AMBIGUOUS_CASES = [
    (2999.0, 2949.0),  # short by ~1.67%
    (1499.0, 1470.0),  # short by ~1.93%
    (4999.0, 5074.0),  # over by ~1.50%
]
for promised, refunded in AMBIGUOUS_CASES:
    pdate = date(2026, 7, 15)
    b.emit(promised, pdate, "5-7", "",  # deliberately no note -- nothing to disambiguate with
           [(refunded, pdate + timedelta(days=3))], "unknown_pattern", refunded)


# --------------------------------------------------------------------------
# Write the three CSVs
# --------------------------------------------------------------------------

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


write_csv(DATA_DIR / "holdout_crm.csv", b.crm_rows,
          ["promise_id", "customer_id", "order_id", "promised_amount",
           "promise_date", "promised_window_days", "agent_notes"])

write_csv(DATA_DIR / "holdout_gateway.csv", b.gateway_rows,
          ["refund_txn_id", "order_id", "refunded_amount",
           "refund_processed_date", "gateway_reference"])

write_csv(DATA_DIR / "holdout_ground_truth.csv", b.ground_truth_rows,
          ["case_no", "promise_id", "refund_txn_id", "crm_order_id", "gateway_order_id",
           "label", "promised_amount", "true_refunded_amount", "promise_date",
           "window_end_days", "refund_processed_date"])

print(f"\nDone. {b.case_no} cases total ({len(b.crm_rows)} CRM promises, "
      f"{len(b.gateway_rows)} gateway transactions).")
print(
    "Composition: 5 clean_match, 3 partial, 3 no_refund (baseline) + "
    "6 double_refund, 6 over_refund, 6 orphan_refund, "
    "4 month-boundary clean_match, 4 month-boundary delayed, 3 unknown_pattern."
)
