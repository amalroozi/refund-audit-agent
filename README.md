# refund-audit-agent

**Razorpay AI Buildathon 2026 | Track 04: AI Finance Controller | Direction: Multi-source reconciliation**

A reconciliation agent that matches refunds a support team **promised** against refunds that actually **cleared** in the payment gateway, and reports every discrepancy with a specific, actionable reason.

---

## Results

| Metric | Value |
|---|---|
| Records processed | 65 (main batch), 40 (second batch) |
| Clean match rate | 38/65 (58.5%) |
| Exceptions surfaced | 27 |
| Classification accuracy vs ground truth | 65/65 (100%) |
| AI triage severity split | 15 critical, 7 explained, 5 needs review |
| AI fallbacks | 4/27, all from missing agent notes |

Every one of the 65 records reaches a final, defensible state. Nothing is skipped, and nothing is force-matched to make the numbers look better.

Read the "Honest limitations" section before interpreting the 100% figure. It is a consistency check, not proof of generalisation, and I explain exactly why below.

---

## The problem

You buy something online. It goes wrong. Support says "refund initiated, 5 to 7 working days." Then nothing arrives, and you have no way to tell whether the payment failed, whether it is just slow, or whether anyone ever pressed the button.

This is not a hypothetical hackathon scenario. Razorpay's own team has written that "Why does a refund take time?" is the most visited post on their blog, still answered daily by a canned support macro across piles of refund tickets, and one of their product leads called refunds a "99% problem."

The root cause is that **two independent systems both believe they know what happened**, and nobody reconciles them:

1. **The CRM / support tool** records what an agent *promised*: an approved refund, an amount, a turnaround window, and a free-text note explaining why.
2. **The payment gateway** records what actually *moved*: a real transaction, a real amount, a real date.

When those two disagree, a customer is left waiting for money that may never have been sent. Finding those cases is still manual work.

---

## What this agent does

For every refund promise in the CRM, the agent determines what actually happened in the gateway, and assigns one of eight outcomes:

| Label | Meaning |
|---|---|
| `clean_match` | Refunded correctly, in full, inside the promised window |
| `partial` | Refund happened but for meaningfully less than promised |
| `delayed` | Correct amount, but processed after the promised deadline |
| `no_refund` | Promised, but no gateway transaction exists at all |
| `over_refund` | Refunded more than was promised |
| `double_refund` | Multiple gateway transactions against the same order |
| `orphan_refund` | Money left the gateway with **no CRM promise behind it** |
| `unknown_pattern` | Does not fit any known rule; the agent abstains rather than guessing |

It then produces a prioritised exception list where every unresolved case carries a specific reason with real figures, not a generic "no match found."

---

## The data problem, and why a naive join fails

The two sources were built by different teams, so they disagree on formatting in ways that silently break naive reconciliation:

| Field | CRM format | Gateway format |
|---|---|---|
| Order ID | `ORD100027` | `order_100027` |
| Amount | `1999.00` / `1,999.00` / `INR 1,999.00` | same three variants, independently chosen |
| Date | `2026-07-14` / `14/07/2026` / `14-Jul-2026` | same three variants |

A direct `join on order_id` matches **zero rows**, silently, and reports a clean run. The agent extracts the numeric portion of the order ID from both sides and joins on that instead, and parses amounts and dates defensively before any comparison happens.

About 12% of CRM rows also have no agent note logged, simulating a field that was never actually enforced. The agent must not depend on notes always being present.

---

## Architecture

```
crm_refund_promises.csv          gateway_refunds.csv
        |                                 |
        +------------- normalise ---------+
        |     (ID extraction, amount parsing, date parsing)
        v
   DETERMINISTIC CLASSIFIER          <-- no LLM, fully reproducible
   forward pass:  every promise -> matching gateway txn?
   reverse pass:  every gateway txn -> matching promise?
        |
        +--> clean_match  (38)  ------------------> done
        |
        +--> exceptions   (27)
                   |
                   v
             AI TRIAGE LAYER              <-- LLM, exceptions only
             reads agent_notes + figures
             assigns severity + justification + confidence
             CANNOT write `label`
                   |
                   v
             PRIORITISED REPORT
             critical -> needs_review -> explained

   GROUND TRUTH SCORING  <-- entirely separate, reads output only,
                             never feeds back into classification
```

### Why the classifier has no AI in it

This was a deliberate decision, not an omission.

Comparing two amounts is arithmetic. Checking whether a date falls past a deadline is arithmetic. An LLM doing this work would be slower, non-reproducible across runs, and occasionally wrong at things a tolerance check is never wrong at. The track's bar demands *measured accuracy*, and you cannot honestly claim measured accuracy on a component that returns different answers on different runs.

So the classifier is pure deterministic logic, and the accuracy figures above are reproducible byte for byte on every run.

### Bidirectional reconciliation

An earlier version of this agent only walked **one direction**: it took each CRM promise and looked for a matching gateway transaction. That version could never detect a gateway refund with **no promise behind it**, which is money leaving the business with no approval trail, and arguably the highest severity finding in this entire workflow.

A reconciliation that only walks one side is not a reconciliation. It is a promise verifier. The reverse pass was added to fix this, and orphan records are surfaced with synthetic `ORPHAN-<order_num>` identifiers so they appear in the exception list alongside everything else.

### One threshold worth explaining

Amount matching tolerance is `max(1% of promised, 1 rupee)`.

This is deliberately **tighter** than the 3% fee deduction present in the partial-refund cases. If the tolerance were looser, genuine partial refunds would be silently absorbed as "close enough" and disappear from the exception list, inflating the match rate while hiding real problems. The threshold was chosen against the failure mode, not to maximise the headline number.

---

## Where AI genuinely earns its place

The LLM does exactly one job: reading the **free-text agent note** against the numbers, and assigning a severity so a human knows what to look at first.

This is the case that justifies its existence. Consider two partial refunds that are numerically identical, both 3% short:

**PRM-200013** — note reads *"Approved partial refund per manager discretion."*
Severity assigned: **explained** (0.95 confidence). Someone decided this. The discrepancy is documented.

**PRM-200001** — note offers no explanation for the shortfall.
Severity assigned: **critical** (0.85 confidence). A customer quietly received less than they were told, with nothing on record justifying it.

Same numbers. Opposite situations. No rule can separate them, because the difference lives entirely in a sentence a human typed. That is a semantic judgment task, and it is the only part of this system where a language model is the right tool.

### How the AI layer is bounded

| Constraint | Implementation |
|---|---|
| Scope | Runs on exceptions only, never on the 38 clean matches |
| Authority | Can write `severity`, `justification`, `confidence`. **Cannot write `label`.** |
| Output | Constrained to a strict JSON schema, validated on parse |
| Blank note | Falls back to `needs_review` before any API call is made |
| API failure | Retries, then falls back to `needs_review`, logged individually |
| Low confidence | Below 0.5 falls back to `needs_review` |
| Quota exhaustion | Circuit breaker halts the batch instead of retrying calls that cannot succeed |

The safe failure mode is **extra human attention**, never a false all clear. A total AI outage degrades this system into "every exception needs review," which is exactly where a finance team was before the agent existed. It never degrades into silently marking problems as fine.

---

## What broke, and what I learned

### The same bug on two different providers

On Google Gemini, every triage response came back truncated mid-JSON. Reading the actual usage payloads showed the model was spending 300 to 400 tokens on internal reasoning **inside the same output budget** as the visible answer, so the JSON never finished.

I raised the budget, fixed it, and later switched providers to Groq for free-tier quota reasons. On `openai/gpt-oss-20b` the failure came back wearing completely different clothes: a `400 json_validate_failed` with an empty `failed_generation` string. Groq's usage data showed 177 of 210 completion tokens consumed by hidden reasoning before the answer began.

Two vendors, two models, two entirely different error signatures, one identical root cause. That is when I stopped treating it as a bug and started treating it as a **property of reasoning models**: hidden reasoning shares the visible output budget, and any code calling them needs headroom plus a fallback path. Fixed with `MAX_OUTPUT_TOKENS = 800` and `reasoning_effort="low"`.

### The fallback path that looked like a failure

An earlier full run returned `27/27 needs_review`. It looked like the AI layer was doing nothing. It was actually the fallback firing correctly against a missing API key. I would rather find that out from a clean degradation than from a system that pretends to work.

### Rate limits are not one thing

Groq returns HTTP 429 for two structurally different situations. A 429 with `x-ratelimit-remaining-requests == 0` means the **daily** quota is gone and retrying is pointless until reset. A 429 with requests still remaining is a transient **per-minute** limit worth retrying with the server's own `retry-after` value.

Blind exponential backoff treats these identically and wastes minutes hammering a wall. The circuit breaker reads the actual headers and distinguishes them, so daily exhaustion halts the batch immediately instead of retrying 27 times.

---

## Honest limitations

I would rather state these than have them found.

**1. The 100% accuracy is a consistency check, not a generalisation result.**
I wrote the synthetic data generator and I wrote the classifier rules. The ground truth labels come from patterns I authored. That number proves the pipeline is internally consistent and correctly implemented. It does **not** prove the approach holds on refund mismatches whose failure modes I did not design. The second batch adds harder patterns (double refunds, orphan rows, over-refunds, month-boundary crossings, ambiguous cases) but I authored those too, so the same caveat applies. A genuine held-out evaluation on data I did not create is the first thing I would do next.

**2. The LLM is unreliable on figures.**
Several justifications render rupee amounts as dollars, and some restate figures imprecisely. This is a real weakness of the language layer, and it is exactly why the AI is architecturally forbidden from writing labels or amounts. Every number a user acts on comes from the deterministic layer. The LLM only ranks and explains.

**3. `unknown_pattern` is a rule, not true abstention.**
It fires on a defined ambiguity band rather than on genuine model uncertainty. Real abstention would require calibrated confidence on the matching decision itself.

**4. Throughput is demonstrated at 65 and 40 records.**
The deterministic core is O(n) with a hash join and should scale well past this, but I have not benchmarked it at production volume.

---

## Running it

```bash
# install
pip3 install -r requirements.txt

# generate the synthetic datasets (seeded, reproducible)
python3 generate_synthetic_data.py
python3 generate_holdout_data.py

# run reconciliation
python3 reconcile.py --dataset main
python3 reconcile.py --dataset holdout
python3 reconcile.py --dataset both
```

The AI triage layer requires a Groq API key. Copy `.env.example` to `.env` and set `GROQ_API_KEY`. **Without a key the pipeline still runs end to end**, with all exceptions falling back to `needs_review` and every fallback logged. The deterministic classification and accuracy scoring are entirely unaffected.

---

## Repository structure

| File | Purpose |
|---|---|
| `generate_synthetic_data.py` | Builds the main 65-case batch with deliberate, realistic mess |
| `generate_holdout_data.py` | Builds the 40-case batch with harder adversarial patterns |
| `reconcile.py` | Normalisation, bidirectional matching, classification, scoring, reporting |
| `ai_triage.py` | LLM severity layer, bounded, with fallbacks and circuit breaker |
| `data/` | Generated CSVs plus separate ground truth files |

Ground truth is written to its own file and is never visible to the classifier. The scoring function reads the classifier's output only, and never feeds back into it.

---

## Design principles

1. **Deterministic where arithmetic suffices. AI only where language understanding is required.**
2. **The AI can prioritise but never decide.** It cannot change a label.
3. **A longer, well explained exception list beats a suspiciously perfect match rate.** Nothing is force-matched.
4. **Failures degrade toward human attention**, never toward a false all clear.
5. **State the limitation before someone else finds it.**