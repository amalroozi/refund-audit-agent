#!/usr/bin/env python3
"""
ai_triage.py
------------
AI-assisted severity triage for refund reconciliation exceptions.

This module runs ONLY on cases the deterministic classifier in reconcile.py
already labeled partial / delayed / no_refund. It never sees, and cannot
influence, clean_match cases.

For each exception it asks Claude to read the numbers + the support agent's
free-text note and assign a *severity* -- a judgment call about how
worrying the discrepancy is -- completely separate from the deterministic
label. The label (what happened) is decided by rules; the severity (how
bad it looks) is decided by the model. The model can never overwrite the
label: this module only ever adds new fields (severity, justification,
confidence, ai_fallback) onto the result dict it's given.

If anything about the AI call is unreliable for a given case -- no note to
reason about, the API call fails after retries, the response isn't valid
JSON, or the model itself reports low confidence -- we fall back to
severity "needs_review" with a justification that says exactly which of
those triggered, and we log it. A run must never abort because the LLM had
a bad moment; every case always ends up with *some* severity.
"""

import json
import re
import time

MODEL = "claude-sonnet-4-6"  # explicitly requested for this triage step
MAX_TOKENS = 300
MAX_RETRIES = 2  # retries AFTER the first attempt -> up to 3 calls total
RETRY_BACKOFF_SECONDS = 1.5
CONFIDENCE_THRESHOLD = 0.5

VALID_SEVERITIES = {"explained", "needs_review", "critical"}

SYSTEM_PROMPT = """You are a triage assistant for a refund reconciliation system.

You are given ONE refund case that a rule-based system has ALREADY classified
(the label is final and you cannot change it). Your only job is to judge how
severe/worrying this exception is for a human operator, using the support
agent's note as context.

Severity definitions:
- explained: the agent note corroborates the discrepancy (e.g. the note
  mentions a partial or discretionary refund, and the gateway shows a
  shortfall consistent with that; or the note explains why a refund is
  delayed or was not issued).
- critical: the note contradicts what actually happened (e.g. it promises a
  full refund but the money is short or missing entirely), OR a large
  amount was promised with no refund at all and nothing in the note
  explains why.
- needs_review: anything ambiguous, or no note was logged at all.

Respond with STRICT JSON only. No markdown code fences, no prose before or
after it. Respond with exactly this shape:
{"severity": "explained|needs_review|critical", "justification": "<one sentence, must reference the agent note content when a note exists>", "confidence": 0.0}
"""


def _build_user_message(case: dict) -> str:
    """Renders the case's numbers/dates/note into a compact prompt. `case`
    is one reconcile.py result dict (see reconcile.classify_case)."""
    refunded = (
        f"{case['refunded_amount']:.2f}"
        if case["refunded_amount"] is not None
        else "NONE -- no gateway transaction at all"
    )
    processed = case["refund_processed_date"].isoformat() if case["refund_processed_date"] else "N/A"
    notes = case["agent_notes"].strip() if case["agent_notes"] else "(no note logged)"

    return (
        f"Deterministic label: {case['label']}\n"
        f"Promised amount: {case['promised_amount']:.2f}\n"
        f"Refunded amount: {refunded}\n"
        f"Promise date: {case['promise_date'].isoformat()}\n"
        f"Refund processed date: {processed}\n"
        f"Promised window: {case['window_start_days']}-{case['window_end_days']} days\n"
        f"Agent notes: {notes}\n"
    )


def _extract_json(raw_text: str) -> dict:
    """Parses the model's reply as JSON. Defensively strips markdown code
    fences in case the model adds them despite being told not to -- this
    is exactly the kind of thing that should degrade to a fallback, not
    crash the run, if it still can't be parsed afterward."""
    text = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _fallback(reason: str) -> dict:
    """The single place that produces a fallback result, so every trigger
    (missing note, API failure, bad JSON, low confidence) produces the
    same shape and gets logged the same way."""
    print(f"[ai_triage] FALLBACK -> needs_review: {reason}")
    return {
        "severity": "needs_review",
        "justification": reason,
        "confidence": None,
        "ai_fallback": True,
    }


def _call_model(client, case: dict) -> dict:
    """Calls Claude for one case, with up to MAX_RETRIES retries on
    transient failures. Raises the last exception if every attempt fails;
    non-retryable errors (auth, bad request) are raised immediately."""
    import anthropic  # imported lazily so the module still loads without the SDK installed

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_message(case)}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            return _extract_json(text)

        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError):
            raise  # not retryable -- retrying won't fix a bad key or a bad request

        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APIStatusError) as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise

    raise last_exc  # pragma: no cover -- loop always returns or raises above


def _triage_one(client, case: dict) -> dict:
    """Applies the hard fallback rules, then calls the model. Never
    raises -- any failure here becomes a logged fallback result instead."""
    if not case["agent_notes"]:
        return _fallback(f"{case['promise_id']}: no agent_notes logged")

    if client is None:
        return _fallback(f"{case['promise_id']}: Anthropic client unavailable (missing SDK or API key)")

    try:
        parsed = _call_model(client, case)
    except Exception as exc:  # noqa: BLE001 -- any API/parse failure degrades to a fallback, on purpose
        return _fallback(f"{case['promise_id']}: API call failed after retries ({exc.__class__.__name__}: {exc})")

    severity = parsed.get("severity")
    confidence = parsed.get("confidence")
    justification = parsed.get("justification")

    if severity not in VALID_SEVERITIES or not isinstance(justification, str) or not isinstance(confidence, (int, float)):
        return _fallback(f"{case['promise_id']}: model response was not valid JSON in the expected shape")

    if confidence < CONFIDENCE_THRESHOLD:
        return _fallback(
            f"{case['promise_id']}: model confidence {confidence:.2f} below the {CONFIDENCE_THRESHOLD} threshold "
            f"(model suggested severity={severity!r})"
        )

    return {
        "severity": severity,
        "justification": justification,
        "confidence": float(confidence),
        "ai_fallback": False,
    }


def _make_client():
    """Builds an Anthropic client if possible; returns None (never raises)
    if the SDK isn't installed or no credentials are configured, so the
    pipeline can still run end-to-end with every case falling back
    cleanly to needs_review."""
    try:
        import anthropic
    except ImportError:
        print("[ai_triage] anthropic SDK not installed -- all exceptions will fall back to needs_review")
        return None

    try:
        return anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / ant profile
    except Exception as exc:  # noqa: BLE001 -- client construction failure is also a clean fallback path
        print(f"[ai_triage] could not create Anthropic client ({exc}) -- all exceptions will fall back to needs_review")
        return None


def apply_ai_triage(results: list) -> dict:
    """Runs AI triage over every non-clean_match case in `results`, mutating
    each one in place with severity/justification/confidence/ai_fallback.
    clean_match cases are left completely untouched. Returns a summary dict
    of counts for the report.
    """
    client = _make_client()

    exceptions = [r for r in results if r["label"] != "clean_match"]
    summary = {"total_exceptions": len(exceptions), "explained": 0, "needs_review": 0,
               "critical": 0, "fallback_count": 0}

    for case in exceptions:
        triage = _triage_one(client, case)
        case.update(triage)
        summary[triage["severity"]] += 1
        if triage["ai_fallback"]:
            summary["fallback_count"] += 1

    return summary
