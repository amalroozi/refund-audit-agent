#!/usr/bin/env python3
"""
ai_triage.py
------------
AI-assisted severity triage for refund reconciliation exceptions, backed
by the Groq API (llama-3.1-8b-instant).

This module runs on every case the deterministic classifier in reconcile.py
did NOT label clean_match -- partial, over_refund, delayed, no_refund,
double_refund, orphan_refund, and unknown_pattern alike. It never sees, and
cannot influence, clean_match cases.

For each exception it asks the model to read the numbers + the support
agent's free-text note and assign a *severity* -- a judgment call about how
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

The provider itself is isolated behind a single function, _call_model() --
everything else (fallback rules, JSON parsing, severity validation, the
circuit breaker) is provider-agnostic and doesn't need to change if the
backend is swapped again.

Credentials: GROQ_API_KEY is read from a .env file at the project root
(via python-dotenv) if one exists, falling back to whatever is already set
in the OS environment otherwise -- see load_dotenv() call below.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file at the project root into the process
# environment BEFORE anything below reads GROQ_API_KEY (os.environ.get()
# calls happen later, inside _make_client()). If no .env file exists here,
# this is a no-op and os.environ (the OS environment) is used as-is --
# load_dotenv() never raises and never overwrites a variable that is
# already set in the OS environment.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

MODEL = "openai/gpt-oss-20b"

# Unlike some larger/"thinking" models, this model's answer IS its output --
# there's no separate internal-reasoning budget competing with the JSON for
# max_tokens, so this only needs to cover the actual ~40-60 token answer.
MAX_OUTPUT_TOKENS = 800

MAX_RETRIES = 2  # retries AFTER the first attempt -> up to 3 calls total
DEFAULT_RETRY_BACKOFF_SECONDS = 1.5  # only used if Groq gives us no usable header
MAX_RETRY_BACKOFF_SECONDS = 60.0     # ceiling even if the server asks for longer
CONFIDENCE_THRESHOLD = 0.5

VALID_SEVERITIES = {"explained", "needs_review", "critical"}

SYSTEM_PROMPT = """You are a triage assistant for a refund reconciliation system.

You are given ONE refund case that a rule-based system has ALREADY classified
(the label is final and you cannot change it). Your only job is to judge how
severe/worrying this exception is for a human operator, using the support
agent's note as context.

The label may be partial, over_refund, delayed, no_refund, double_refund,
orphan_refund, or unknown_pattern. orphan_refund cases have NO CRM promise
at all (so there is nothing to compare the note against, if one even
exists) -- do not treat the missing promise fields as an error.

Severity definitions:
- explained: the agent note corroborates the discrepancy (e.g. the note
  mentions a partial or discretionary refund, and the gateway shows a
  shortfall consistent with that; or the note explains why a refund is
  delayed, over-refunded, or was not issued).
- critical: the note contradicts what actually happened (e.g. it promises a
  full refund but the money is short, missing, or duplicated), OR a large
  amount is involved (no_refund, double_refund, orphan_refund, over_refund)
  with nothing in the note to explain it.
- needs_review: anything ambiguous, or no note was logged at all.

Respond with STRICT JSON only. No markdown code fences, no prose before or
after it. Respond with exactly this shape:
{"severity": "explained|needs_review|critical", "justification": "<one sentence, must reference the agent note content when a note exists>", "confidence": 0.0}
"""


def _build_user_message(case: dict) -> str:
    """Renders the case's numbers/dates/note into a compact prompt. `case`
    is one reconcile.py result dict (see reconcile.classify_case). Several
    fields are None for orphan_refund cases (there is no CRM promise behind
    them at all), so every field is rendered defensively rather than
    assumed present."""
    promised = f"{case['promised_amount']:.2f}" if case["promised_amount"] is not None else "N/A (no CRM promise exists for this order)"
    refunded = (
        f"{case['refunded_amount']:.2f}"
        if case["refunded_amount"] is not None
        else "NONE -- no gateway transaction at all"
    )
    promise_date = case["promise_date"].isoformat() if case["promise_date"] else "N/A"
    processed = case["refund_processed_date"].isoformat() if case["refund_processed_date"] else "N/A"
    window = (
        f"{case['window_start_days']}-{case['window_end_days']} days"
        if case["window_start_days"] is not None else "N/A"
    )
    notes = case["agent_notes"].strip() if case["agent_notes"] else "(no note logged)"

    return (
        f"Deterministic label: {case['label']}\n"
        f"Promised amount: {promised}\n"
        f"Refunded amount: {refunded}\n"
        f"Promise date: {promise_date}\n"
        f"Refund processed date: {processed}\n"
        f"Promised window: {window}\n"
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


def _response_headers(exc):
    """Groq's SDK exceptions carry the raw httpx response on .response for
    any APIStatusError subclass (RateLimitError, InternalServerError, ...);
    APIConnectionError has no response at all. Centralized here so every
    caller can ask for headers without re-deriving this each time."""
    response = getattr(exc, "response", None)
    return getattr(response, "headers", None) or {}


def _retry_after_seconds(exc, attempt: int) -> float:
    """Reads Groq's own `retry-after` response header (seconds, only set on
    a 429) -- that's a far better estimate of a genuinely transient
    per-minute rate limit than a guessed backoff. Falls back to a short
    fixed schedule if the header is missing or malformed. Always capped so
    a single retry can't stall the run."""
    raw = _response_headers(exc).get("retry-after")
    if raw is not None:
        try:
            return min(float(raw), MAX_RETRY_BACKOFF_SECONDS)
        except ValueError:
            pass
    return min(DEFAULT_RETRY_BACKOFF_SECONDS * (attempt + 1), MAX_RETRY_BACKOFF_SECONDS)


def _is_daily_quota_exhausted(exc) -> bool:
    """Per Groq's own docs, x-ratelimit-*-requests headers always describe
    the Requests-Per-Day (RPD) quota, while x-ratelimit-*-tokens headers
    describe the Tokens-Per-Minute (TPM) quota. So a 429 with zero requests
    remaining is a daily exhaustion no retry within this run can outlast;
    a 429 that still has requests remaining is just a transient per-minute
    token-rate limit, which IS worth retrying."""
    remaining = _response_headers(exc).get("x-ratelimit-remaining-requests")
    if remaining is None:
        return False
    try:
        return float(remaining) <= 0
    except ValueError:
        return False


def _call_model(client, case: dict) -> dict:
    """Calls Groq for one case, with up to MAX_RETRIES retries on transient
    failures. Raises the last exception if every attempt fails;
    non-retryable errors (bad request, auth, not found) are raised
    immediately. This is the ONLY function that knows which LLM provider is
    behind ai_triage -- swapping providers again means rewriting this
    function and _make_client(), nothing else."""
    import groq  # imported lazily so the module still loads without the SDK installed

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(case)},
                ],
            )
            return _extract_json(response.choices[0].message.content or "")

        except groq.RateLimitError as exc:
            # 429 -- only worth retrying when it's NOT a daily-quota
            # exhaustion (which no retry within this run can outlast).
            if attempt < MAX_RETRIES and not _is_daily_quota_exhausted(exc):
                last_exc = exc
                time.sleep(_retry_after_seconds(exc, attempt))
                continue
            raise

        except groq.InternalServerError as exc:
            # 5xx -- transient, worth retrying
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(_retry_after_seconds(exc, attempt))
                continue
            raise

        except groq.APIConnectionError as exc:
            # Network-level failure, no response/headers to read -- fall
            # back to the fixed schedule.
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(DEFAULT_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise

    raise last_exc  # pragma: no cover -- loop always returns or raises above


def _triage_one(client, case: dict) -> dict:
    """Applies the hard fallback rules, then calls the model. Never
    raises -- any failure here becomes a logged fallback result instead."""
    if not case["agent_notes"]:
        return _fallback(f"{case['promise_id']}: no agent_notes logged")

    if client is None:
        return _fallback(f"{case['promise_id']}: Groq client unavailable (missing SDK or GROQ_API_KEY)")

    try:
        parsed = _call_model(client, case)
    except Exception as exc:  # noqa: BLE001 -- any API/parse failure degrades to a fallback, on purpose
        import groq
        result = _fallback(f"{case['promise_id']}: API call failed after retries ({exc.__class__.__name__}: {exc})")
        # Flagged separately (and popped before the caller stores this dict)
        # so apply_ai_triage() can stop burning the rest of the batch's
        # retry time on a quota that provably won't recover today.
        result["daily_quota_exhausted"] = isinstance(exc, groq.RateLimitError) and _is_daily_quota_exhausted(exc)
        return result

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
    """Builds a Groq client if possible; returns None (never raises) if the
    SDK isn't installed or GROQ_API_KEY isn't configured, so the pipeline
    can still run end-to-end with every case falling back cleanly to
    needs_review."""
    try:
        import groq
    except ImportError:
        print("[ai_triage] groq SDK not installed -- all exceptions will fall back to needs_review")
        return None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[ai_triage] GROQ_API_KEY not set (checked .env then OS environment) "
              "-- all exceptions will fall back to needs_review")
        return None

    try:
        return groq.Groq(api_key=api_key)
    except Exception as exc:  # noqa: BLE001 -- client construction failure is also a clean fallback path
        print(f"[ai_triage] could not create Groq client ({exc}) -- all exceptions will fall back to needs_review")
        return None


def apply_ai_triage(results: list) -> dict:
    """Runs AI triage over every non-clean_match case in `results`, mutating
    each one in place with severity/justification/confidence/ai_fallback.
    clean_match cases are left completely untouched. Returns a summary dict
    of counts for the report.

    Trips a circuit breaker the first time a case reveals the day's free
    quota is exhausted (Groq's x-ratelimit-remaining-requests header hit
    zero): every remaining case in this batch is fallback'd immediately,
    with no further API calls, since none of them can succeed until the
    quota resets regardless of how long we wait within this run.
    """
    client = _make_client()

    exceptions = [r for r in results if r["label"] != "clean_match"]
    summary = {"total_exceptions": len(exceptions), "explained": 0, "needs_review": 0,
               "critical": 0, "fallback_count": 0}

    quota_exhausted_for_today = False
    for case in exceptions:
        if quota_exhausted_for_today:
            triage = _fallback(
                f"{case['promise_id']}: skipped -- Groq's free-tier daily quota was "
                f"already exhausted earlier in this run"
            )
        else:
            triage = _triage_one(client, case)
            if triage.pop("daily_quota_exhausted", False):
                quota_exhausted_for_today = True
                print("[ai_triage] Groq free-tier daily quota exhausted -- skipping "
                      "remaining exceptions without further API calls")

        case.update(triage)
        summary[triage["severity"]] += 1
        if triage["ai_fallback"]:
            summary["fallback_count"] += 1

    return summary
