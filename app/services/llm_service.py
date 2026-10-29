"""
app/services/llm_service.py
────────────────────────────
Stage C: AI-based semantic metadata extraction via Ollama.

Design
──────
• Talks to the Ollama HTTP API through the `ollama` Python client.
• Accepts pre-extracted document text and (optionally) few-shot examples.
• Returns a validated RawLLMOutput Pydantic object.
• Does NOT use any rule-based pattern matching or regex to locate field
  values; that work is delegated entirely to the LLM.
• Handles malformed JSON responses gracefully.
• Raises ModelUnavailableError when Ollama is unreachable or the
  requested model is not available.

This service is deliberately decoupled from document reading / OCR so
that an alternative LLM provider (e.g. a hosted API) can be plugged in
later by swapping only this module.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.config import get_settings
from app.schemas import RawLLMOutput

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Custom errors
# ─────────────────────────────────────────────────────────────


class ModelUnavailableError(Exception):
    """Raised when the configured Ollama model cannot be reached."""


class ExtractionParseError(Exception):
    """Raised when the LLM returns text that cannot be parsed as JSON."""


# ─────────────────────────────────────────────────────────────
# Extraction prompt template
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI legal document analyst specialising in extracting structured metadata from rental and lease agreements.

Inspect the COMPLETE DOCUMENT text thoroughly and extract exactly these six metadata fields as a JSON object:

  agreement_value        – The monthly rent or total agreement value stated in the document (e.g. "12000" or "6500").
                           Return the numeric rent value as a clean string without currency symbols or commas.
                           CRITICAL RENT RULES:
                           * Inspect both the numeric amount and the amount written in words.
                           * If the numeric amount and the amount written in words CONFLICT, prioritize the written amount in words per project policy.
                           * Do NOT confuse monthly rent with a security deposit or advance (e.g. 80000 or 165000).

  agreement_start_date   – The date on which the rental period or tenancy term begins (commencement date).
                           Use format DD.MM.YYYY (e.g. "01.04.2010" or "20.05.2007"). Do not confuse with execution/signing date.

  agreement_end_date     – The date on which the rental period ends (expiry / termination date).
                           Use format DD.MM.YYYY (e.g. "31.03.2011" or "31.03.2009").
                           * If an explicit end date is stated in the text, extract it verbatim.
                           * For calendar-month counting: leases starting on the 1st of a month end on the last calendar day of the concluding month (e.g. 01.04.2010 -> 31.03.2011). Leases starting mid-month on day D (e.g. 15.12.2012) end on day D - 1 (e.g. 14.11.2013).

  renewal_notice_days    – The advance notice period (in DAYS) required before renewal, termination, or vacating premises.
                           Convert month periods to days assuming 30 days per month:
                           * "one month notice" or "1 month" -> 30
                           * "two months' notice" or "2 months notice" -> 60
                           * "three months notice" or "3 months" -> 90
                           * If specified in days (e.g. 15, 30, 60), return that integer.
                           * Do not confuse notice period with lease duration or rent due date.

  party_one              – The LANDLORD / LESSOR / property owner / First Party.
                           Extract the clean contracting entity or person name WITHOUT salutations or honorifics (e.g. "Hanumaiah", "S.Sakunthala", "KAPIL MEHROTRA").
                           Do NOT extract locality names or street names from the property address.
                           Exclude father's, spouse's, or parent's name introduced by S/O, D/O, W/O, Slo, son of, etc.

  party_two              – The TENANT / LESSEE / occupier / Second Party.
                           Extract the clean contracting entity or person name WITHOUT salutations or honorifics (e.g. "Vishal Bhardwaj", "B.Kishore").
                           CRITICAL: Exclude father's name introduced by S/O, D/O, W/O, Slo, son of, etc. (e.g. for "Sri Vishal Bhardwaj S/O Charnel Singh", extract "Vishal Bhardwaj").
                           For organizations (e.g. "SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd"), extract company name without "SRI " and exclude representative clauses like "represented by Mr. XYZ".

Rules:
1. COMPLETE DOCUMENT INSPECTION: Inspect the full text before deciding a field is missing.
2. Return ONLY a valid JSON object with the six required fields. No markdown formatting, code fences, or explanatory prose.
3. If a field cannot be determined, return null.

Required JSON format:
{
  "agreement_value": "<string or null>",
  "agreement_start_date": "<string or null>",
  "agreement_end_date": "<string or null>",
  "renewal_notice_days": <integer or null>,
  "party_one": "<string or null>",
  "party_two": "<string or null>"
}"""



_CLIENT_CACHE: dict[str, Any] = {}


def _get_client(base_url: str):
    """Reuse loaded Ollama client to avoid reconnect overhead."""
    if base_url not in _CLIENT_CACHE:
        import ollama as ollama_client
        _CLIENT_CACHE[base_url] = ollama_client.Client(host=base_url)
    return _CLIENT_CACHE[base_url]


def _build_user_message(
    document_text: str,
    few_shot_examples: Optional[list[dict]] = None,
) -> str:
    """
    Build the user-turn message, optionally prepending concise few-shot example.
    """
    parts: list[str] = []

    if few_shot_examples:
        parts.append("Here is an example of correct extraction:\n")
        for i, ex in enumerate(few_shot_examples[:1], 1):
            parts.append(f"--- Example ---")
            parts.append(f"DOCUMENT TEXT:\n{ex['document_text'][:1200]}")
            parts.append(
                f"EXPECTED OUTPUT:\n{json.dumps(ex['expected_output'], indent=2)}"
            )
        parts.append("\n--- Now extract from the following document ---\n")

    parts.append(f"DOCUMENT TEXT:\n{document_text}")
    return "\n".join(parts)


def _extract_json_from_response(text: str) -> dict:
    """
    Parse JSON from the LLM response text.

    Tries in order:
      1. Direct JSON.parse of the full text.
      2. Extraction of the first {...} block (model sometimes adds prose).

    Raises
    ------
    ExtractionParseError
        If no valid JSON object can be found.
    """
    text = text.strip()

    # Attempt 1: full text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Attempt 3: strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    raise ExtractionParseError(
        f"Could not parse a valid JSON object from model response. "
        f"Response preview: {text[:300]!r}"
    )


def call_llm(
    document_text: str,
    few_shot_examples: Optional[list[dict]] = None,
) -> RawLLMOutput:
    """
    Send *document_text* to the configured Ollama model and return
    a validated :class:`RawLLMOutput`.

    Parameters
    ----------
    document_text:
        Full text of the document to analyse.
    few_shot_examples:
        Optional list of {document_text, expected_output} dicts.

    Returns
    -------
    RawLLMOutput
        Validated Pydantic model with extracted field values.

    Raises
    ------
    ModelUnavailableError
        If Ollama is unreachable or the model is not installed.
    ExtractionParseError
        If the model returns unparseble output.
    """
    settings = get_settings()

    try:
        import ollama as ollama_client  # type: ignore[import]
    except ImportError as exc:
        raise ModelUnavailableError(
            "The 'ollama' package is required. Install it: pip install ollama"
        ) from exc

    user_message = _build_user_message(document_text, few_shot_examples)

    logger.info(
        "Calling Ollama model '%s' (text length: %d chars)",
        settings.ollama_model,
        len(document_text),
    )

    # Generation options: small nonzero temperature and repeat_penalty prevent token repeat loops
    chat_options: dict = {
        "temperature": getattr(settings, "ollama_temperature", 0.1),
        "repeat_penalty": getattr(settings, "ollama_repeat_penalty", 1.1),
        "top_p": 0.9,
        "num_predict": min(settings.ollama_num_predict, 250),
        "num_gpu": settings.ollama_num_gpu,
        "num_ctx": settings.ollama_num_ctx,
    }

    def _stream_chat(client, options: dict) -> str:
        """
        Use streaming mode so Ollama keeps the TCP connection alive token-by-token
        rather than holding one long silent connection that Windows kills.
        Enforces format='json' for structured JSON output without natural-language summary.
        """
        chunks: list[str] = []
        stream = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            format="json",
            options=options,
            keep_alive=settings.ollama_keep_alive,
            stream=True,
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                chunks.append(token)
        text = "".join(chunks).strip()
        if not text:
            raise ExtractionParseError(
                "Model returned an empty response. "
                "The model may have run out of tokens (num_predict too small) "
                "or failed internally. Check Ollama logs."
            )
        return text

    try:
        client = _get_client(settings.ollama_base_url)
        try:
            raw_text = _stream_chat(client, chat_options)
        except ExtractionParseError:
            raise  # propagate empty-response error directly
        except Exception as chat_exc:
            err_msg = str(chat_exc).lower()
            # If GPU/CUDA error occurred, retry immediately in CPU mode
            if (
                any(k in err_msg for k in ("cuda", "device kernel", "0xc0000409", "image is invalid"))
                and chat_options.get("num_gpu", 0) != 0
            ):
                logger.warning(
                    "CUDA error detected (%s). Retrying with num_gpu=0 (CPU mode)…", chat_exc
                )
                chat_options["num_gpu"] = 0
                raw_text = _stream_chat(client, chat_options)
            # If token repeat limit occurred, perform at most one controlled retry with adjusted generation settings
            elif any(k in err_msg for k in ("repeat limit", "token repeat", "repeat_penalty")):
                logger.warning(
                    "Token repeat limit reached (%s). Performing controlled retry with adjusted settings (temperature=0.2, repeat_penalty=1.15)…",
                    chat_exc,
                )
                retry_options = dict(chat_options)
                retry_options["temperature"] = 0.2
                retry_options["repeat_penalty"] = 1.15
                try:
                    raw_text = _stream_chat(client, retry_options)
                except Exception as retry_exc:
                    logger.error("Controlled retry also failed: %s", retry_exc)
                    raise ExtractionParseError(
                        f"Ollama prediction aborted due to token repeat limit: {retry_exc}"
                    ) from retry_exc
            else:
                raise

    except (ExtractionParseError, ModelUnavailableError):
        raise  # already typed — let caller handle

    except ollama_client.ResponseError as exc:
        err_lower = str(exc).lower()
        if any(k in err_lower for k in ("repeat limit", "token repeat")):
            raise ExtractionParseError(
                f"Ollama prediction aborted due to token repeat limit: {exc}"
            ) from exc
        if "not found" in err_lower or "404" in str(exc):
            raise ModelUnavailableError(
                f"Model '{settings.ollama_model}' is not available in Ollama. "
                f"Download it with: ollama pull {settings.ollama_model}"
            ) from exc
        raise ModelUnavailableError(
            f"Ollama server returned an error: {exc}. "
            f"Ensure Ollama is running at {settings.ollama_base_url}."
        ) from exc

    except Exception as exc:
        err_str = str(exc).lower()
        if any(k in err_str for k in ("connection refused", "wsarecv", "aborted", "reset", "connect", "timeout", "eof")):
            raise ModelUnavailableError(
                f"Connection to Ollama was lost mid-generation at {settings.ollama_base_url}. "
                "This usually means Ollama ran out of memory or was killed by Windows. "
                "Try restarting Ollama with: ollama serve"
            ) from exc
        raise ModelUnavailableError(
            f"Unexpected error calling Ollama: {exc}"
        ) from exc

    # Parse JSON from model output
    raw_dict = _extract_json_from_response(raw_text)

    logger.debug("Raw LLM output: %s", json.dumps(raw_dict, indent=2)[:500])

    # Validate with Pydantic (permissive – extra fields ignored)
    try:
        return RawLLMOutput.model_validate(raw_dict)
    except Exception as exc:
        raise ExtractionParseError(
            f"LLM output failed Pydantic validation: {exc}. "
            f"Raw dict keys: {list(raw_dict.keys())}"
        ) from exc


def check_model_available() -> tuple[bool, Optional[str]]:
    """
    Probe the Ollama server to verify connectivity and model availability.

    Returns
    -------
    (available: bool, error_message: str | None)
    """
    settings = get_settings()
    try:
        import ollama as ollama_client  # type: ignore[import]
        client = ollama_client.Client(host=settings.ollama_base_url)
        models_resp = client.list()
        # models_resp.models is a list of Model objects
        available_names = [m.model for m in models_resp.models]
        # Check if configured model (with or without tag) is present
        model_name = settings.ollama_model
        found = any(
            name == model_name or name.startswith(model_name.split(":")[0])
            for name in available_names
        )
        if not found:
            return False, (
                f"Model '{model_name}' not found. "
                f"Available: {available_names}. "
                f"Run: ollama pull {model_name}"
            )
        return True, None
    except Exception as exc:
        return False, (
            f"Cannot reach Ollama at {settings.ollama_base_url}: {exc}. "
            "Start Ollama with: ollama serve"
        )
