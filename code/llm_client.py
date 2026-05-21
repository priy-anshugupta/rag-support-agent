"""
OpenAI API Client — Handles LLM calls with retries, rate limiting, and JSON parsing.

Uses the modern OpenAI SDK with gpt-4o-mini for fast, structured output.
Includes robust JSON extraction.
"""
import json
import re
import time
import os
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_RETRIES, OPENAI_DELAY_SECONDS


# Configure the client
_api_keys = []
_current_key_idx = 0
_client = None

def _get_client(rotate: bool = False):
    """Lazy-initialize the OpenAI client, with optional key rotation."""
    global _client, _api_keys, _current_key_idx
    
    if not _api_keys:
        raw_keys = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not raw_keys:
            print("[OpenAI] WARNING: OPENAI_API_KEY not set. Set it via environment variable.")
            return None
        _api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        
    if rotate and len(_api_keys) > 1:
        _current_key_idx = (_current_key_idx + 1) % len(_api_keys)
        print(f"[OpenAI] 🔄 Rate limit hit. Rotating to API key #{_current_key_idx + 1}...")
        _client = None  # Force re-init
        
    if _client is None and _api_keys:
        _client = OpenAI(api_key=_api_keys[_current_key_idx])
        
    return _client


def _clean_json_response(raw: str) -> str:
    """Strip markdown code fences and extract JSON from response."""
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    return raw.strip()


def _extract_json(raw: str) -> dict | None:
    """Try multiple strategies to extract valid JSON from a response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
    first_brace = raw.find('{')
    last_brace = raw.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass
            
    return None


def call_llm(prompt: str, retries: int = OPENAI_MAX_RETRIES) -> dict:
    """
    Call OpenAI API with the given prompt and return parsed JSON response.
    """
    client = _get_client()
    if client is None:
        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": "Unable to process this ticket automatically. Escalating to human support.",
            "justification": "OpenAI API client not configured (missing API key).",
            "request_type": "product_issue"
        }
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Always output pure JSON without markdown blocks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic output for reproducibility
                max_completion_tokens=2000,  # Allow richer, more detailed responses
                seed=42,  # Fixed seed for deterministic, reproducible results
                response_format={ "type": "json_object" }
            )
            
            raw = response.choices[0].message.content.strip()
            cleaned = _clean_json_response(raw)
            result = _extract_json(cleaned)
            
            if result:
                required_keys = ["status", "product_area", "response", "justification", "request_type"]
                for key in required_keys:
                    if key not in result:
                        result[key] = "unknown" if key != "response" else "Unable to process this request."
                return result
                
            print(f"[OpenAI] JSON parse failed on attempt {attempt + 1}/{retries}")
            time.sleep(2 ** attempt)
            
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str or "quota" in error_str:
                if len(_api_keys) > 1:
                    client = _get_client(rotate=True)
                else:
                    wait_time = OPENAI_DELAY_SECONDS * (2 ** attempt)
                    print(f"[OpenAI] Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            elif "block" in error_str or "safety" in error_str:
                return {
                    "status": "escalated",
                    "product_area": "security",
                    "response": "This ticket has been flagged for review and escalated to our support team.",
                    "justification": "Content triggered safety filters and requires human review.",
                    "request_type": "invalid"
                }
            else:
                print(f"[OpenAI] API error on attempt {attempt + 1}: {e}")
                time.sleep(OPENAI_DELAY_SECONDS)
                
    return {
        "status": "escalated",
        "product_area": "unknown",
        "response": "Unable to process this ticket automatically. Escalating to human support for assistance.",
        "justification": "Automated processing failed after multiple attempts.",
        "request_type": "product_issue"
    }
