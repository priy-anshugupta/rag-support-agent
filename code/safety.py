"""
Safety & Malicious Input Detection Module.

Catches prompt injection, malicious commands, and out-of-scope requests
BEFORE they reach the LLM. This is a critical security layer.
"""
import re
from typing import TypedDict


class SafetyResult(TypedDict):
    is_injection: bool
    is_malicious: bool
    is_out_of_scope: bool
    is_empty: bool
    reason: str


# ── Prompt Injection Patterns ──────────────────────────────────────────────────
# These detect attempts to manipulate the agent's behavior
INJECTION_PATTERNS = [
    # English injection patterns
    r"ignore\s+(previous|all|above|prior|every)\s+instructions?",
    r"show\s+(me\s+)?(your\s+)?(system\s*prompt|internal\s*rules|retrieved\s*doc|context|instructions|logic)",
    r"reveal\s+(your\s+)?(prompt|instructions|rules|logic|documents|context)",
    r"print\s+(your\s+)?(system|internal|hidden)",
    r"bypass\s+(safety|filters?|rules?|restrictions?|guardrails?)",
    r"you\s+are\s+now",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(a\s+|an\s+)?(?!support|customer|agent)",
    r"forget\s+(everything|your\s+instructions|all)",
    r"disregard\s+(your\s+|all\s+)?instructions",
    r"what\s+(documents?|context|corpus|chunks?)\s+(did\s+you|have\s+you|are\s+you)\s+(retrieve|use|get)",
    r"override\s+(your|all|the)\s+(rules?|instructions?|settings?)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"jailbreak",
    r"DAN\s*mode",
    
    # French injection patterns (Ticket 25 specific)
    r"logique\s+exacte",
    r"r[eè]gles?\s+internes?",
    r"affiche\s+(toutes?|les|tout)",
    r"documents?\s+r[eé]cup[eé]r[eé]s?",
    
    # Spanish injection patterns
    r"muestra\s+(todas?|los|las)\s+(reglas?|instrucciones?)",
    r"ignora\s+(las\s+)?instrucciones",
    
    # Generic multilingual
    r"interne\s+regeln",  # German
    r"mostra\s+(tutte|le)\s+regole",  # Italian
]

# -- Malicious / Harmful Request Patterns --
# NOTE: These are for genuinely dangerous requests (hack, virus, exfiltrate).
# "Delete files" and similar are NOT malicious -- they are out-of-scope and
# should be replied to as invalid (see sample Ticket 24).
MALICIOUS_PATTERNS = [
    r"rm\s+-rf",
    r"drop\s+(table|database)",
    r"execute\s+(code|command|script|shell)",
    r"hack\s+(into|the|a)",
    r"(get|give)\s+me\s+(the\s+)?(password|credentials?|api\s*key|secret|token)",
    r"(exfiltrate|steal|leak)\s+(data|information|credentials?)",
    r"format\s+(the\s+)?(hard\s*drive|disk|system)",
    r"(create|write|generate)\s+(a\s+)?(virus|malware|trojan|ransomware)",
]

# -- Out of Scope Patterns --
# NOTE: Removed 'fill forms' — Ticket 6 (HackerRank infosec) is a legitimate
# request that should go to LLM, not be hard-blocked.
# Added 'delete files' patterns here as out-of-scope (not malicious) per sample.
OUT_OF_SCOPE_PATTERNS = [
    r"who\s+(is|was)\s+the\s+(actor|president|ceo|minister|king|queen)",
    r"what\s+is\s+(the\s+)?(capital|population|height|weight|distance)",
    r"(iron\s*man|marvel|movie|film|actor|actress|celebrity)\s",
    r"(recipe|cook|bake|ingredients)\s+for",
    r"(weather|forecast)\s+(in|for|at)",
    r"help\s+me\s+with\s+my\s+homework",
    r"(give|write|provide)\s+(me\s+)?(the\s+)?code\s+to\s+delete",
    r"delete\s+(all\s+)?files?\s+from",
]

# ── Compiled Pattern Cache ─────────────────────────────────────────────────────
_compiled_injection = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_compiled_malicious = [re.compile(p, re.IGNORECASE) for p in MALICIOUS_PATTERNS]
_compiled_oos = [re.compile(p, re.IGNORECASE) for p in OUT_OF_SCOPE_PATTERNS]


def check_safety(text: str) -> SafetyResult:
    """
    Analyze input text for safety concerns.
    Returns a SafetyResult dict with flags for each type of concern.
    
    Priority: injection > malicious > out_of_scope > empty > clean
    """
    if not text or not text.strip() or text.strip().lower() in ("nan", "none", ""):
        return SafetyResult(
            is_injection=False,
            is_malicious=False,
            is_out_of_scope=False,
            is_empty=True,
            reason="Empty or missing ticket content"
        )
    
    text_lower = text.lower().strip()
    
    # Check prompt injection FIRST (highest severity)
    for pattern in _compiled_injection:
        match = pattern.search(text_lower)
        if match:
            return SafetyResult(
                is_injection=True,
                is_malicious=False,
                is_out_of_scope=False,
                is_empty=False,
                reason=f"Prompt injection attempt detected: matched '{match.group()}'"
            )
    
    # Check malicious/harmful requests
    for pattern in _compiled_malicious:
        match = pattern.search(text_lower)
        if match:
            return SafetyResult(
                is_injection=False,
                is_malicious=True,
                is_out_of_scope=False,
                is_empty=False,
                reason=f"Malicious/harmful request detected: matched '{match.group()}'"
            )
    
    # Check out-of-scope requests
    for pattern in _compiled_oos:
        match = pattern.search(text_lower)
        if match:
            return SafetyResult(
                is_injection=False,
                is_malicious=False,
                is_out_of_scope=True,
                is_empty=False,
                reason=f"Out of scope request: matched '{match.group()}'"
            )
    
    return SafetyResult(
        is_injection=False,
        is_malicious=False,
        is_out_of_scope=False,
        is_empty=False,
        reason="clean"
    )


def is_greeting_or_thanks(text: str) -> bool:
    """Check if the ticket is just a greeting or thank you message."""
    text_lower = text.strip().lower()
    
    # Check common short greetings/thanks
    greeting_patterns = [
        r"^(thank\s*you|thanks|thx|ty)(\s+for\s+helping(\s+me)?)?\s*[.!]?\s*$",
        r"^(hi|hello|hey|good\s*(morning|afternoon|evening))\s*[.!]?\s*$",
        r"^(ok|okay|sure|great|nice|good)\s*[.!]?\s*$",
        r"^(bye|goodbye|see\s*you|cheers)\s*[.!]?\s*$",
        r"^happy\s+to\s+help\s*[.!]?\s*$",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, text_lower):
            return True
    
    return False


def is_vague_ticket(text: str) -> bool:
    """Check if the ticket is too vague to process meaningfully."""
    text_lower = text.strip().lower()
    vague_patterns = [
        r"^it'?s?\s+not\s+working",
        r"^(help|help\s*me|help\s*needed)\.?!?\s*$",
        r"^(not\s+working|doesn'?t\s+work|broken)\s*\.?!?\s*$",
        r"^(something\s+is\s+wrong|there'?s?\s+a\s+problem)\s*\.?!?\s*$",
        r"^fix\s+(it|this)\s*\.?!?\s*$",
    ]
    for pattern in vague_patterns:
        if re.match(pattern, text_lower):
            return True
    return False
