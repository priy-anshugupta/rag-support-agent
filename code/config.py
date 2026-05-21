"""
Configuration for the Multi-Domain Support Triage Agent.
All constants, API keys, escalation rules, and mappings live here.
"""
import os
from pathlib import Path

# -- API Configuration --
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5.4-mini"  # Upgraded: excellent reasoning at low cost

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data"
INPUT_CSV = PROJECT_ROOT / "support_tickets" / "support_tickets.csv"
SAMPLE_CSV = PROJECT_ROOT / "support_tickets" / "sample_support_tickets.csv"
OUTPUT_CSV = PROJECT_ROOT / "support_tickets" / "output.csv"

# Log file path per AGENTS.md spec
LOG_FILE = Path(os.environ.get("USERPROFILE", os.environ.get("HOME", "."))) / "hackerrank_orchestrate" / "log.txt"

# ── Retrieval Settings ─────────────────────────────────────────────────────────
TOP_K_CHUNKS = 5          # Number of corpus chunks to retrieve per ticket
MIN_RELEVANCE_SCORE = 0.02  # Minimum TF-IDF cosine similarity to consider relevant
CHUNK_MAX_CHARS = 3000     # Max characters per chunk sent to LLM

# ── Companies ──────────────────────────────────────────────────────────────────
COMPANIES = ["HackerRank", "Claude", "Visa"]

COMPANY_CORPUS_MAP = {
    "hackerrank": CORPUS_DIR / "hackerrank",
    "claude": CORPUS_DIR / "claude",
    "visa": CORPUS_DIR / "visa",
}

# ── Company Inference Keywords ─────────────────────────────────────────────────
COMPANY_KEYWORDS = {
    "HackerRank": [
        "hackerrank", "hacker rank", "test", "assessment", "recruiter", "candidate",
        "coding test", "interview", "proctoring", "screen", "hiring", "resume builder",
        "certificate", "mock interview", "lti", "submissions", "challenges",
        "apply tab", "interviewer", "campus", "skillup", "engage"
    ],
    "Claude": [
        "claude", "anthropic", "conversation", "ai model", "bedrock", "workspace",
        "ai assistant", "chat", "prompt", "api", "token", "model", "crawler",
        "crawling", "lti key", "claude code", "claude desktop"
    ],
    "Visa": [
        "visa", "card", "merchant", "payment", "cheque", "transaction",
        "cardholder", "debit", "credit", "atm", "cash advance", "dispute",
        "charge", "traveller", "travel", "identity theft", "stolen card",
        "visa card", "minimum spend"
    ],
}

# -- Hard Escalation Keywords --
# If ANY of these appear in (issue + subject), escalate immediately without LLM.
# IMPORTANT: These are OVERRIDDEN by high retrieval confidence (>0.15) in agent.py.
# This means if the corpus has a clear answer (e.g., stolen Visa cheques), we
# still reply with the answer instead of blindly escalating.
ESCALATION_KEYWORDS = [
    # Fraud & Security (but NOT "stolen" -- corpus has answers for Visa stolen cheques/cards)
    "fraud", "unauthorized", "identity theft", "hacked",
    "security vulnerability", "breach", "compromised", "scam",

    # Billing & Payments (require human intervention)
    "refund", "invoice", "order id", "order_id", "cs_live_",

    # Account access issues requiring admin
    "restore my access", "not the admin", "not the owner",
    "not the workspace owner",

    # Legal / regulatory
    "legal action", "lawsuit", "report to authorities",

    # Platform-wide outages
    "none of the submissions", "all requests are failing",
    "stopped working completely", "site is down",
    "completely down",

    # Impossible requests (agent cannot fulfill)
    "change my score", "tell the company", "move me to the next round",
    "ban the seller", "increase my score",

    # Bug bounty / vulnerability reports
    "bug bounty", "major vulnerability",

    # Service completely down
    "resume builder is down",

    # Candidate assessment rescheduling (candidates CANNOT self-service this)
    "rescheduling of my", "reschedule my assessment", "reschedule my test",
]

# ── Valid Output Values ────────────────────────────────────────────────────────
VALID_STATUSES = ["replied", "escalated"]
VALID_REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

# -- Rate Limiting --
OPENAI_DELAY_SECONDS = 1  # Delay between API calls to respect rate limits
OPENAI_MAX_RETRIES = 3
