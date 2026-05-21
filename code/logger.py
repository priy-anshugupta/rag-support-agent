"""
Chat Transcript Logger — Writes structured log entries per AGENTS.md spec.

Handles both the AGENTS.md-required log file (in USERPROFILE) and
the per-ticket processing log.
"""
import os
from datetime import datetime
from pathlib import Path

from config import LOG_FILE


def ensure_log_dir():
    """Create the log directory if it doesn't exist."""
    log_dir = LOG_FILE.parent
    log_dir.mkdir(parents=True, exist_ok=True)


def log_session_start():
    """Log a session start entry per AGENTS.md §5.1."""
    ensure_log_dir()
    
    now = datetime.now().astimezone()
    # Calculate time remaining until May 2, 2026 11:00 IST
    from datetime import timezone, timedelta
    end_time = datetime(2026, 5, 2, 11, 0, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    remaining = end_time - now
    
    days = remaining.days
    hours, rem = divmod(remaining.seconds, 3600)
    minutes = rem // 60
    
    entry = f"""
## [{now.isoformat()}] SESSION START

Agent: Antigravity (Support Triage Agent)
Repo Root: {Path(__file__).resolve().parent.parent}
Branch: main
Worktree: main
Parent Agent: none
Language: py
Time Remaining: {days}d {hours}h {minutes}m
"""
    
    with open(LOG_FILE, "a", encoding="utf-8", newline='\n') as f:
        f.write(entry)


def log_ticket_processing(
    ticket_num: int,
    ticket: dict,
    result: dict,
    log_file: str | Path | None = None
):
    """
    Log a single ticket processing entry.
    
    Args:
        ticket_num: Ticket number (1-indexed)
        ticket: Input ticket dict with Issue, Subject, Company keys
        result: Processing result dict
        log_file: Override log file path (defaults to config LOG_FILE)
    """
    target_file = Path(log_file) if log_file else LOG_FILE
    ensure_log_dir()
    
    now = datetime.now().astimezone()
    issue_text = str(ticket.get("Issue", ""))[:500]
    subject = str(ticket.get("Subject", "N/A"))
    company = str(ticket.get("Company", "N/A"))
    
    # Build retrieved chunks summary
    chunks_summary = ""
    for chunk in result.get("retrieved_chunks", []):
        source = os.path.basename(chunk.get("source", "unknown"))
        score = chunk.get("score", 0)
        chunks_summary += f"  - {source} (score: {score})\n"
    
    if not chunks_summary:
        chunks_summary = "  (no relevant chunks found)\n"
    
    entry = f"""
{'=' * 60}
TICKET #{ticket_num} — {now.isoformat()}
{'=' * 60}

[INPUT]
Company:  {company}
Subject:  {subject}
Issue:    {issue_text}

[SAFETY CHECK]
{result.get('safety_reason', 'clean')}

[RETRIEVAL]
{chunks_summary}
[REASONING]
{result.get('justification', 'N/A')}

[OUTPUT]
Status:       {result.get('status', 'N/A')}
Product Area: {result.get('product_area', 'N/A')}
Request Type: {result.get('request_type', 'N/A')}
Response:
{result.get('response', 'N/A')}
"""
    
    with open(target_file, "a", encoding="utf-8", newline='\n') as f:
        f.write(entry)


def log_agent_turn(title: str, user_prompt: str, summary: str, actions: list[str]):
    """
    Log a per-turn entry per AGENTS.md §5.2.
    
    Args:
        title: Short title (max 80 chars)
        user_prompt: The user's prompt (secrets redacted)
        summary: 2-5 sentence summary of what was done
        actions: List of actions taken
    """
    ensure_log_dir()
    
    now = datetime.now().astimezone()
    repo_root = Path(__file__).resolve().parent.parent
    
    actions_text = "\n".join(f"* {a}" for a in actions) if actions else "* No file changes"
    
    entry = f"""
## [{now.isoformat()}] {title[:80]}

User Prompt (verbatim, secrets redacted):
{user_prompt[:1000]}

Agent Response Summary:
{summary}

Actions:
{actions_text}

Context:
tool=Antigravity
branch=main
repo_root={repo_root}
worktree=main
parent_agent=none
"""
    
    with open(LOG_FILE, "a", encoding="utf-8", newline='\n') as f:
        f.write(entry)


def log_run_summary(total_tickets: int, results_summary: dict):
    """Log a summary of the complete agent run."""
    ensure_log_dir()
    
    now = datetime.now().astimezone()
    
    entry = f"""
{'=' * 60}
AGENT RUN SUMMARY — {now.isoformat()}
{'=' * 60}

Total tickets processed: {total_tickets}
Replied: {results_summary.get('replied', 0)}
Escalated: {results_summary.get('escalated', 0)}

Request types:
  product_issue: {results_summary.get('product_issue', 0)}
  feature_request: {results_summary.get('feature_request', 0)}
  bug: {results_summary.get('bug', 0)}
  invalid: {results_summary.get('invalid', 0)}

Safety flags:
  Injection attempts: {results_summary.get('injections', 0)}
  Malicious requests: {results_summary.get('malicious', 0)}
  Out of scope: {results_summary.get('out_of_scope', 0)}
"""
    
    with open(LOG_FILE, "a", encoding="utf-8", newline='\n') as f:
        f.write(entry)
