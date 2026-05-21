"""
Main Entry Point -- Multi-Domain Support Triage Agent

Reads support tickets from CSV, processes each through the triage pipeline,
and writes structured results to output.csv and log.txt.

Usage:
    python main.py                           # Process support_tickets.csv
    python main.py --input path/to/input.csv # Process custom input
    python main.py --validate                # Validate against sample tickets
    python main.py --stats                   # Show corpus statistics
"""
import sys
import os
import io
import time
import argparse
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add code directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import (
    INPUT_CSV, OUTPUT_CSV, SAMPLE_CSV, CORPUS_DIR,
    OPENAI_API_KEY, OPENAI_DELAY_SECONDS, LOG_FILE
)
from retriever import CorpusRetriever
from agent import process_ticket
from logger import log_session_start, log_ticket_processing, log_run_summary


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 60)
    print("  Multi-Domain Support Triage Agent")
    print("  HackerRank Orchestrate -- May 2026")
    print("=" * 60)
    print(f"  Time: {datetime.now().astimezone().isoformat()}")
    print(f"  Corpus: {CORPUS_DIR}")
    print(f"  Log: {LOG_FILE}")
    print("=" * 60 + "\n")


def validate_environment():
    """Check that all required resources are available."""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY environment variable not set!")
        errors.append("  Set it with: $env:OPENAI_API_KEY='your-key-here'  (PowerShell)")
        errors.append("  Or: set OPENAI_API_KEY=your-key-here  (CMD)")
    
    if not CORPUS_DIR.exists():
        errors.append(f"Corpus directory not found: {CORPUS_DIR}")
    
    if errors:
        print("\n[!] CONFIGURATION ERRORS:")
        for e in errors:
            print(f"  {e}")
        print()
        if not OPENAI_API_KEY:
            return False
    
    return True


def process_tickets(input_csv: Path, output_csv: Path):
    """Process all tickets from input CSV and write results to output CSV."""
    print(f"[*] Loading tickets from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"[*] Found {len(df)} tickets to process\n")
    
    # Initialize retriever
    print("[*] Building corpus index...")
    retriever = CorpusRetriever(CORPUS_DIR)
    stats = retriever.get_corpus_stats()
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   Sources: {stats['unique_sources']}")
    print(f"   Companies: {stats['company_distribution']}\n")
    
    # Initialize log
    log_session_start()
    
    results = []
    run_summary = {
        "replied": 0, "escalated": 0,
        "product_issue": 0, "feature_request": 0, "bug": 0, "invalid": 0,
        "injections": 0, "malicious": 0, "out_of_scope": 0,
    }
    
    for i, row in df.iterrows():
        ticket = {
            "Issue": row.get("Issue", ""),
            "Subject": row.get("Subject", ""),
            "Company": row.get("Company", "None"),
        }
        
        issue_preview = str(ticket["Issue"])[:60].replace("\n", " ")
        print(f"\n{'---' * 20}")
        print(f"Ticket {i + 1}/{len(df)}: {issue_preview}...")
        print(f"  Company: {ticket['Company']} | Subject: {str(ticket['Subject'])[:50]}")
        
        # Process the ticket
        result = process_ticket(ticket, retriever, i + 1)
        
        # Track summary stats
        status = result.get("status", "escalated")
        request_type = result.get("request_type", "product_issue")
        run_summary[status] = run_summary.get(status, 0) + 1
        run_summary[request_type] = run_summary.get(request_type, 0) + 1
        
        if "injection" in result.get("safety_reason", "").lower():
            run_summary["injections"] += 1
        if "malicious" in result.get("safety_reason", "").lower():
            run_summary["malicious"] += 1
        if "out of scope" in result.get("safety_reason", "").lower():
            run_summary["out_of_scope"] += 1
        
        # Build output row — MUST match problem_statement.md exactly
        output_row = {
            "issue": ticket["Issue"],
            "subject": ticket["Subject"],
            "company": ticket["Company"],
            "status": result["status"],
            "product_area": result["product_area"],
            "response": result["response"],
            "justification": result["justification"],
            "request_type": result["request_type"],
        }
        results.append(output_row)
        
        # Log the processing
        log_ticket_processing(i + 1, ticket, result)
        
        print(f"  [OK] Status: {result['status'].upper()} | Type: {result['request_type']} | Area: {result['product_area']}")
        
        # Rate limit delay between tickets that used the LLM
        # (safety_reason == 'clean' means it DID go through the LLM pipeline)
        if result.get("safety_reason", "clean") == "clean":
            time.sleep(OPENAI_DELAY_SECONDS)
    
    # Write output CSV
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False)
    print(f"\n{'=' * 60}")
    print(f"[DONE] Results saved to: {output_csv}")
    print(f"{'=' * 60}")
    
    # Print summary
    print(f"\n[SUMMARY]")
    print(f"   Total: {len(results)} tickets")
    print(f"   Replied: {run_summary['replied']} | Escalated: {run_summary['escalated']}")
    print(f"   Product Issues: {run_summary['product_issue']}")
    print(f"   Bugs: {run_summary['bug']}")
    print(f"   Invalid: {run_summary['invalid']}")
    print(f"   Feature Requests: {run_summary['feature_request']}")
    print(f"   Safety: {run_summary['injections']} injections, {run_summary['malicious']} malicious, {run_summary['out_of_scope']} OOS")
    
    # Log summary
    log_run_summary(len(results), run_summary)
    
    return results


def validate_against_samples():
    """Validate agent output against sample tickets with expected values."""
    if not SAMPLE_CSV.exists():
        print(f"Sample file not found: {SAMPLE_CSV}")
        return
    
    print("\n[*] Validating against sample tickets...")
    sample_df = pd.read_csv(SAMPLE_CSV)
    
    retriever = CorpusRetriever(CORPUS_DIR)
    
    correct = 0
    total = len(sample_df)
    
    for i, row in sample_df.iterrows():
        ticket = {
            "Issue": row.get("Issue", ""),
            "Subject": row.get("Subject", ""),
            "Company": row.get("Company", "None"),
        }
        
        result = process_ticket(ticket, retriever, i + 1)
        
        expected_status = str(row.get("Status", "")).lower().strip()
        expected_type = str(row.get("Request Type", "")).lower().strip()
        
        actual_status = result["status"].lower()
        actual_type = result["request_type"].lower()
        
        status_match = actual_status == expected_status
        type_match = actual_type == expected_type
        
        if status_match and type_match:
            correct += 1
            marker = "[PASS]"
        else:
            marker = "[FAIL]"
        
        issue_preview = str(ticket["Issue"])[:50].replace("\n", " ")
        print(f"  {marker} #{i+1}: {issue_preview}...")
        if not status_match:
            print(f"      Status: expected={expected_status}, got={actual_status}")
        if not type_match:
            print(f"      Type: expected={expected_type}, got={actual_type}")
        
        time.sleep(OPENAI_DELAY_SECONDS)
    
    print(f"\n  Score: {correct}/{total} ({100*correct/total:.1f}%)")


def show_corpus_stats():
    """Display corpus statistics."""
    retriever = CorpusRetriever(CORPUS_DIR)
    stats = retriever.get_corpus_stats()
    
    print("\n[CORPUS STATS]")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Unique source files: {stats['unique_sources']}")
    print(f"\n  By company:")
    for company, count in sorted(stats['company_distribution'].items()):
        print(f"    {company}: {count} chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(INPUT_CSV),
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_CSV),
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate against sample tickets"
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show corpus statistics"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.stats:
        show_corpus_stats()
        return
    
    if not validate_environment():
        sys.exit(1)
    
    if args.validate:
        validate_against_samples()
        return
    
    process_tickets(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
