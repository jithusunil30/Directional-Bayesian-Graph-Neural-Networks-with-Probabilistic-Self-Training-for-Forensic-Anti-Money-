#!/usr/bin/env python3
"""
Interactive Search and Content Retrieval Engine for the AML GNN Research Paper.
Supports keyword lookup, section dumping, table extraction, theorem review, and an interactive shell.

Usage Examples:
    python search_paper.py "accuracy illusion"
    python search_paper.py --section "4.3"
    python search_paper.py --table 7
    python search_paper.py --tables
    python search_paper.py --theorems
    python search_paper.py --interactive
"""

import os
import sys
import re

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MANUSCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESEARCH_PAPER_COMPLETE_DRAFT.md")

def load_manuscript():
    if not os.path.exists(MANUSCRIPT_PATH):
        print(f"Error: Manuscript not found at {MANUSCRIPT_PATH}")
        sys.exit(1)
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def search_text(query, context_lines=3, max_results=12):
    text = load_manuscript()
    lines = text.split("\n")
    matches = []
    
    # Try regex, fallback to literal
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except Exception:
        pattern = re.compile(re.escape(query), re.IGNORECASE)
    
    for idx, line in enumerate(lines):
        if pattern.search(line):
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            snippet = "\n".join(lines[start:end])
            matches.append((idx + 1, snippet))
            
    print("\n" + "=" * 70)
    print(f"[SEARCH RESULTS] FOR QUERY: '{query}' ({len(matches)} match{'es' if len(matches) != 1 else ''} found)")
    print("=" * 70 + "\n")
    
    if not matches:
        print("No matches found. Suggested topics to search:")
        print("  - 'accuracy illusion' or 'falsification' (Phase 2 diagnostics)")
        print("  - 'edge connectivity' or 'connectivity matrix' (86.9% edge severing)")
        print("  - 'homophily' or 'heterophily' (Bitcoin flow topology)")
        print("  - 'Bayesian' or 'epistemic uncertainty' (Dir-Res-BGNN & MC Dropout)")
        print("  - 'peeling chain' or 'smurfing' (Laundering topological patterns)")
        print("  - 'Tier 1' or 'SAR' (3-Tier AML forensic compliance framework)\n")
        return
        
    for line_num, snippet in matches[:max_results]:
        print(f"--- [Line {line_num} in RESEARCH_PAPER_COMPLETE_DRAFT.md] ---")
        print(snippet)
        print("-" * 65 + "\n")
        
    if len(matches) > max_results:
        print(f"... and {len(matches) - max_results} more occurrences in the manuscript.\n")

def get_section(sec_query):
    text = load_manuscript()
    # Split by markdown headers
    pattern = re.compile(r'^(#+\s+.+)$', re.MULTILINE)
    headers = list(pattern.finditer(text))
    
    found = False
    for i, match in enumerate(headers):
        header_text = match.group(1)
        if re.search(re.escape(sec_query), header_text, re.IGNORECASE) or re.search(sec_query, header_text, re.IGNORECASE):
            start_pos = match.start()
            end_pos = headers[i+1].start() if i + 1 < len(headers) else len(text)
            section_body = text[start_pos:end_pos].strip()
            
            print("\n" + "=" * 75)
            print(f"[SECTION VIEW] {header_text}")
            print("=" * 75 + "\n")
            print(section_body)
            print("\n" + "=" * 75 + "\n")
            found = True
            break
            
    if not found:
        print(f"\nSection matching '{sec_query}' not found. Run with --sections to see all available headings.\n")

def get_table(table_query):
    text = load_manuscript()
    pattern = re.compile(r'(###\s+[^#\n]*?(?:Table|\bResults\b|\bBenchmark\b|\bConnectivity\b|\bSynthes\w*\b|\bMatrix\b)[^\n]*\n\n(?:\|[^\n]+\n)+)', re.IGNORECASE)
    tables = pattern.findall(text)
    
    found = False
    for t_idx, t_block in enumerate(tables, 1):
        if str(table_query).strip().isdigit() and int(table_query) == t_idx:
            print("\n" + "=" * 75)
            print(f"[TABLE VIEW] Table #{t_idx}")
            print("=" * 75 + "\n")
            print(t_block.strip())
            print("\n" + "=" * 75 + "\n")
            found = True
            break
        elif str(table_query).lower() in t_block.lower():
            print("\n" + "=" * 75)
            print(f"[TABLE VIEW] Table Match #{t_idx}")
            print("=" * 75 + "\n")
            print(t_block.strip())
            print("\n" + "=" * 75 + "\n")
            found = True
            
    if not found:
        print(f"\nTable '{table_query}' not found. Run with --tables to view the table index.\n")

def list_sections():
    text = load_manuscript()
    sections = re.findall(r'^(#+\s+.+)$', text, re.MULTILINE)
    print("\n" + "=" * 70)
    print("[MANUSCRIPT OUTLINE & SECTIONS]")
    print("=" * 70)
    for s in sections:
        indent = "  " * (s.count('#') - 1)
        print(f"{indent}{s}")
    print("\n")

def list_tables():
    text = load_manuscript()
    # Find all headings followed by markdown tables
    pattern = re.compile(r'###\s+([^#\n]*?(?:Table|\bResults\b|\bBenchmark\b|\bConnectivity\b|\bSynthes\w*\b|\bMatrix\b)[^\n]*)', re.IGNORECASE)
    tables = pattern.findall(text)
    print("\n" + "=" * 70)
    print("[EXPERIMENTAL & TOPOLOGICAL BENCHMARK TABLES]")
    print("=" * 70)
    if not tables:
        # Fallback to any markdown table header
        tables = re.findall(r'(###\s+.*?(?:Nodes|Graph|Splits|Matrix|Benchmark|Results|Evaluation).*)', text)
    for idx, t in enumerate(tables, 1):
        print(f"  • Table {idx}: {t.strip()}")
    print("\n")

def list_theorems():
    text = load_manuscript()
    theorems = re.findall(r'###\s+(2\.\d+\s+[^\n]+|4\.\d+\s+[^\n]+)', text)
    print("\n" + "=" * 70)
    print("[MATHEMATICAL THEOREMS & FORMULATIONS]")
    print("=" * 70)
    for th in theorems:
        print(f"  • {th}")
    print("\n")

def interactive_mode():
    print("\n" + "=" * 70)
    print("       AML GNN RESEARCH PAPER - INTERACTIVE SEARCH ENGINE       ")
    print("=" * 70)
    print("Commands:")
    print("  <search term>    : Search manuscript for keywords or phrases")
    print("  sec <heading>    : Print entire section (e.g., 'sec Accuracy', 'sec 4.3')")
    print("  tbl <number>     : Print specific table (e.g., 'tbl 2', 'tbl 7')")
    print("  outline          : View full table of contents")
    print("  tables           : List all 11 experimental tables")
    print("  theorems         : List all mathematical theorems")
    print("  quit / exit      : Exit interactive search\n")
    
    while True:
        try:
            cmd = input("Search Paper >> ").strip()
            if not cmd:
                continue
            if cmd.lower() in ["quit", "exit", "q"]:
                print("Exiting search engine. Good luck with your paper research!")
                break
            elif cmd.lower() in ["outline", "sections"]:
                list_sections()
            elif cmd.lower() in ["tables", "tbls"]:
                list_tables()
            elif cmd.lower() in ["theorems", "thems", "math"]:
                list_theorems()
            elif cmd.lower().startswith("sec "):
                get_section(cmd[4:].strip())
            elif cmd.lower().startswith("tbl "):
                get_table(cmd[4:].strip())
            else:
                search_text(cmd)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search engine.")
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        interactive_mode()
    else:
        arg1 = sys.argv[1].strip()
        if arg1 in ["--interactive", "-i"]:
            interactive_mode()
        elif arg1 in ["--sections", "-s", "--outline"]:
            list_sections()
        elif arg1 in ["--tables", "-t"]:
            list_tables()
        elif arg1 in ["--theorems", "-m"]:
            list_theorems()
        elif arg1 in ["--section", "-sec"] and len(sys.argv) > 2:
            get_section(" ".join(sys.argv[2:]))
        elif arg1 in ["--table", "-tbl"] and len(sys.argv) > 2:
            get_table(" ".join(sys.argv[2:]))
        else:
            query = " ".join(sys.argv[1:])
            search_text(query)

