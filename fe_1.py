"""
Feature Engineering Agent
--------------------------
Reads a CSV dataset, builds a rich summary (schema, stats, sample rows),
sends it to an LLM, and gets back:
  1. Feature engineering suggestions (with reasoning)
  2. Auto-generated Python code that implements those features

Requirements:
    pip install groq python-dotenv pandas jupyter nbformat nbconvert ipykernel

Setup:
    GROQ_API_KEY=gsk_your_key_here  in .env
    Get your free key at: https://console.groq.com

Usage:
    python feature_engineering_agent.py
"""

import os
import re
import sys
import time
import textwrap
import traceback
import subprocess
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in .env file.")

MODEL       = "openai/gpt-oss-120b"  # Groq model ID (no :groq suffix needed)
MAX_RETRIES      = 5
BACKOFF          = 2    # seconds, doubles each retry
MAX_EXEC_RETRIES = 3    # max times agent will fix and re-run code on error
EXEC_RETRY_DELAY = 3    # seconds to wait between execution retries (rate limit buffer)


client = Groq(api_key=GROQ_API_KEY)

# ── Code Post-Processor ───────────────────────────────────────────────────────

def fix_code_output(code: str) -> str:
    """
    Auto-fix common LLM code generation mistakes:
    1. Literal \n inside f-strings → actual newline + new print statement
    2. print(f"\n✓ ...") → print() then print(f"✓ ...")
    3. print("\nText") → print() then print("Text")
    """
    import re

    # Fix: print(f"\n{...}") or print("\nText") → print(); print(f"{...}") 
    # Pattern: opening quote followed by \n
    def fix_print_newline(match):
        prefix = match.group(1)   # print( or print(f
        quote  = match.group(2)   # " or '
        rest   = match.group(3)   # everything after \n inside the string
        return f'print()\n{prefix}{quote}{rest}'

    code = re.sub(
        r'(print\(f?)([\'"])\s*\\n',
        fix_print_newline,
        code
    )

    # Fix: f"\n something" anywhere (not just print) → actual newline char
    code = re.sub(r'(?<!\\)\\n', '\n', code)

    return code



# ── Retry wrapper ─────────────────────────────────────────────────────────────

def call_llm(messages: list[dict], max_tokens: int = 8192) -> str:
    """Call LLM with exponential backoff on rate limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "429" in err or
                "rate_limit" in err.lower() or
                "rate limit" in err.lower() or
                "too many" in err.lower()
            )
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = BACKOFF ** attempt
                print(f"  ⚠ Rate limit hit — retrying in {wait}s "
                      f"(attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise


# ── Dataset Summariser ────────────────────────────────────────────────────────

def summarise_dataset(df: pd.DataFrame, target_col: str = None,
                       n_sample: int = 5) -> str:
    """
    Build a compact, token-efficient dataset summary for the LLM.
    Includes: schema, per-column stats, sample rows, and null info.
    """
    lines = []

    # 1. Basic info
    lines.append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    if target_col:
        lines.append(f"Target column: {target_col}")
    lines.append("")

    # 2. Column schema + stats
    lines.append("── Column Details ──────────────────────────────────────")
    for col in df.columns:
        dtype   = str(df[col].dtype)
        n_null  = int(df[col].isna().sum())
        pct_null = round(n_null / len(df) * 100, 1)
        n_unique = int(df[col].nunique())
        tag     = " ← TARGET" if col == target_col else ""

        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            lines.append(
                f"  {col} [{dtype}]{tag}\n"
                f"    nulls={pct_null}%  unique={n_unique}  "
                f"min={stats['min']:.3g}  max={stats['max']:.3g}  "
                f"mean={stats['mean']:.3g}  std={stats['std']:.3g}"
            )
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            lines.append(
                f"  {col} [datetime]{tag}\n"
                f"    nulls={pct_null}%  "
                f"range={df[col].min()} → {df[col].max()}"
            )
        else:
            top_vals = df[col].value_counts().head(5).to_dict()
            lines.append(
                f"  {col} [{dtype}]{tag}\n"
                f"    nulls={pct_null}%  unique={n_unique}  "
                f"top_values={top_vals}"
            )
        lines.append("")

    # 3. Sample rows
    lines.append("── Sample Rows (first 5) ───────────────────────────────")
    lines.append(df.head(n_sample).to_string(index=False))
    lines.append("")

    # 4. Correlations with target (numeric only)
    if target_col and target_col in df.columns:
        numeric_df = df.select_dtypes(include="number")
        if target_col in numeric_df.columns and len(numeric_df.columns) > 1:
            corr = numeric_df.corr()[target_col].drop(target_col).sort_values(
                key=abs, ascending=False
            ).head(10)
            lines.append("── Top Correlations with Target ────────────────────────")
            lines.append(corr.to_string())
            lines.append("")

    return "\n".join(lines)


# ── Agent 1: Feature Suggestion Agent ────────────────────────────────────────

def suggestion_agent(dataset_summary: str, target_col: str = None) -> str:
    """
    Agent 1: Analyse the dataset summary and suggest feature engineering ideas.
    Returns structured suggestions with reasoning.
    """
    print("\n[Agent 1] Generating feature engineering suggestions...")

    target_context = (
        f"The goal is to predict '{target_col}'." if target_col
        else "No specific target column — suggest generally useful features."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior data scientist specialising in feature engineering. "
                "Analyse the dataset summary provided and suggest impactful features to engineer. "
                "For each suggestion:\n"
                "  - Name the new feature\n"
                "  - Explain WHY it would be useful (business/statistical reasoning)\n"
                "  - Specify which existing columns are needed\n"
                "  - Rate impact: High / Medium / Low\n"
                "Group suggestions by type: "
                "Datetime Features, Interaction Features, Aggregation Features, "
                "Encoding Features, Ratio/Normalisation Features, Domain-Specific Features.\n"
                "Also flag any data quality issues you notice."
                "Always write in bullet points and not in tables."
            )
        },
        {
            "role": "user",
            "content": (
                f"{target_context}\n\n"
                f"Here is the dataset summary:\n\n{dataset_summary}\n\n"
                "Suggest all valuable features to engineer."
            )
        }
    ]

    suggestions = call_llm(messages, max_tokens=8192)
    print("  ✓ Suggestions received")
    return suggestions


# ── Agent 2: Code Generation Agent (chunked by feature group) ────────────────

def _parse_feature_groups(suggestions: str) -> dict[str, str]:
    """
    Split the suggestions text into groups by section header (### ...).
    Returns a dict of {group_name: group_text}.
    Falls back to a single group if no headers found.
    """
    import re
    groups = {}
    parts = re.split(r'(###[^\n]+)', suggestions)
    if len(parts) <= 1:
        return {"All Features": suggestions}
    current_header = "General"
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("###"):
            current_header = part.lstrip("#").strip()
        else:
            if current_header not in groups:
                groups[current_header] = ""
            groups[current_header] += part + "\n"
    # Remove empty groups
    return {k: v.strip() for k, v in groups.items() if v.strip()}


def code_agent(dataset_summary: str, suggestions: str,
               target_col: str = None, csv_path: str = "your_file.csv") -> str:
    """
    Agent 2: Generate Python/pandas code chunked by feature group.
    Each group is sent as a separate LLM call to avoid token truncation.
    All partial code blocks are merged into one final script.
    """
    print("\n[Agent 2] Generating feature engineering code...")

    groups = _parse_feature_groups(suggestions)
    total = len(groups)
    print(f"  → Split into {total} feature group(s): {list(groups.keys())}")

    all_code_blocks = []
    all_imports = set()

    system_prompt = (
        "You are an expert Python/pandas developer. "
        "Write clean, production-ready Python code for the feature engineering section provided. "
        "Requirements:\n"
        "  1. Assume df_copy already exists as a copy of the original DataFrame.\n"
        "  2. Add a comment above each feature explaining what it does.\n"
        "  3. Wrap any datetime parsing in try/except.\n"
        "  4. Do NOT redefine df_copy or add imports — those will be handled separately.\n"
        "  5. Implement EVERY feature mentioned in this section completely.\n"
        "  6. Return ONLY the feature engineering statements, no wrapper code."
    )

    for i, (group_name, group_text) in enumerate(groups.items(), 1):
        print(f"  → Coding group {i}/{total}: '{group_name}'...")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Dataset summary:\n{dataset_summary}\n\n"
                    f"Implement ONLY this feature group — '{group_name}':\n\n{group_text}"
                )
            }
        ]
        block = call_llm(messages, max_tokens=4096)
        # Strip markdown fences
        if "```" in block:
            lines = [l for l in block.splitlines() if not l.strip().startswith("```")]
            block = "\n".join(lines)
        all_code_blocks.append(f"# ── {group_name} ──\n{block.strip()}")
        time.sleep(1)  # respect rate limits between chunks

    # Build final merged script
    merged_body = "\n\n".join(all_code_blocks)

    final_code = f"""import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"{csv_path if csv_path else 'your_file.csv'}")
df_copy = df.copy()

{merged_body}

# ── Summary of engineered features ──
new_cols = [c for c in df_copy.columns if c not in df.columns]
print()
print(f"✓ {{len(new_cols)}} new features engineered:")
for col in new_cols:
    print(f"  • {{col}}  dtype={{df_copy[col].dtype}}")

print()
print("Updated DataFrame shape:", df_copy.shape)
"""
    final_code = fix_code_output(final_code)
    print(f"  ✓ Code generation complete ({len(all_code_blocks)} group(s) merged)")
    return final_code



# ── Agent 3: Self-Review Agent (chunked) ─────────────────────────────────────

# Max characters per review chunk (~3000 tokens input + room for fixes in output)
REVIEW_CHUNK_CHARS = 3000

REVIEW_SYSTEM_PROMPT = (
    "You are an expert Python code reviewer. "
    "You will be given a SECTION of auto-generated pandas feature engineering code. "
    "Find and fix ALL bugs in this section. "
    "Common bugs to look for:\n"
    "  1. Using original string column values (e.g. == 'yes') AFTER those columns were one-hot encoded or mapped to 0/1.\n"
    "  2. Broken newlines — print(f'\\nText') must be split into print(); print(f'Text').\n"
    "  3. Division by zero — add +1 guard when dividing by columns that could be 0.\n"
    "  4. Referencing columns that don't exist in the dataset.\n"
    "  5. Duplicate feature definitions.\n"
    "  6. Any syntax errors or undefined variables.\n"
    "IMPORTANT: Return ONLY the fixed code for this section — "
    "no explanations, no markdown fences, no imports, no df_copy = df.copy()."
)


def _chunk_code_by_section(code: str, max_chars: int = REVIEW_CHUNK_CHARS) -> list[tuple[str, str]]:
    """
    Split code into reviewable chunks by section comments (# ── ... ──).
    Each chunk is (section_header, section_code).
    Falls back to line-count splitting if no headers found.
    """
    import re
    # Split on section headers like: # ── Encoding Features ──
    parts = re.split(r'(# ── .+ ──[^\n]*)', code)

    if len(parts) <= 1:
        # No section headers — split by line count
        lines = code.splitlines()
        step  = max(1, max_chars // 80)   # ~80 chars per line estimate
        return [
            (f"Lines {i+1}-{min(i+step, len(lines))}",
             "\n".join(lines[i:i+step]))
            for i in range(0, len(lines), step)
        ]

    chunks, current_header, current_body = [], "Header", ""
    for part in parts:
        if re.match(r'# ── .+ ──', part):
            if current_body.strip():
                chunks.append((current_header, current_body.strip()))
            current_header, current_body = part.strip(), ""
        else:
            current_body += part

    if current_body.strip():
        chunks.append((current_header, current_body.strip()))

    # Merge tiny chunks (< 200 chars) with the next one to avoid wasted API calls
    merged, i = [], 0
    while i < len(chunks):
        header, body = chunks[i]
        while i + 1 < len(chunks) and len(body) < 200:
            i += 1
            body += "\n\n" + chunks[i][0] + "\n" + chunks[i][1]
        merged.append((header, body))
        i += 1

    return merged


def self_review_agent(code: str, csv_path: str, dataset_summary: str) -> str:
    """
    Agent 3: Reviews and fixes generated code chunk by chunk to stay within
    token limits. Each section (# ── ... ──) is reviewed separately, then
    all fixed sections are reassembled into the final script.
    """
    # ── Extract header and footer (imports + df setup + summary block) ──
    lines = code.splitlines()
    header_lines, footer_lines, body_lines = [], [], []
    in_body = False
    footer_start_patterns = ("# ── Summary", "# ── Outlier", "new_cols =", "print()")

    for i, line in enumerate(lines):
        if not in_body and line.startswith("# ──"):
            in_body = True
        if in_body and any(line.strip().startswith(p) for p in footer_start_patterns):
            footer_lines = lines[i:]
            break
        if in_body:
            body_lines.append(line)
        else:
            header_lines.append(line)

    header = "\n".join(header_lines)
    footer = "\n".join(footer_lines)
    body   = "\n".join(body_lines)

    # ── Chunk the body by section ──
    chunks = _chunk_code_by_section(body)
    total  = len(chunks)
    print(f"\n[Agent 3 - Self Review] Reviewing {total} section(s) for bugs...")

    fixed_sections = []

    for i, (section_name, section_code) in enumerate(chunks, 1):
        print(f"  → Reviewing section {i}/{total}: '{section_name}'...")

        # Skip empty/trivial sections
        if len(section_code.strip()) < 30:
            fixed_sections.append(section_code)
            continue

        messages = [
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Dataset columns for reference:\n{dataset_summary[:800]}\n\n"
                    f"Section '{section_name}':\n\n{section_code}"
                )
            }
        ]

        fixed_section = call_llm(messages, max_tokens=4096)

        # Strip markdown fences
        if "```" in fixed_section:
            flines = [l for l in fixed_section.splitlines()
                      if not l.strip().startswith("```")]
            fixed_section = "\n".join(flines)

        fixed_sections.append(f"{section_name}\n{fixed_section.strip()}")
        time.sleep(1)  # rate limit buffer between review calls

    # ── Reassemble: header + fixed body + footer ──
    fixed_body = "\n\n".join(fixed_sections)
    final = f"{header}\n\n{fixed_body}\n\n{footer}"

    print(f"  ✓ Self-review complete ({total} section(s) reviewed)")
    return final


# ── Code Execution Engine ─────────────────────────────────────────────────────

def _strip_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    if "```" in code:
        lines = [l for l in code.splitlines() if not l.strip().startswith("```")]
        return "\n".join(lines)
    return code


def _inject_csv_saver(code: str, csv_path: str) -> str:
    """
    Inject code at the end of the script to:
      1. Save df_copy as a new engineered CSV
      2. Print the new columns summary
    """
    # Normalise to forward slashes — backslashes cause SyntaxWarning on Windows
    csv_path    = csv_path.replace("\\", "/").replace("\\", "/")
    base        = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv  = f"{base}_engineered.csv"
    save_block  = f"""
# ── Auto-injected by agent: save engineered DataFrame ──
_new_cols = [c for c in df_copy.columns if c not in df.columns]
print()
print(f"✓ {{len(_new_cols)}} new features engineered:")
for _col in _new_cols:
    print(f"  • {{_col:35s}}  dtype={{df_copy[_col].dtype}}")
print()
print(f"Original shape : {{df.shape}}")
print(f"Engineered shape: {{df_copy.shape}}")
df_copy.to_csv("{output_csv}", index=False)
print(f"Engineered dataset saved → {output_csv}")
"""
    # Remove any existing summary/save block to avoid duplicates
    code = re.sub(
        r"# ── (Summary|Auto-injected).*",
        "",
        code,
        flags=re.DOTALL
    )
    return code.rstrip() + "\n" + save_block


def _sanitise_code_for_windows(code: str) -> str:
    """
    Fix two persistent Windows-specific issues in LLM-generated code:
    1. Backslash paths in strings → forward slashes (avoids SyntaxWarning)
    2. Unicode symbols (✓ • ✗ →) in print() → ASCII equivalents
       (avoids cp1252 UnicodeEncodeError on Windows terminals)
    """
    import re

    # Fix backslash paths: pd.read_csv(".\\file.csv") → pd.read_csv("./file.csv")
    def fix_path(m):
        return m.group(0).replace("\\\\", "/").replace("\\", "/")
    code = re.sub(r'(read_csv|to_csv|open)\(["\'][^"\']*["\']\)', fix_path, code)

    # Replace Unicode symbols with ASCII in print statements
    unicode_map = {
        "\u2713": "[OK]",   # ✓
        "\u2717": "[X]",    # ✗
        "\u2714": "[OK]",   # ✔
        "\u2192": "->",     # →
        "\u2022": "*",      # •
        "\u2714": "[OK]",   # ✔
        "\u2705": "[OK]",   # ✅
        "\u274c": "[X]",    # ❌
        "\u231b": "[~]",    # ⏳
        "\u2728": "*",      # ✨
        "\u2718": "[X]",    # ✘
    }
    for uni, ascii_rep in unicode_map.items():
        code = code.replace(uni, ascii_rep)
    # Also replace the literal symbols directly
    literal_map = {
        "✓": "[OK]", "✗": "[X]", "✔": "[OK]", "→": "->",
        "•": "*",    "✅": "[OK]","❌": "[X]", "⏳": "[~]",
        "✨": "*",   "✘": "[X]", "📦": "",    "📐": "",
    }
    for sym, rep in literal_map.items():
        code = code.replace(sym, rep)

    return code


def execute_code(code: str, csv_path: str) -> tuple[bool, str, str]:
    """
    Execute the feature engineering code in an isolated subprocess.
    Returns (success, stdout, stderr).
    Sanitises Windows path and encoding issues before execution.
    """
    # Sanitise before writing to disk
    code = _sanitise_code_for_windows(code)

    exec_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "_fe_exec_temp.py")
    with open(exec_path, "w", encoding="utf-8") as f:
        f.write(code)

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"   # force UTF-8 output on Windows

        result = subprocess.run(
            [sys.executable, exec_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            cwd=os.path.dirname(os.path.abspath(csv_path)) or ".",
            env=env
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TimeoutError: code execution exceeded 120 seconds."
    finally:
        if os.path.exists(exec_path):
            os.remove(exec_path)


def error_fix_agent(code: str, error: str, csv_path: str, dataset_summary: str) -> str:
    """
    Agent 4: Given the code and the runtime error it produced,
    ask the LLM to fix the specific error and return corrected code.
    Keeps fixes targeted — sends only the error + full code context.
    """
    print(f"\n[Agent 4 - Error Fix] Fixing runtime error...")
    print(f"  Error: {error.strip()[:200]}...")

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python debugger. "
                "You will be given Python pandas code and the runtime error it produced. "
                "Fix ONLY the error — do not change any other logic. "
                "Common causes:\n"
                "  1. KeyError — column was already encoded/renamed, use the new column name.\n"
                "  2. TypeError — wrong dtype operation (e.g. string math), add type guards.\n"
                "  3. ZeroDivisionError — add +1 guard.\n"
                "  4. ValueError — mismatched shapes or bad datetime parsing.\n"
                "  5. AttributeError — column no longer exists after get_dummies renamed it.\n"
                "Return ONLY the complete fixed Python code. No explanations. No markdown fences."
            )
        },
        {
            "role": "user",
            "content": (
                f"Dataset summary:\n{dataset_summary[:600]}\n\n"
                f"Runtime error:\n{error}\n\n"
                f"Full code that caused the error:\n\n{code}"
            )
        }
    ]

    fixed = call_llm(messages, max_tokens=8192)
    fixed = _strip_fences(fixed)
    print("  ✓ Fix generated")
    return fixed


def run_with_autofix(code: str, csv_path: str, dataset_summary: str) -> tuple[str, bool]:
    """
    Execute code and automatically fix + retry on errors.
    Returns (final_code, success).
    Respects rate limits with EXEC_RETRY_DELAY between fix attempts.
    """
    print("\n[Execution Engine] Running feature engineering code...")
    current_code = _inject_csv_saver(code, csv_path)

    for attempt in range(1, MAX_EXEC_RETRIES + 1):
        success, stdout, stderr = execute_code(current_code, csv_path)

        if success:
            print(f"  ✓ Execution successful (attempt {attempt})")
            if stdout.strip():
                print("\n── Execution Output ──────────────────────────────────")
                print(stdout.strip())
            return current_code, True

        # Execution failed
        error_msg = stderr or stdout
        print(f"  ✗ Execution failed (attempt {attempt}/{MAX_EXEC_RETRIES})")
        print(f"  Error preview: {error_msg.strip()[:300]}")

        if attempt < MAX_EXEC_RETRIES:
            print(f"  ⏳ Waiting {EXEC_RETRY_DELAY}s before fix attempt (rate limit buffer)...")
            time.sleep(EXEC_RETRY_DELAY)
            current_code = error_fix_agent(current_code, error_msg, csv_path, dataset_summary)
            current_code = _inject_csv_saver(current_code, csv_path)  # re-inject saver after fix
        else:
            print(f"  ✗ Max fix attempts ({MAX_EXEC_RETRIES}) reached. Saving last fixed version.")
            print("\n── Final Error ───────────────────────────────────────")
            print(error_msg.strip())

    return current_code, False

# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(suggestions: str, code: str, csv_path: str):
    """Save suggestions to a markdown file and final code to a Python file."""
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # Save suggestions
    suggestions_path = f"{base}_feature_suggestions.md"
    with open(suggestions_path, "w", encoding="utf-8") as f:
        f.write(f"# Feature Engineering Suggestions\n")
        f.write(f"Dataset: `{csv_path}`\n\n")
        f.write(suggestions)
    print(f"  ✓ Suggestions saved  → {suggestions_path}")

    # Save final (post-execution, post-fix) code
    code_path = f"{base}_feature_engineering.py"
    clean_code = _strip_fences(code)
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(f'"""Feature engineering code for {csv_path} — verified by execution agent"""\n\n')
        f.write(clean_code)
    print(f"  ✓ Final code saved   → {code_path}")

    return suggestions_path, code_path


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run(csv_path: str, target_col: str = None):
    print("=" * 60)
    print("  Feature Engineering Agent")
    print(f"  CSV    : {csv_path}")
    print(f"  Target : {target_col or 'None specified'}")
    print(f"  Model  : {MODEL}")
    print("=" * 60)

    # Load dataset (already loaded in __main__, but support direct calls too)
    print(f"\n[Loading] Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass
    print(f"  ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    # Build summary
    print("\n[Summarising] Building dataset summary...")
    summary = summarise_dataset(df, target_col=target_col)
    print(f"  ✓ Summary built (~{len(summary)//4} tokens)")

    # Agent 1: Suggestions
    suggestions = suggestion_agent(summary, target_col=target_col)

    # Agent 2: Code
    code = code_agent(summary, suggestions, target_col=target_col, csv_path=csv_path)

    # Agent 3: Self-review and fix
    code = self_review_agent(code, csv_path, summary)

    # Agent 4: Execute + auto-fix errors
    code, exec_success = run_with_autofix(code, csv_path, summary)

    # Save final outputs
    print("\n[Saving] Writing output files...")
    save_outputs(suggestions, code, csv_path)

    # Final status
    print("\n" + "=" * 60)
    if exec_success:
        base       = os.path.splitext(os.path.basename(csv_path))[0]
        output_csv = os.path.join(os.path.dirname(os.path.abspath(csv_path)),
                                  f"{base}_engineered.csv")
        print("  ✅ Pipeline complete — code ran successfully!")
        if os.path.exists(output_csv):
            engineered_df = pd.read_csv(output_csv)
            print(f"  📦 Engineered CSV   → {output_csv}")
            print(f"  📐 Shape            → {engineered_df.shape}")
            new_cols = engineered_df.shape[1] - pd.read_csv(csv_path).shape[1]
            print(f"  ✨ New features      → {new_cols}")
    else:
        print("  ⚠  Pipeline complete — code saved but execution had errors.")
        print("     Check the _feature_engineering.py file and fix manually.")
    print("=" * 60)

    print("\n── Suggestions Preview ───────────────────────────────────\n")
    print(textwrap.shorten(suggestions, width=600, placeholder=" ..."))


# ── CSV Discovery ─────────────────────────────────────────────────────────────

def find_csv_files(search_dir: str = ".") -> list[str]:
    """Recursively find all CSV files in the given directory."""
    csv_files = []
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return sorted(csv_files)


def pick_csv_agent(csv_files: list[str]) -> str:
    """
    Agent 0a: Given a list of CSV file paths, pick the most suitable one
    for feature engineering analysis.
    """
    print("\n[CSV Picker Agent] Choosing best CSV from available files...")
    file_list = "\n".join(f"  {i+1}. {f}" for i, f in enumerate(csv_files))
    messages = [
        {
            "role": "system",
            "content": (
                "You are a data science expert. Given a list of CSV file paths, "
                "pick the single most suitable one for machine learning feature engineering. "
                "Prefer files that sound like structured ML datasets (e.g. customer, sales, churn, titanic, loan, fraud). "
                "Avoid files that look like configs, logs, or metadata. "
                "Reply with ONLY the exact file path — nothing else."
            )
        },
        {
            "role": "user",
            "content": f"Available CSV files:\n{file_list}\n\nWhich file is best for feature engineering?"
        }
    ]
    chosen = call_llm(messages, max_tokens=100).strip()
    # Validate the returned path is in our list
    for f in csv_files:
        if f in chosen or os.path.basename(f) in chosen:
            print(f"  ✓ Selected: {f}")
            return f
    # Fallback to first file if response doesn't match
    print(f"  ✓ Defaulting to: {csv_files[0]}")
    return csv_files[0]


def identify_target_column(df: pd.DataFrame, csv_path: str) -> str | None:
    """
    Rule-based target column identifier — no LLM, fully deterministic.

    Scoring system (higher = more likely to be target):
      +10  Exact name match to known target keywords
      +7   Partial name match to known target keywords
      +5   Last column in the DataFrame (common ML convention)
      +4   Binary column (only 2 unique values)
      +3   Low cardinality integer (2-10 unique values)
      +3   Filename contains a hint matching the column name
      +2   Column is boolean dtype
      -3   Column looks like an ID (monotonic, all unique)
      -3   Column name suggests it is a feature, not a target
      -5   High cardinality continuous float (unlikely target)

    Picks the highest-scoring column. Returns None if top score < 4
    (no confident match found).
    """
    import re
    import numpy as np

    print("\n[Target Identifier] Identifying target column (rule-based)...")

    n_rows   = len(df)
    filename = os.path.splitext(os.path.basename(csv_path))[0].lower()
    columns  = list(df.columns)

    # ── Keyword lists ──────────────────────────────────────────────────────
    EXACT_TARGET_NAMES = {
        # Explicit ML targets
        "target", "label", "labels", "class", "classes", "y", "output",
        # Medical / survival
        "churn", "churned", "survived", "survival", "outcome", "result",
        "fraud", "fraudulent", "default", "defaulted", "converted",
        "purchased", "bought", "clicked", "opened", "responded",
        "approved", "rejected", "passed", "failed", "diagnosed",
        "readmitted", "admitted", "died", "death", "attrition",
        "cancel", "cancelled", "canceled", "renew", "renewed",
        "subscribed", "unsubscribed", "deposit", "loan_status",
        # Numeric regression targets — ONLY exact names, not substrings
        "price", "salary", "revenue", "sales", "profit", "score",
        "rating", "grade", "risk", "hazard", "event",
        "income", "wage", "cost", "spend", "spent", "amount",
        "tax", "tip", "fare", "charges", "bmi", "age",
    }

    PARTIAL_TARGET_KEYWORDS = [
        "target", "label", "churn", "fraud", "default",
        "outcome", "result", "predict", "response", "dependent",
        "flag", "indicator", "survived", "death",
        "purchase", "convert", "cancel", "admit", "diagnos",
        "readmit", "attrition", "subscri", "deposit",
    ]

    FEATURE_KEYWORDS = [
        "id", "_id", "index", "uuid", "key", "code", "name",
        "date", "time", "year", "month", "day", "hour",
        "description", "comment", "note", "text", "address",
        "phone", "email", "url", "ip", "zip", "postal",
        "longitude", "latitude", "lng", "lat",
    ]

    scores   = {}
    evidence = {}

    for col in columns:
        col_lower  = col.lower().strip()
        series     = df[col].dropna()
        n_unique   = df[col].nunique()
        col_dtype  = df[col].dtype
        score      = 0
        reasons    = []

        # ── Positive signals ────────────────────────────────────────────
        # Exact name match
        if col_lower in EXACT_TARGET_NAMES:
            score += 10
            reasons.append(f"exact keyword match ({col_lower})")

        # Partial name match
        elif any(kw in col_lower for kw in PARTIAL_TARGET_KEYWORDS):
            score += 7
            reasons.append(f"partial keyword match")

        # Last column (ML datasets often put target last — weak signal only)
        if col == columns[-1]:
            score += 2
            reasons.append("last column")

        # Binary column
        if n_unique == 2:
            score += 4
            reasons.append("binary (2 unique values)")

        # Low-cardinality integer
        elif 2 < n_unique <= 10 and pd.api.types.is_integer_dtype(col_dtype):
            score += 3
            reasons.append(f"low-cardinality int ({n_unique} unique)")

        # Boolean
        if pd.api.types.is_bool_dtype(col_dtype):
            score += 2
            reasons.append("bool dtype")

        # Filename hint — e.g. "churn.csv" and column "churn"
        if col_lower in filename or filename in col_lower:
            score += 3
            reasons.append("filename hint")

        # Numeric with bounded range [0,1] (probability-style target)
        if pd.api.types.is_numeric_dtype(col_dtype):
            mn, mx = series.min(), series.max()
            if 0 <= mn and mx <= 1 and n_unique > 2:
                score += 2
                reasons.append("bounded [0,1] numeric")

        # ── Negative signals ────────────────────────────────────────────
        # Looks like ID column (monotonic or all unique integers)
        if n_unique == n_rows:
            score -= 3
            reasons.append("all-unique (likely ID)")

        if pd.api.types.is_integer_dtype(col_dtype) and df[col].is_monotonic_increasing:
            score -= 3
            reasons.append("monotonic increasing (likely index)")

        # Name suggests it is a feature
        if any(kw in col_lower for kw in FEATURE_KEYWORDS):
            score -= 3
            reasons.append("feature keyword in name")

        # High-cardinality continuous float (very unlikely to be target)
        if pd.api.types.is_float_dtype(col_dtype) and n_unique > n_rows * 0.8:
            score -= 5
            reasons.append("high-cardinality float (likely continuous feature)")

        # First column penalty (IDs / keys are usually first)
        if col == columns[0] and n_unique == n_rows:
            score -= 2
            reasons.append("first column, all unique")

        # String/object column with 3+ categories = almost certainly a feature
        if col_dtype == object and n_unique >= 3:
            score -= 6
            reasons.append(f"multi-category string ({n_unique} categories) — likely feature")

        # String/object column with 2 categories (yes/no) — weak target
        if col_dtype == object and n_unique == 2:
            score -= 2
            reasons.append("binary string — likely categorical feature")

        # Pure continuous numeric with many unique values is rarely a regression
        # target unless it explicitly matches a keyword (already rewarded above)
        if pd.api.types.is_float_dtype(col_dtype) and n_unique > 20:
            if not any(kw in col_lower for kw in ["price", "salary", "revenue",
                       "sales", "profit", "income", "cost", "amount", "score",
                       "rating", "spend", "wage"]):
                score -= 2
                reasons.append("high-cardinality float without target keyword")

        scores[col]   = score
        evidence[col] = reasons

    # ── Pick winner ────────────────────────────────────────────────────────
    best_col   = max(scores, key=scores.get)
    best_score = scores[best_col]

    # Print scoring table
    print(f"  {'Column':<30} {'Score':>6}  Evidence")
    print(f"  {'-'*30} {'-'*6}  {'-'*40}")
    for col in sorted(scores, key=scores.get, reverse=True)[:8]:
        marker = " <-- TARGET" if col == best_col and best_score >= 4 else ""
        ev     = ", ".join(evidence[col]) if evidence[col] else "no signals"
        print(f"  {col:<30} {scores[col]:>6}  {ev}{marker}")

    if best_score < 4:
        print("\n  [!] No confident target found (top score < 4) — proceeding without target")
        return None

    print(f"\n  [OK] Target identified: '{best_col}'  (score={best_score})")
    return best_col


# ── Auto-Run Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    search_dir = "."

    print("=" * 60)
    print("  Feature Engineering Agent  (Auto Mode)")
    print(f"  Scanning for CSV files in: {os.path.abspath(search_dir)}")
    print("=" * 60)

    # Step 0: Discover CSV files
    csv_files = find_csv_files(search_dir)

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{os.path.abspath(search_dir)}'. "
            "Please place at least one CSV file in the current directory."
        )

    print(f"\n[Discovery] Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  • {f}")

    # Step 0a: Let LLM pick the best CSV (if multiple found)
    if len(csv_files) == 1:
        chosen_csv = csv_files[0]
        print(f"\n[CSV Picker] Only one CSV found — using: {chosen_csv}")
    else:
        chosen_csv = pick_csv_agent(csv_files)

    # Step 0b: Load and let LLM pick the target column
    print(f"\n[Loading] Reading {chosen_csv}...")
    df = pd.read_csv(chosen_csv)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"  ✓ Auto-parsed '{col}' as datetime")
            except (ValueError, TypeError):
                pass
    print(f"  ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    target_col = identify_target_column(df, chosen_csv)

    # Run full pipeline
    run(csv_path=chosen_csv, target_col=target_col)