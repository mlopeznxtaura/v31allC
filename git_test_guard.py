#!/usr/bin/env python3
"""git_test_guard.py - Automated self-healing version control transaction guard.
Runs stage3 inference tests inside WSL, parses metrics, and auto-commits on success.
"""

import os
import sys
import subprocess
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = str(ROOT_DIR)

print("🚦 [Git Test Guard] Initializing automated validation guard...")

# 1. Run integration tests under WSL environment
print("🔍 Running test suite test_full_cycle_4563.py inside WSL...")
test_cmd = ["wsl", "python3", "/mnt/d/NextAura/v31all_1/v31allC/test_full_cycle_4563.py"]

try:
    res = subprocess.run(test_cmd, capture_output=True, text=True, cwd=REPO_DIR)
except Exception as ex:
    print(f"❌ [FAIL] Failed to execute WSL test runner process: {ex}")
    sys.exit(1)

# Check execution outputs
stdout = res.stdout or ""
stderr = res.stderr or ""
exit_code = res.returncode

print("\n--- TEST RUN OUTPUT ---")
print(stdout)
if stderr:
    print("--- STDERR ---")
    print(stderr)
print("-----------------------\n")

# Detect failure signals (non-zero exit, exceptions, tracebacks)
has_failed = (exit_code != 0) or ("traceback" in stdout.lower()) or ("exception" in stdout.lower()) or ("error" in stdout.lower() and "unsloth" not in stdout.lower())

if has_failed:
    print("🚨 [ABORT] Test suite execution FAILED or exceptions detected. Version control transaction blocked!")
    # Optional: could write a debug crash dump here
    sys.exit(1)

print("✅ [PASS] Integration tests completed successfully with exit code 0.")

# 2. Extract metrics from stdout for the commit message
records_match = re.search(r"records\s*:\s*(\d+)", stdout)
v_mean_match = re.search(r"V_mean=([\d.]+)", stdout)
gate_match = re.search(r"gate OPEN:\s*(\d+)/(\d+)", stdout)

num_records = records_match.group(1) if records_match else "unknown"
v_mean = float(v_mean_match.group(1)) if v_mean_match else 0.0
gate_open = gate_match.group(1) if gate_match else "0"
gate_total = gate_match.group(2) if gate_match else "0"

print(f"📈 Telemetry Metrics: Records={num_records} | V_mean={v_mean:.4f} | Gate Open={gate_open}/{gate_total}")

# 3. Check Git Status
status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_DIR)
changes = status_res.stdout.strip().split("\n")
changes = [c.strip() for c in changes if c.strip()]

if not changes:
    print("ℹ️ [IDLE] No modifications or untracked changes found in repository. Nothing to commit.")
    sys.exit(0)

print(f"📂 Detected {len(changes)} modified/untracked files:")
modified_files = []
for change in changes:
    print(f"  └─ {change}")
    # Extract file path
    parts = change.split(maxsplit=1)
    if len(parts) == 2:
         modified_files.append(parts[1])

# Determine appropriate scope from changes
scope = "core"
if any("view.py" in f for f in modified_files):
    scope = "stage2/view"
elif any("inference_engine" in f for f in modified_files):
    scope = "stage3/inference"
elif any("stage1" in f for f in modified_files):
    scope = "stage1/core"

# 4. Construct Conventional Commit Message
commit_header = f"test-guard({scope}): pass (V_mean={v_mean:.4f}, gate={gate_open}/{gate_total})"
commit_body = "Automated commit triggered by Git Test Guard after successful validation cycle.\n\n"
commit_body += "Modified files:\n" + "\n".join(f"- {f}" for f in modified_files) + "\n\n"
commit_body += "Test metrics:\n"
commit_body += f"- Total parsed records: {num_records}\n"
commit_body += f"- Active V gate ratio: {gate_open}/{gate_total} OPEN\n"
commit_body += f"- Trajectory V_mean: {v_mean:.4f}"

commit_msg_file = ROOT_DIR / ".commit_msg.tmp"
commit_msg_file.write_text(f"{commit_header}\n\n{commit_body}")

print(f"\n📝 Proposed Commit Header: {commit_header}")

# 5. Execute git transaction
try:
    print("🚀 Staging modifications...")
    subprocess.run(["git", "add", "."], check=True, cwd=REPO_DIR)
    
    print("💾 Creating git commit checkpoint...")
    subprocess.run(["git", "commit", "-F", str(commit_msg_file)], check=True, cwd=REPO_DIR)
    
    print("🎉 [SUCCESS] Version control transaction successfully committed!")
    
    # Cleanup temp file
    if commit_msg_file.exists():
        os.remove(commit_msg_file)
        
except Exception as ex:
    print(f"❌ [FAIL] Git commit transaction failed: {ex}")
    sys.exit(1)
