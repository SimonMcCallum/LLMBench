"""
LLM-Bench Daemon — Git-based inbox/outbox task processor

Watches .claude/inbox/ for new task files, processes them, and writes
results to .claude/outbox/. Supports two execution modes:

1. Structured benchmark tasks (YAML frontmatter) → runs benchmark directly
2. Free-form Claude tasks (no frontmatter) → runs `claude -p` headlessly

Communication protocol:
  1. Push a .md file to .claude/inbox/ (e.g., 001_benchmark_qwen.md)
  2. Daemon detects on next poll (via Task Scheduler every 5 min)
  3. Parses frontmatter: structured → runner.py, free-form → claude -p
  4. Writes result to .claude/outbox/<same_name>
  5. Commits and pushes result + any code changes

Bidirectional Q&A:
  - Claude writes questions to .claude/questions/<task>_Q<n>.md
  - User answers by pushing to .claude/answers/<task>_Q<n>.md
  - Daemon resumes task with answers on next poll

Evolved from NNCONFIDENCE/claude_daemon.py.

Usage:
    python service/daemon.py                 # Process once (for Task Scheduler)
    python service/daemon.py --interval 300  # Poll loop every 5 min
    python service/daemon.py --dry-run       # Show what would be processed
"""

import argparse
import json
import os
import subprocess
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================
# PATHS
# ============================================================

REPO_DIR = Path(__file__).parent.parent.resolve()
CLAUDE_DIR = REPO_DIR / ".claude"
INBOX_DIR = CLAUDE_DIR / "inbox"
OUTBOX_DIR = CLAUDE_DIR / "outbox"
PROCESSING_DIR = CLAUDE_DIR / "processing"
QUESTIONS_DIR = CLAUDE_DIR / "questions"
ANSWERS_DIR = CLAUDE_DIR / "answers"
DECISIONS_FILE = CLAUDE_DIR / "decisions.md"
STATE_FILE = CLAUDE_DIR / "daemon_state.json"
LOG_FILE = CLAUDE_DIR / "daemon.log"

ALLOWED_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
NEEDS_INPUT_MARKER = "NEEDS_INPUT"


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================
# GIT OPERATIONS
# ============================================================

def run_git(args, cwd=None):
    """Run a git command and return stdout."""
    if cwd is None:
        cwd = REPO_DIR
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_pull(logger):
    """Fetch and pull latest changes."""
    try:
        run_git(["fetch", "origin"])
        try:
            run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
            run_git(["pull", "--rebase", "origin"])
        except RuntimeError:
            for branch in ["main", "master"]:
                try:
                    run_git(["pull", "--rebase", "origin", branch])
                    break
                except RuntimeError:
                    continue
        return True
    except RuntimeError as e:
        logger.warning(f"Pull failed: {e}")
        return False


def commit_and_push(task_name, logger, commit_msg=None):
    """Commit all changes and push."""
    try:
        run_git(["add", f".claude/outbox/{task_name}"])

        for subdir in ["questions", "processing"]:
            try:
                run_git(["add", f".claude/{subdir}/"])
            except RuntimeError:
                pass

        try:
            run_git(["add", ".claude/decisions.md"])
        except RuntimeError:
            pass

        # Stage results (leaderboard, history)
        try:
            run_git(["add", "results/"])
        except RuntimeError:
            pass

        # Stage any other modified files (skip inbox, state)
        status = run_git(["status", "--porcelain"])
        for line in status.split("\n"):
            if not line.strip():
                continue
            filepath = line[3:].strip().strip('"')
            if ".claude/inbox/" in filepath or "daemon_state" in filepath:
                continue
            if "data/" in filepath:
                continue  # Don't commit large data files
            if line[0] in ("M", "A", "?", " ") and line[1] in ("M", "A", "?", " "):
                try:
                    run_git(["add", filepath])
                except RuntimeError:
                    pass

        if commit_msg is None:
            commit_msg = (
                f"[llm-bench-daemon] Processed: {task_name}\n\n"
                f"Automatically processed by LLM-Bench daemon"
            )
        run_git(["commit", "-m", commit_msg])
        logger.info("Changes committed")

        run_git(["push"])
        logger.info("Pushed to remote")
        return True

    except RuntimeError as e:
        logger.error(f"Git commit/push failed: {e}")
        return False


# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"processed": {}, "last_poll": None}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(STATE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


# ============================================================
# TASK DETECTION
# ============================================================

def find_new_tasks(state, logger):
    """Find inbox files that haven't been processed yet."""
    if not INBOX_DIR.exists():
        return []

    tasks = []
    for f in sorted(INBOX_DIR.glob("*.md")):
        name = f.name
        try:
            file_hash = run_git(
                ["log", "-1", "--format=%H", "--", str(f.relative_to(REPO_DIR))]
            )
        except RuntimeError:
            file_hash = "untracked"

        processed_hash = state["processed"].get(name, {}).get("commit_hash")
        if file_hash != processed_hash and file_hash:
            tasks.append({"path": f, "name": name, "commit_hash": file_hash})
            logger.info(f"New/updated task: {name}")

    return tasks


def check_pending_tasks(state, logger):
    """Check tasks waiting for user input that now have answers."""
    resumed = []
    for task_name, info in list(state["processed"].items()):
        if info.get("status") != "pending_input":
            continue

        task_stem = Path(task_name).stem
        answers = {}
        questions = {}

        if ANSWERS_DIR.exists():
            for f in sorted(ANSWERS_DIR.glob(f"{task_stem}_Q*.md")):
                with open(f, "r", encoding="utf-8") as af:
                    answers[f.name] = af.read()

        if QUESTIONS_DIR.exists():
            for f in sorted(QUESTIONS_DIR.glob(f"{task_stem}_Q*.md")):
                with open(f, "r", encoding="utf-8") as qf:
                    questions[f.name] = qf.read()

        if not questions:
            continue

        if all(q in answers for q in questions):
            logger.info(f"Task {task_name}: all questions answered, resuming")
            resumed.append({
                "name": task_name,
                "path": INBOX_DIR / task_name,
                "questions": questions,
                "answers": answers,
                "original_content": info.get("task_content", ""),
            })

    return resumed


# ============================================================
# TASK EXECUTION
# ============================================================

def parse_frontmatter(content: str) -> tuple:
    """Parse YAML frontmatter from task file. Returns (config_dict, body_text)."""
    import re
    import yaml

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        return None, content

    try:
        config = yaml.safe_load(match.group(1))
        body = match.group(2)
        return config, body
    except yaml.YAMLError:
        return None, content


def run_benchmark_task(config, body, task_name, logger):
    """Execute a structured benchmark task directly."""
    logger.info(f"Running benchmark task: {task_name}")
    logger.info(f"Config: {json.dumps(config, default=str)}")

    try:
        # Import runner here to avoid circular imports
        sys.path.insert(0, str(REPO_DIR))
        from service.runner import run_benchmark

        result = run_benchmark(config)
        return {
            "success": True,
            "result": json.dumps(result, indent=2, default=str),
            "type": "benchmark",
        }
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "result": None,
            "type": "benchmark",
        }


def run_claude_task(content, task_name, logger, decisions_context="", qa_context=None):
    """Execute a free-form task via claude -p."""
    logger.info(f"Running Claude Code for: {task_name}")

    parts = [
        f"You are working on the LLM-Bench project at {REPO_DIR}. "
        f"This task was submitted remotely via .claude/inbox/{task_name}.",
    ]

    if decisions_context:
        parts.append(f"\n\nACCUMULATED CONTEXT:\n---\n{decisions_context}\n---")

    if qa_context:
        parts.append(f"\n\nPREVIOUS Q&A FOR THIS TASK:\n{qa_context}")

    parts.append(f"\n\nTASK:\n{content}")

    parts.append(
        f"\n\nINSTRUCTIONS:\n"
        f"- Complete the task and provide a clear summary.\n"
        f"- If you need clarification, write questions to .claude/questions/ "
        f"and include '{NEEDS_INPUT_MARKER}' in your response.\n"
    )

    prompt = "\n".join(parts)

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--allowedTools", ALLOWED_TOOLS, "--output-format", "json"],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=7200,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr[:2000], "result": None, "type": "claude"}

        try:
            output = json.loads(result.stdout)
            return {
                "success": True,
                "result": output.get("result", result.stdout),
                "cost": output.get("cost_usd"),
                "duration": output.get("duration_ms"),
                "type": "claude",
            }
        except json.JSONDecodeError:
            return {"success": True, "result": result.stdout, "type": "claude"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timed out after 2 hours", "result": None, "type": "claude"}
    except FileNotFoundError:
        return {"success": False, "error": "claude CLI not found", "result": None, "type": "claude"}


def write_response(task_name, result, logger):
    """Write task result to outbox."""
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    response_path = OUTBOX_DIR / task_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = f"# Response: {task_name}\n"
    content += f"**Processed:** {timestamp}\n"
    content += f"**Type:** {result.get('type', 'unknown')}\n\n"

    if result["success"]:
        if result.get("cost"):
            content += f"**Cost:** ${result['cost']:.4f}\n"
        if result.get("duration"):
            content += f"**Duration:** {result['duration'] / 1000:.1f}s\n"
        content += f"\n---\n\n{result['result']}\n"
    else:
        content += f"**Status:** FAILED\n"
        content += f"**Error:** {result['error']}\n"

    with open(response_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Response written to: {response_path}")


def process_task(task, state, logger, dry_run=False):
    """Process a single task from inbox."""
    task_path = task["path"]
    task_name = task["name"]

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Processing: {task_name}")

    with open(task_path, "r", encoding="utf-8") as f:
        task_content = f.read()

    logger.info(f"Task content ({len(task_content)} chars):\n{task_content[:500]}")

    if dry_run:
        config, body = parse_frontmatter(task_content)
        task_type = "benchmark" if config and config.get("type") == "benchmark" else "claude"
        logger.info(f"[DRY RUN] Would process as: {task_type}")
        return

    # Move to processing
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    processing_marker = PROCESSING_DIR / task_name
    processing_marker.write_text(f"Started: {datetime.now().isoformat()}\n")

    # Determine task type
    config, body = parse_frontmatter(task_content)
    decisions_context = ""
    if DECISIONS_FILE.exists():
        try:
            decisions_context = DECISIONS_FILE.read_text(encoding="utf-8")
        except Exception:
            pass

    if config and config.get("type") in ("benchmark", "compare", "calibration"):
        result = run_benchmark_task(config, body, task_name, logger)
    else:
        result = run_claude_task(task_content, task_name, logger, decisions_context)

    # Write response
    write_response(task_name, result, logger)

    # Clean up processing marker
    if processing_marker.exists():
        processing_marker.unlink()

    # Check if needs input
    result_text = str(result.get("result", ""))
    needs_input = NEEDS_INPUT_MARKER in result_text

    if needs_input:
        pushed = commit_and_push(
            task_name, logger,
            commit_msg=f"[llm-bench-daemon] Questions for: {task_name}",
        )
        state["processed"][task_name] = {
            "commit_hash": task.get("commit_hash", ""),
            "processed_at": datetime.now().isoformat(),
            "status": "pending_input",
            "task_content": task_content,
            "success": result["success"],
        }
    else:
        pushed = commit_and_push(task_name, logger)
        state["processed"][task_name] = {
            "commit_hash": task.get("commit_hash", ""),
            "processed_at": datetime.now().isoformat(),
            "status": "completed",
            "success": result["success"],
        }

    save_state(state)
    status = "NEEDS_INPUT" if needs_input else ("OK" if result["success"] else "FAILED")
    logger.info(f"Task {task_name}: {status}")


def resume_task(resumed_task, state, logger):
    """Re-run a task that was waiting for input, now with answers."""
    task_name = resumed_task["name"]
    questions = resumed_task["questions"]
    answers = resumed_task["answers"]
    task_content = resumed_task["original_content"]

    if not task_content and resumed_task["path"].exists():
        task_content = resumed_task["path"].read_text(encoding="utf-8")

    qa_parts = []
    for qname in sorted(questions.keys()):
        qa_parts.append(f"**Question ({qname}):**\n{questions[qname].strip()}")
        if qname in answers:
            qa_parts.append(f"**Answer:**\n{answers[qname].strip()}")
    qa_context = "\n\n".join(qa_parts)

    decisions_context = ""
    if DECISIONS_FILE.exists():
        try:
            decisions_context = DECISIONS_FILE.read_text(encoding="utf-8")
        except Exception:
            pass

    result = run_claude_task(task_content, task_name, logger, decisions_context, qa_context)
    write_response(task_name, result, logger)

    pushed = commit_and_push(task_name, logger)
    state["processed"][task_name] = {
        "commit_hash": state["processed"].get(task_name, {}).get("commit_hash", ""),
        "processed_at": datetime.now().isoformat(),
        "status": "completed",
        "success": result["success"],
    }
    save_state(state)
    logger.info(f"Resumed task {task_name}: {'OK' if result['success'] else 'FAILED'}")


# ============================================================
# MAIN LOOP
# ============================================================

def poll_once(logger, dry_run=False):
    """Single poll iteration."""
    state = load_state()

    git_pull(logger)

    # Check resumed tasks
    resumed_tasks = check_pending_tasks(state, logger)
    for resumed in resumed_tasks:
        resume_task(resumed, state, logger)

    # Find new tasks
    tasks = find_new_tasks(state, logger)
    if tasks:
        logger.info(f"Found {len(tasks)} new task(s)")
        for task in tasks:
            process_task(task, state, logger, dry_run=dry_run)
    elif not resumed_tasks:
        logger.info("No new tasks")

    state["last_poll"] = datetime.now().isoformat()
    save_state(state)


def main():
    parser = argparse.ArgumentParser(description="LLM-Bench Daemon")
    parser.add_argument("--interval", type=int, default=0,
                        help="Poll interval in seconds (0 = once and exit)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without running")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("LLM-BENCH DAEMON")
    logger.info(f"  Repo:     {REPO_DIR}")
    logger.info(f"  Inbox:    {INBOX_DIR}")
    logger.info(f"  Outbox:   {OUTBOX_DIR}")
    logger.info(f"  Mode:     {'dry-run' if args.dry_run else 'poll' if args.interval else 'once'}")
    logger.info("=" * 60)

    # Ensure directories exist
    for d in [INBOX_DIR, OUTBOX_DIR, PROCESSING_DIR, QUESTIONS_DIR, ANSWERS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.interval > 0:
        while True:
            try:
                poll_once(logger, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"Poll error: {e}", exc_info=True)
            time.sleep(args.interval)
    else:
        poll_once(logger, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
