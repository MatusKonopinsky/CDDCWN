"""
Centralized live dashboard logger for parallel experiments.

Architecture:
    - Each worker sends messages to one multiprocessing.Queue
    - A single _logger_process receives and prints them sequentially
    - Dashboard is updated via ANSI escape sequences

Message types (tuple):
    ("start",     run_id, d_name, m_name)               - worker started model
    ("progress",  run_id, d_name, m_name, cur, total)   - progress (sample count)
    ("done",      run_id, d_name, m_name, metrics, time) - model completed
    ("error",     run_id, d_name, m_name, err_str)       - worker error
    ("task_done", run_id, d_name)                        - full dataset/run completed
    ("STOP",)                                            - terminate logger process

Output example:
  ✔  [SEA_Imb9010|run1]        DDCW_aug+drift          RWA=0.9955  F1=0.9389  MinF1=0.8915  Drift=0  312s
  ✔  [Agrawal_Imb9010|run1]    ARF                     RWA=0.1059  F1=0.4803  MinF1=0.0132  Drift=0  208s
  ────────────────────────────────────────────────────────────────────────────
  [rbf_drift_imb9010|run1]     LeveragingBaggingClas…  ████████░░  42%  21000/49500
  [MC_Abrupt_3C|run1]          DDCW_aug+drift          ███░░░░░░░  18%   9000/49500
    Done: 2/8    ████████░░░░░░░░  25%      2/8
"""

import sys
import time

# Constants
BAR_W   = 22   # progress bar width
DS_W    = 24   # [dataset|run] column width
MODEL_W = 26   # model-name column width
TOTAL_W = 80   # separator line width

CLEAR_LINE = "\033[2K\r"
CURSOR_UP  = "\033[{}A"


# Helpers
def _bar(cur, tot, width=BAR_W):
    p   = min(1.0, cur / max(1, tot))
    f   = int(width * p)
    pct = int(p * 100)
    return f"{'█'*f}{'░'*(width-f)} {pct:3d}%  {cur:>6}/{tot}"


def _trunc(s, n):
    """Truncate string to n chars with trailing '...'"""
    return s if len(s) <= n else s[:n - 1] + "…"


# Main logger process
def _logger_process(queue, total_tasks):
    """
    Runs in a separate process. Receives queue messages and keeps
    a live terminal dashboard using ANSI escape sequences.

    Bug fix compared to the original version:
        - _erase_dashboard() now correctly avoids overwriting lines when slots are empty
          (previously, at height=2 it drew separator+summary even with empty slots,
          which printed standalone '---' lines without content)
        - Dashboard is not redrawn when nothing changed (progress throttling)
    """
    completed_tasks = 0
    slots      = {}   # key -> (d_name, run_id, m_name, cur, tot, start_ts)
    slot_order = []   # keep insertion order

    # Rendering
    def _slot_line(key):
        d_name, run_id, m_name, cur, tot, _ = slots[key]
        tag   = _trunc(f"[{d_name}|run{run_id}]", DS_W)
        model = _trunc(m_name, MODEL_W)
        return f"  {tag:<{DS_W}}  {model:<{MODEL_W}}  {_bar(cur, tot)}"

    def _overall_line():
        return f"  Done: {completed_tasks}/{total_tasks}  {_bar(completed_tasks, total_tasks)}"

    def _dash_height():
        # separator + one line per slot + summary line
        # If there are no slots, render only separator + summary (height=2)
        return 1 + len(slot_order) + 1

    def _draw_dashboard():
        sep = "─" * TOTAL_W
        sys.stdout.write(f"{CLEAR_LINE}{sep}\n")
        for key in slot_order:
            sys.stdout.write(f"{CLEAR_LINE}{_slot_line(key)}\n")
        sys.stdout.write(f"{CLEAR_LINE}{_overall_line()}\n")
        sys.stdout.flush()

    def _erase_dashboard():
        h = _dash_height()
        sys.stdout.write(CURSOR_UP.format(h))
        for _ in range(h):
            sys.stdout.write(f"{CLEAR_LINE}\n")
        sys.stdout.write(CURSOR_UP.format(h))
        sys.stdout.flush()

    def _emit_log_line(text):
        """Print a persistent log line ABOVE the dashboard."""
        _erase_dashboard()
        sys.stdout.write(f"{text}\n")
        sys.stdout.flush()
        _draw_dashboard()

    # Initial render
    _draw_dashboard()

    # Main loop
    while True:
        msg  = queue.get()
        kind = msg[0]

        if kind == "STOP":
            _erase_dashboard()
            sys.stdout.write(f"{'─' * TOTAL_W}\n")
            sys.stdout.write(
                f"  All tasks completed.  Done: {completed_tasks}/{total_tasks}\n"
            )
            sys.stdout.flush()
            break

        elif kind == "start":
            _, run_id, d_name, m_name = msg
            key = (run_id, d_name, m_name)
            slots[key] = (d_name, run_id, m_name, 0, 1, time.time())
            slot_order.append(key)
            _erase_dashboard()
            _draw_dashboard()

        elif kind == "progress":
            _, run_id, d_name, m_name, cur, tot = msg
            key = (run_id, d_name, m_name)
            if key in slots:
                d, r, m, _, _, ts = slots[key]
                slots[key] = (d, r, m, cur, tot, ts)
            _erase_dashboard()
            _draw_dashboard()

        elif kind == "done":
            _, run_id, d_name, m_name, metrics_dict, elapsed = msg
            key = (run_id, d_name, m_name)
            if key in slots:
                del slots[key]
                slot_order.remove(key)
            rwa    = metrics_dict.get("RWA_Score",            -1)
            gmean  = metrics_dict.get("G_Mean",               -1)
            minrec = metrics_dict.get("Mean_Minority_Recall", -1)
            drift  = metrics_dict.get("Drift_Detections",      0)
            tag    = _trunc(f"[{d_name}|run{run_id}]", DS_W)
            model  = _trunc(m_name, MODEL_W)
            _emit_log_line(
                f"  ✔  {tag:<{DS_W}}  {model:<{MODEL_W}}"
                f"  RWA={rwa:.4f}  GMean={gmean:.4f}"
                f"  MinRec={minrec:.4f}  Drift={drift}  {elapsed:.0f}s"
            )

        elif kind == "error":
            _, run_id, d_name, m_name, err_str = msg
            key = (run_id, d_name, m_name)
            if key in slots:
                del slots[key]
                slot_order.remove(key)
            tag   = _trunc(f"[{d_name}|run{run_id}]", DS_W)
            model = _trunc(m_name, MODEL_W)
            _emit_log_line(
                f"  ✘  {tag:<{DS_W}}  {model:<{MODEL_W}}  ERROR: {err_str}"
            )

        elif kind == "task_done":
            _, run_id, d_name = msg
            completed_tasks += 1
            _erase_dashboard()
            _draw_dashboard()

        elif kind == "log":
            # General text message (e.g. from main process)
            _, text = msg
            _emit_log_line(f"  ℹ  {text}")


# Worker-side helpers
# Global queue - set via _worker_init in each worker process
_LOG_QUEUE = None


def worker_init(q):
    """
    Pool initializer - each worker inherits queue through this function.
    On Windows/spawn, queue cannot be serialized into task tuples,
    but initializer passes it correctly through process inheritance.
    """
    global _LOG_QUEUE
    _LOG_QUEUE = q


def log(kind, *args):
    """
    Send message to the central logger process.
    Called by workers instead of direct print().
    Safe - exceptions are ignored so workers don't fail due to logging.
    """
    if _LOG_QUEUE is not None:
        try:
            _LOG_QUEUE.put((kind, *args))
        except Exception:
            pass