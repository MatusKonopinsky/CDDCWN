"""
utils/logger.py — Centralizovaný live dashboard logger pre paralelné experimenty.

Architektúra:
  - Každý worker posiela správy do jednej multiprocessing.Queue
  - Jediný _logger_process ich prijíma a vypisuje sekvenčne
  - Dashboard sa aktualizuje cez ANSI escape sekvencie

Typy správ (tuple):
  ("start",     run_id, d_name, m_name)               — worker začal model
  ("progress",  run_id, d_name, m_name, cur, total)   — priebeh (počet vzoriek)
  ("done",      run_id, d_name, m_name, metrics, time) — model dokončený
  ("error",     run_id, d_name, m_name, err_str)       — chyba workera
  ("task_done", run_id, d_name)                        — celý dataset/run hotový
  ("STOP",)                                            — ukončenie logger procesu

Výzor výstupu:
  ✔  [SEA_Imb9010|run1]        DDCW_aug+drift          RWA=0.9955  F1=0.9389  MinF1=0.8915  Drift=0  312s
  ✔  [Agrawal_Imb9010|run1]    ARF                     RWA=0.1059  F1=0.4803  MinF1=0.0132  Drift=0  208s
  ────────────────────────────────────────────────────────────────────────────
  [rbf_drift_imb9010|run1]     LeveragingBaggingClas…  ████████░░  42%  21000/49500
  [MC_Abrupt_3C|run1]          DDCW_aug+drift          ███░░░░░░░  18%   9000/49500
  Hotovo: 2/8  ████████░░░░░░░░  25%      2/8
"""

import sys
import time

# ── Konštanty ──────────────────────────────────────────────────────────────
BAR_W   = 22   # šírka progress baru
DS_W    = 24   # šírka stĺpca [dataset|run]
MODEL_W = 26   # šírka stĺpca meno modelu
TOTAL_W = 80   # šírka oddeľovacieho riadku

CLEAR_LINE = "\033[2K\r"
CURSOR_UP  = "\033[{}A"


# ── Helpers ────────────────────────────────────────────────────────────────

def _bar(cur, tot, width=BAR_W):
    p   = min(1.0, cur / max(1, tot))
    f   = int(width * p)
    pct = int(p * 100)
    return f"{'█'*f}{'░'*(width-f)} {pct:3d}%  {cur:>6}/{tot}"


def _trunc(s, n):
    """Skráti reťazec na n znakov s '…' na konci."""
    return s if len(s) <= n else s[:n - 1] + "…"


# ── Hlavný logger proces ───────────────────────────────────────────────────

def _logger_process(queue, total_tasks):
    """
    Beží v samostatnom procese. Prijíma správy z queue a udržiava
    live dashboard v terminály pomocou ANSI escape sekvencií.

    Opravený bug oproti pôvodnej verzii:
      - _erase_dashboard() teraz správne neprepíše riadky keď sú sloty prázdne
        (pôvodne pri height=2 kreslil separator+summary aj pri prázdnych slotoch,
        čo spôsobovalo výpis samotných '───' riadkov bez obsahu)
      - Dashboard sa nevykreslí znovu ak sa nič nezmenilo (progress throttling)
    """
    completed_tasks = 0
    slots      = {}   # key -> (d_name, run_id, m_name, cur, tot, start_ts)
    slot_order = []   # zachovanie poradia pridania

    # ── Rendering ──────────────────────────────────────────────────────────

    def _slot_line(key):
        d_name, run_id, m_name, cur, tot, _ = slots[key]
        tag   = _trunc(f"[{d_name}|run{run_id}]", DS_W)
        model = _trunc(m_name, MODEL_W)
        return f"  {tag:<{DS_W}}  {model:<{MODEL_W}}  {_bar(cur, tot)}"

    def _overall_line():
        return f"  Hotovo: {completed_tasks}/{total_tasks}  {_bar(completed_tasks, total_tasks)}"

    def _dash_height():
        # separator + jeden riadok per slot + summary riadok
        # Ak nie sú žiadne sloty, vykreslíme len separator + summary (výška=2)
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
        """Vypíše trvalý log riadok NAD dashboard."""
        _erase_dashboard()
        sys.stdout.write(f"{text}\n")
        sys.stdout.flush()
        _draw_dashboard()

    # Prvotné vykreslenie
    _draw_dashboard()

    # ── Hlavná slučka ──────────────────────────────────────────────────────
    while True:
        msg  = queue.get()
        kind = msg[0]

        if kind == "STOP":
            _erase_dashboard()
            sys.stdout.write(f"{'─' * TOTAL_W}\n")
            sys.stdout.write(
                f"  Všetky úlohy dokončené.  Hotovo: {completed_tasks}/{total_tasks}\n"
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
                f"  ✘  {tag:<{DS_W}}  {model:<{MODEL_W}}  CHYBA: {err_str}"
            )

        elif kind == "task_done":
            _, run_id, d_name = msg
            completed_tasks += 1
            _erase_dashboard()
            _draw_dashboard()

        elif kind == "log":
            # Všeobecná textová správa (napr. z hlavného procesu)
            _, text = msg
            _emit_log_line(f"  ℹ  {text}")


# ── Worker-side helpers ────────────────────────────────────────────────────

# Globálna queue — nastavená cez _worker_init v každom worker procese
_LOG_QUEUE = None


def worker_init(q):
    """
    Pool initializer — každý worker zdedí queue cez túto funkciu.
    Na Windows/spawn sa queue nemôže serializovať do task-tuplov,
    ale cez initializer sa predáva správne cez dedičnosť procesu.
    """
    global _LOG_QUEUE
    _LOG_QUEUE = q


def log(kind, *args):
    """
    Odošle správu do centrálneho logger procesu.
    Volajú workery namiesto priameho print().
    Bezpečné — výnimky sa ignorujú, aby nespadol worker kvôli logu.
    """
    if _LOG_QUEUE is not None:
        try:
            _LOG_QUEUE.put((kind, *args))
        except Exception:
            pass