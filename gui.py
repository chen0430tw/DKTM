#!/usr/bin/env python3
"""
DKTM GUI — Hot Restart Control Panel
=====================================
One-click kernel reset via WinPE.  Requires Administrator privileges.

Usage:
    python gui.py
    python gui.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext

# Add DKTM package to path
sys.path.insert(0, str(Path(__file__).parent))

from dktm import config as dktm_config
from dktm import platform_ops

# ── Colours ───────────────────────────────────────────────────────────────────
C_BG       = "#1e1e2e"
C_PANEL    = "#2a2a3e"
C_BORDER   = "#3a3a5c"
C_TEXT     = "#cdd6f4"
C_DIM      = "#6c7086"
C_GREEN    = "#a6e3a1"
C_RED      = "#f38ba8"
C_YELLOW   = "#f9e2af"
C_BLUE     = "#89b4fa"
C_BUTTON   = "#89b4fa"
C_BTN_TEXT = "#1e1e2e"
C_LOG_BG   = "#181825"

COUNTDOWN_SECS = 5


class QueueHandler(logging.Handler):
    """Push log records into a queue for the GUI thread to consume."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(("log", self.format(record), record.levelno))


class DKTMApp:
    """Main GUI application."""

    # States
    IDLE        = "idle"
    CHECKING    = "checking"
    READY       = "ready"
    PREPARING   = "preparing"
    COUNTDOWN   = "countdown"
    COMMITTING  = "committing"
    REBOOTING   = "rebooting"
    ABORTED     = "aborted"
    ERROR       = "error"

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.state   = self.IDLE
        self._cancel_event = threading.Event()
        self._msg_queue: queue.Queue = queue.Queue()

        # Load config
        config_path = str(Path(__file__).parent / "config.yaml")
        cfg = dktm_config.load_config(config_path)
        self._exec_cfg = cfg.get("executor", {})
        if dry_run:
            self._exec_cfg["mode"] = "dry-run"

        # Logging
        self._log_queue: queue.Queue = queue.Queue()
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        qh = QueueHandler(self._log_queue)
        qh.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(qh)
        self.logger = logging.getLogger("dktm.gui")

        # Build UI
        self.root = tk.Tk()
        self._build_ui()
        self.root.after(100, self._drain_queue)
        self.root.after(300, self._startup_check)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        r = self.root
        r.title("DKTM Hot Restart")
        r.configure(bg=C_BG)
        r.resizable(False, False)
        r.protocol("WM_DELETE_WINDOW", self._on_close)

        bold14 = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        bold20 = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        reg10  = tkfont.Font(family="Segoe UI", size=10)
        mono9  = tkfont.Font(family="Consolas",  size=9)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(r, bg=C_BG, pady=12)
        hdr.pack(fill=tk.X, padx=20)
        tk.Label(hdr, text="DKTM", font=bold20, bg=C_BG, fg=C_BLUE).pack(side=tk.LEFT)
        tk.Label(hdr, text=" Hot Restart", font=bold20, bg=C_BG, fg=C_TEXT).pack(side=tk.LEFT)
        if self.dry_run:
            tk.Label(hdr, text=" [DRY-RUN]", font=reg10, bg=C_BG,
                     fg=C_YELLOW).pack(side=tk.LEFT, padx=(8, 0))

        # ── Status indicators ─────────────────────────────────────────────────
        status_frame = tk.Frame(r, bg=C_PANEL, bd=0, relief=tk.FLAT,
                                highlightbackground=C_BORDER, highlightthickness=1)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        inner = tk.Frame(status_frame, bg=C_PANEL, pady=8, padx=14)
        inner.pack(fill=tk.X)

        self._indicators: dict[str, tk.Label] = {}
        items = [
            ("admin",  "Administrator"),
            ("winpe",  "WinPE"),
            ("disk",   "Disk"),
        ]
        for key, label in items:
            col = tk.Frame(inner, bg=C_PANEL)
            col.pack(side=tk.LEFT, padx=(0, 24))
            dot = tk.Label(col, text="●", font=reg10, bg=C_PANEL, fg=C_DIM)
            dot.pack(side=tk.LEFT, padx=(0, 4))
            tk.Label(col, text=label, font=reg10, bg=C_PANEL, fg=C_DIM).pack(side=tk.LEFT)
            self._indicators[key] = dot

        # ── Main button area ──────────────────────────────────────────────────
        btn_frame = tk.Frame(r, bg=C_BG, pady=10)
        btn_frame.pack(fill=tk.X, padx=20)

        self._main_btn = tk.Button(
            btn_frame,
            text="HOT RESTART",
            font=bold14,
            bg=C_BUTTON, fg=C_BTN_TEXT,
            activebackground=C_BLUE, activeforeground=C_BTN_TEXT,
            relief=tk.FLAT, bd=0, padx=32, pady=12,
            cursor="hand2",
            command=self._on_restart_click,
        )
        self._main_btn.pack(fill=tk.X)

        # Countdown bar (hidden until needed)
        self._countdown_frame = tk.Frame(r, bg=C_BG)
        self._countdown_frame.pack(fill=tk.X, padx=20, pady=(0, 4))

        self._countdown_label = tk.Label(
            self._countdown_frame, text="", font=bold14,
            bg=C_BG, fg=C_YELLOW,
        )
        self._countdown_label.pack(side=tk.LEFT)

        self._cancel_btn = tk.Button(
            self._countdown_frame,
            text="Cancel",
            font=reg10,
            bg=C_RED, fg=C_BTN_TEXT,
            activebackground="#d20f39", activeforeground=C_BTN_TEXT,
            relief=tk.FLAT, bd=0, padx=16, pady=6,
            cursor="hand2",
            command=self._on_cancel_click,
        )
        self._cancel_btn.pack(side=tk.RIGHT)
        self._countdown_frame.pack_forget()

        # ── Status line ───────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Checking system…")
        tk.Label(r, textvariable=self._status_var, font=reg10,
                 bg=C_BG, fg=C_DIM, anchor=tk.W).pack(
            fill=tk.X, padx=20, pady=(0, 4)
        )

        # ── Log area ──────────────────────────────────────────────────────────
        log_outer = tk.Frame(r, bg=C_BORDER, padx=1, pady=1)
        log_outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 16))
        self._log = scrolledtext.ScrolledText(
            log_outer,
            font=mono9, bg=C_LOG_BG, fg=C_TEXT,
            insertbackground=C_TEXT,
            relief=tk.FLAT, bd=0,
            wrap=tk.WORD, state=tk.DISABLED,
            width=64, height=14,
        )
        self._log.pack(fill=tk.BOTH, expand=True)
        self._log.tag_config("ERROR",   foreground=C_RED)
        self._log.tag_config("WARNING", foreground=C_YELLOW)
        self._log.tag_config("INFO",    foreground=C_TEXT)
        self._log.tag_config("DEBUG",   foreground=C_DIM)

        r.update_idletasks()
        w, h = r.winfo_reqwidth(), r.winfo_reqheight()
        x = (r.winfo_screenwidth()  - w) // 2
        y = (r.winfo_screenheight() - h) // 2
        r.geometry(f"+{x}+{y}")
        self._set_btn_state(enabled=False)

    # ── Logging / queue ───────────────────────────────────────────────────────

    def _append_log(self, text: str, level: int = logging.INFO) -> None:
        tag = {
            logging.DEBUG:   "DEBUG",
            logging.INFO:    "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR:   "ERROR",
            logging.CRITICAL:"ERROR",
        }.get(level, "INFO")

        self._log.configure(state=tk.NORMAL)
        self._log.insert(tk.END, text.rstrip() + "\n", tag)
        self._log.see(tk.END)
        self._log.configure(state=tk.DISABLED)

    def _drain_queue(self) -> None:
        """Pull messages from the queue and update the UI. Called via after()."""
        try:
            while True:
                kind, *args = self._log_queue.get_nowait()
                if kind == "log":
                    text, level = args
                    self._append_log(text, level)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_queue)

    # ── Indicator helpers ─────────────────────────────────────────────────────

    def _set_indicator(self, key: str, ok: Optional[bool]) -> None:
        colours = {True: C_GREEN, False: C_RED, None: C_DIM}
        self._indicators[key].configure(fg=colours.get(ok, C_DIM))

    def _set_status(self, text: str, colour: str = C_DIM) -> None:
        self._status_var.set(text)

    def _set_btn_state(self, enabled: bool, text: str = "HOT RESTART",
                       colour: str = C_BUTTON) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self._main_btn.configure(state=state, text=text, bg=colour,
                                 fg=C_BTN_TEXT if enabled else C_DIM)

    # ── Startup check ─────────────────────────────────────────────────────────

    def _startup_check(self) -> None:
        self.state = self.CHECKING
        self._set_btn_state(False, "Checking…")
        threading.Thread(target=self._do_startup_check, daemon=True).start()

    def _do_startup_check(self) -> None:
        cfg = self._exec_cfg
        issues = []

        # Admin
        try:
            import ctypes
            is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            is_admin = True  # non-Windows / dry-run
        self.root.after(0, self._set_indicator, "admin", is_admin)
        if not is_admin:
            issues.append("Not running as Administrator")

        # WinPE entries
        if sys.platform == "win32":
            from dktm.platform_windows import PlatformOps as WinOps
            win_ops = WinOps(
                winpe_entry_ids=cfg.get("winpe_entry_ids", []),
                marker_path=cfg.get("marker_path", "dktm_transition.marker"),
                dry_run=self.dry_run,
            )
            entries = win_ops._resolve_winpe_entry_ids()
            winpe_ok = len(entries) > 0
            self.root.after(0, self._set_indicator, "winpe", winpe_ok)
            if not winpe_ok:
                issues.append("No WinPE/WinRE entry found in BCD")
            else:
                self.logger.info("WinPE entry: %s", entries[0])
        else:
            self.root.after(0, self._set_indicator, "winpe", None)

        # Disk space
        try:
            import ctypes as ct
            free = ct.c_ulonglong(0)
            sys_drive = os.environ.get("SystemDrive", "C:") + "\\"
            ct.windll.kernel32.GetDiskFreeSpaceExW(
                sys_drive, None, None, ct.byref(free)
            )
            free_mb = free.value // (1024 * 1024)
            disk_ok = free_mb >= 100
            self.root.after(0, self._set_indicator, "disk", disk_ok)
            if not disk_ok:
                issues.append(f"Low disk space: {free_mb} MB")
            else:
                self.logger.info("Disk: %d MB free on %s", free_mb, sys_drive)
        except Exception:
            self.root.after(0, self._set_indicator, "disk", None)

        if issues:
            self.root.after(0, self._set_status, "  ".join(issues), C_RED)
            self.root.after(0, self._set_btn_state, False, "HOT RESTART")
            self.state = self.ERROR
        else:
            self.root.after(0, self._set_status, "Ready")
            self.root.after(0, self._set_btn_state, True, "HOT RESTART", C_BUTTON)
            self.state = self.READY

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_restart_click(self) -> None:
        if self.state != self.READY:
            return
        self._cancel_event.clear()
        self.state = self.PREPARING
        self._set_btn_state(False, "Preparing…")
        self._set_status("Preparing system…")
        threading.Thread(target=self._do_prepare, daemon=True).start()

    def _on_cancel_click(self) -> None:
        if self.state == self.COUNTDOWN:
            self._cancel_event.set()

    def _on_close(self) -> None:
        if self.state in (self.COMMITTING, self.REBOOTING):
            return  # don't close during critical phase
        self._cancel_event.set()
        self.root.destroy()

    # ── Worker threads ────────────────────────────────────────────────────────

    def _do_prepare(self) -> None:
        """health_check → freeze → flush (runs in background thread)."""
        try:
            self.logger.info("── Health check ──")
            platform_ops.health_check(dry_run=self.dry_run)
            self.logger.info("── Quiescing services ──")
            platform_ops.freeze_services(dry_run=self.dry_run)
            self.logger.info("── Flushing I/O ──")
            platform_ops.flush_buffers(dry_run=self.dry_run)
        except Exception as exc:
            self.logger.error("Preparation failed: %s", exc)
            self.root.after(0, self._on_prepare_failed, str(exc))
            return
        self.root.after(0, self._start_countdown)

    def _on_prepare_failed(self, msg: str) -> None:
        self.state = self.ERROR
        self._set_btn_state(True, "HOT RESTART")
        self._set_status(f"Failed: {msg}")

    def _start_countdown(self) -> None:
        self.state = self.COUNTDOWN
        self._countdown_frame.pack(fill=tk.X, padx=20, pady=(0, 4))
        self._do_countdown(COUNTDOWN_SECS)

    def _do_countdown(self, remaining: int) -> None:
        if self._cancel_event.is_set() or self.state != self.COUNTDOWN:
            self._abort_countdown()
            return
        if remaining > 0:
            self._countdown_label.configure(
                text=f"Rebooting in {remaining}s…  (press Cancel to abort)"
            )
            self.root.after(1000, self._do_countdown, remaining - 1)
        else:
            self._countdown_label.configure(text="Rebooting…")
            self._cancel_btn.configure(state=tk.DISABLED)
            self.state = self.COMMITTING
            threading.Thread(target=self._do_commit, daemon=True).start()

    def _abort_countdown(self) -> None:
        self.state = self.READY
        self._countdown_frame.pack_forget()
        self._countdown_label.configure(text="")
        self._set_btn_state(True, "HOT RESTART", C_BUTTON)
        self._set_status("Aborted — system unchanged")
        self.logger.info("Hot restart aborted by user")
        self._cancel_event.clear()

    def _do_commit(self) -> None:
        """commit_transition → flush → reboot (runs in background thread)."""
        cfg = self._exec_cfg
        try:
            self.logger.info("── Writing BCD bootsequence ──")
            platform_ops.commit_transition(
                winpe_entry_ids=cfg.get("winpe_entry_ids", []),
                marker_path=cfg.get("marker_path", "dktm_transition.marker"),
                auto_reboot=False,
                dry_run=self.dry_run,
                transition_method=cfg.get("transition_method", "auto"),
                fallback_method=cfg.get("fallback_method", "winre"),
            )
            self.logger.info("── Final flush ──")
            platform_ops.flush_buffers(dry_run=self.dry_run)
            self.logger.info("── Rebooting ──")
            self.state = self.REBOOTING
            platform_ops.reboot(dry_run=self.dry_run)
        except Exception as exc:
            self.logger.error("Commit failed: %s", exc)
            self.root.after(0, self._on_commit_failed, str(exc))

    def _on_commit_failed(self, msg: str) -> None:
        self.state = self.ERROR
        self._countdown_frame.pack_forget()
        self._set_btn_state(True, "HOT RESTART")
        self._set_status(f"Failed: {msg}")
        # Attempt rollback
        cfg = self._exec_cfg
        try:
            self.logger.warning("Attempting rollback…")
            platform_ops.rollback_transition(
                winpe_entry_ids=cfg.get("winpe_entry_ids", []),
                marker_path=cfg.get("marker_path", "dktm_transition.marker"),
                dry_run=self.dry_run,
            )
            self.logger.info("Rollback complete")
        except Exception as exc2:
            self.logger.error("Rollback also failed: %s", exc2)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="DKTM Hot Restart GUI")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without making system changes")
    args = parser.parse_args()

    if sys.platform != "win32" and not args.dry_run:
        print("ERROR: requires Windows. Use --dry-run to test on other platforms.")
        sys.exit(1)

    app = DKTMApp(dry_run=args.dry_run)
    app.run()


if __name__ == "__main__":
    main()
