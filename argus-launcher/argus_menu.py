#!/usr/bin/env python3
import os
import subprocess
import tkinter as tk
from tkinter import messagebox


INSTALL_DIR = os.path.dirname(os.path.abspath(__file__))

BG = "#FFF6D9"
SURFACE = "#FFFDF4"
CARD = "#FFFFFF"
TEXT = "#342A1F"
MUTED = "#786B5B"
ACCENT = "#E8A82B"
READY = "#2E8B57"
BORDER = "#EADDBB"

MODULES = [
    ("✦", "Hand Wash", "Soap and rubbing compliance", True),
    ("⚙", "Equipment", "General tray + stain check", True),
    ("✚", "Surgery", "Procedure monitoring", False),
    ("◈", "Gloves", "Glove compliance", False),
]


class ArgusMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Argus Monitor")
        self.root.geometry("920x530")
        self.root.minsize(650, 470)
        self.root.configure(bg=BG)
        self.images = {}

        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=36, pady=(28, 12))

        tk.Label(
            header, text="ARGUS", bg=BG, fg=ACCENT,
            font=("DejaVu Sans", 13, "bold")
        ).pack(anchor="w")
        tk.Label(
            header, text="Monitoring Center", bg=BG, fg=TEXT,
            font=("DejaVu Sans", 28, "bold")
        ).pack(anchor="w", pady=(3, 2))
        tk.Label(
            header,
            text="Choose a module. Scroll sideways to view every option.",
            bg=BG, fg=MUTED, font=("DejaVu Sans", 11)
        ).pack(anchor="w")

        body = tk.Frame(root, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        body.pack(fill="both", expand=True, padx=28, pady=(4, 24))

        self.canvas = tk.Canvas(
            body, bg=SURFACE, highlightthickness=0, height=315
        )
        scrollbar = tk.Scrollbar(
            body, orient="horizontal", command=self.canvas.xview,
            troughcolor=BG, bg=ACCENT
        )
        self.canvas.configure(xscrollcommand=scrollbar.set)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(12, 0))
        scrollbar.pack(fill="x", padx=18, pady=(0, 14))

        self.cards = tk.Frame(self.canvas, bg=SURFACE)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.cards, anchor="nw"
        )
        self.cards.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._fit_height)
        self.canvas.bind_all("<Shift-MouseWheel>", self._mouse_scroll)

        for module in MODULES:
            self._add_card(*module)

    def _add_card(self, symbol, title, description, enabled):
        card = tk.Frame(
            self.cards, width=245, height=275, bg=CARD,
            highlightbackground=BORDER, highlightthickness=1,
            cursor="hand2" if enabled else "arrow"
        )
        card.pack(side="left", padx=12, pady=20)
        card.pack_propagate(False)

        tk.Label(
            card, text=symbol, bg="#FFF1BF", fg=ACCENT,
            font=("DejaVu Sans", 34, "bold"), width=3, height=1
        ).pack(pady=(22, 10))
        tk.Label(
            card, text=title, bg=CARD, fg=TEXT,
            font=("DejaVu Sans", 16, "bold")
        ).pack()
        tk.Label(
            card, text=description, bg=CARD, fg=MUTED,
            font=("DejaVu Sans", 10), wraplength=205
        ).pack(padx=16, pady=(7, 12))

        if enabled:
            button = tk.Button(
                card, text="OPEN", command=lambda name=title: self.launch(name),
                bg=ACCENT, fg="#201708", activebackground="#F2BE51",
                activeforeground="#201708", relief="flat",
                font=("DejaVu Sans", 10, "bold"), padx=28, pady=8,
                cursor="hand2"
            )
        else:
            button = tk.Button(
                card, text="COMING SOON",
                command=lambda name=title: self.coming_soon(name),
                bg="#F2EBD8", fg=MUTED, activebackground="#F2EBD8",
                relief="flat", font=("DejaVu Sans", 9, "bold"),
                padx=18, pady=8
            )
        button.pack(side="bottom", pady=20)

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_height(self, event):
        self.canvas.itemconfigure(self.window_id, height=max(event.height, 315))

    def _mouse_scroll(self, event):
        self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

    def coming_soon(self, name):
        messagebox.showinfo(name, f"{name} is coming soon.")

    def launch(self, name):
        launchers = {
            "Hand Wash": ("run_handwash.sh", "Argus Hand Wash"),
            "Equipment": ("run_equipment.sh", "Argus Equipment Scan"),
        }
        script, terminal_title = launchers[name]
        launcher = os.path.join(INSTALL_DIR, script)
        try:
            subprocess.Popen([
                "gnome-terminal", f"--title={terminal_title}", "--",
                "bash", "-lc", launcher
            ])
        except OSError as exc:
            messagebox.showerror("Could not launch", str(exc))


if __name__ == "__main__":
    app_root = tk.Tk()
    ArgusMenu(app_root)
    app_root.mainloop()
