from __future__ import annotations

from pathlib import Path
import subprocess
import tkinter as tk


NAVY, NAVY_2 = "#071D28", "#0D2B38"
SURFACE, WHITE = "#F4F7F6", "#FFFFFF"
INK, MUTED = "#142B34", "#667C83"
TEAL, CYAN, CORAL = "#36D1B4", "#55C8E8", "#FF826F"
BORDER, PALE = "#DCE7E5", "#EBF3F1"


class ArgusLauncher:
    def __init__(self, root):
        self.root = root
        root.title("Argus — Surgical Safety Intelligence")
        root.geometry("1060x700")
        root.minsize(880, 620)
        root.configure(bg=SURFACE)
        self._sidebar()
        self._content()

    def _sidebar(self):
        bar = tk.Frame(self.root, width=270, bg=NAVY)
        bar.pack(side="left", fill="y")
        bar.pack_propagate(False)
        brand = tk.Canvas(bar, width=220, height=82, bg=NAVY, highlightthickness=0)
        brand.pack(anchor="w", padx=27, pady=(23, 2))
        brand.create_arc(8, 15, 55, 58, start=18, extent=144, style="arc",
                         outline=TEAL, width=4)
        brand.create_arc(8, 15, 55, 58, start=198, extent=144, style="arc",
                         outline=CYAN, width=4)
        brand.create_oval(26, 29, 38, 41, fill=TEAL, outline="")
        brand.create_text(70, 22, text="ARGUS", fill=WHITE, anchor="nw",
                          font=("DejaVu Sans", 17, "bold"))
        brand.create_text(70, 48, text="SURGICAL SAFETY AI", fill="#7898A1",
                          anchor="nw", font=("DejaVu Sans", 7, "bold"))

        tk.Label(bar, text="MONITORING CENTER", bg=NAVY, fg="#6F929B",
                 font=("DejaVu Sans", 8, "bold")).pack(anchor="w", padx=35,
                                                        pady=(22, 10))
        nav = (("Overview", True), ("Hand hygiene", False),
               ("Instruments", False), ("Sterile field", False),
               ("Sessions", False), ("Reports", False))
        for label, selected in nav:
            row = tk.Frame(bar, bg=NAVY_2 if selected else NAVY, height=43)
            row.pack(fill="x", padx=18, pady=1)
            row.pack_propagate(False)
            if selected:
                tk.Frame(row, bg=TEAL, width=3).pack(side="left", fill="y")
            tk.Label(row, text=label, bg=row["bg"],
                     fg=WHITE if selected else "#91AAB0",
                     font=("DejaVu Sans", 10, "bold" if selected else "normal")).pack(
                         side="left", padx=16)

        footer = tk.Frame(bar, bg=NAVY_2, height=78)
        footer.pack(side="bottom", fill="x", padx=18, pady=18)
        footer.pack_propagate(False)
        tk.Label(footer, text="●  SYSTEM READY", bg=NAVY_2, fg=TEAL,
                 font=("DejaVu Sans", 8, "bold")).pack(anchor="w", padx=15,
                                                        pady=(15, 4))
        tk.Label(footer, text="On-device processing  •  Beta", bg=NAVY_2,
                 fg="#7898A1", font=("DejaVu Sans", 7)).pack(anchor="w", padx=15)

    def _content(self):
        shell = tk.Frame(self.root, bg=SURFACE)
        shell.pack(side="left", fill="both", expand=True)
        header = tk.Frame(shell, bg=SURFACE, height=112)
        header.pack(fill="x", padx=38)
        header.pack_propagate(False)
        titles = tk.Frame(header, bg=SURFACE)
        titles.pack(side="left", pady=(25, 0))
        tk.Label(titles, text="Monitoring center", bg=SURFACE, fg=INK,
                 font=("DejaVu Sans", 25, "bold")).pack(anchor="w")
        tk.Label(titles, text="Select a module to begin a safety session.",
                 bg=SURFACE, fg=MUTED, font=("DejaVu Sans", 10)).pack(anchor="w", pady=4)
        badge = tk.Label(header, text="  LOCAL & PRIVATE  ", bg=PALE, fg="#368A7C",
                         font=("DejaVu Sans", 8, "bold"), pady=8)
        badge.pack(side="right", pady=(28, 0))

        body = tk.Frame(shell, bg=SURFACE)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg=SURFACE, highlightthickness=0)
        scroll = tk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y", padx=(0, 8), pady=(0, 12))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=SURFACE)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self.window, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", lambda _e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _e: self.canvas.yview_scroll(3, "units"))
        self._modules()

    def _wheel(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _modules(self):
        hand = self._module_header("HH", "Hand hygiene", "Live hand-rubbing and hygiene validation",
                                   "AVAILABLE", TEAL)
        modes = tk.Frame(hand, bg=WHITE)
        modes.pack(fill="x", padx=18, pady=(4, 18))
        choices = (
            ("Real sink test", "WHO-guided 40-60 second workflow", "--real", TEAL),
            ("Room hand test", "Live hands; soap and water bypassed", "--room", CYAN),
            ("Interactive simulator", "Explore the workflow without a camera", "--demo", CORAL),
        )
        for index, (title, detail, mode, accent) in enumerate(choices):
            card = tk.Frame(modes, bg="#F7FAF9", highlightbackground=BORDER,
                            highlightthickness=1, height=82)
            card.grid(row=0, column=index, sticky="nsew", padx=5)
            card.grid_propagate(False)
            modes.grid_columnconfigure(index, weight=1, uniform="mode")
            tk.Frame(card, bg=accent, height=3).pack(fill="x")
            tk.Label(card, text=title, bg="#F7FAF9", fg=INK,
                     font=("DejaVu Sans", 9, "bold")).pack(anchor="w", padx=11, pady=(9, 2))
            tk.Label(card, text=detail, bg="#F7FAF9", fg=MUTED, wraplength=165,
                     justify="left", font=("DejaVu Sans", 7)).pack(anchor="w", padx=11)
            button = tk.Button(card, text="OPEN  →", command=lambda m=mode: self.open(m),
                               bg="#F7FAF9", fg="#168A77", activebackground="#F7FAF9",
                               relief="flat", bd=0, font=("DejaVu Sans", 7, "bold"),
                               cursor="hand2")
            button.place(relx=1, rely=1, x=-8, y=-6, anchor="se")

        tk.Label(self.inner, text="ARGUS PLATFORM", bg=SURFACE, fg="#789095",
                 font=("DejaVu Sans", 8, "bold")).pack(anchor="w", padx=39, pady=(22, 8))
        modules = (
            ("IN", "Instrument intelligence", "Detect, count, and track instruments across procedure zones."),
            ("SF", "Sterile-field monitoring", "Flag possible boundary crossings and contamination events."),
            ("PS", "Procedure sessions", "Create procedures, rooms, timestamps, and monitoring timelines."),
            ("AL", "Live alerts", "Surface informational, warning, and critical events for review."),
            ("HR", "Human review", "Confirm, dismiss, annotate, and label safety alerts."),
            ("SR", "Safety reports", "Generate evidence-backed procedure summaries and count comparisons."),
        )
        for code, title, detail in modules:
            self._coming_soon(code, title, detail)
        tk.Label(self.inner, text="Argus supports human review and does not independently diagnose malpractice.",
                 bg=SURFACE, fg="#8A9B9E", font=("DejaVu Sans", 8)).pack(
                     anchor="w", padx=39, pady=(10, 30))

    def _module_header(self, code, title, detail, status, accent):
        card = tk.Frame(self.inner, bg=WHITE, highlightbackground=BORDER,
                        highlightthickness=1)
        card.pack(fill="x", padx=38, pady=(2, 0))
        top = tk.Frame(card, bg=WHITE, height=82)
        top.pack(fill="x")
        top.pack_propagate(False)
        icon = tk.Label(top, text=code, bg=accent, fg=NAVY, width=4, height=2,
                        font=("DejaVu Sans", 10, "bold"))
        icon.pack(side="left", padx=18, pady=16)
        text = tk.Frame(top, bg=WHITE)
        text.pack(side="left", pady=18)
        tk.Label(text, text=title, bg=WHITE, fg=INK,
                 font=("DejaVu Sans", 14, "bold")).pack(anchor="w")
        tk.Label(text, text=detail, bg=WHITE, fg=MUTED,
                 font=("DejaVu Sans", 9)).pack(anchor="w", pady=3)
        tk.Label(top, text=f"  {status}  ", bg="#E7F7F2", fg="#198572",
                 font=("DejaVu Sans", 7, "bold"), pady=5).pack(side="right", padx=18)
        return card

    def _coming_soon(self, code, title, detail):
        card = tk.Frame(self.inner, bg=WHITE, highlightbackground=BORDER,
                        highlightthickness=1, height=86)
        card.pack(fill="x", padx=38, pady=5)
        card.pack_propagate(False)
        tk.Label(card, text=code, bg=PALE, fg="#547078", width=4, height=2,
                 font=("DejaVu Sans", 9, "bold")).pack(side="left", padx=17)
        text = tk.Frame(card, bg=WHITE)
        text.pack(side="left", pady=17)
        tk.Label(text, text=title, bg=WHITE, fg=INK,
                 font=("DejaVu Sans", 11, "bold")).pack(anchor="w")
        tk.Label(text, text=detail, bg=WHITE, fg=MUTED,
                 font=("DejaVu Sans", 8)).pack(anchor="w", pady=4)
        tk.Label(card, text="COMING SOON", bg="#F1F4F3", fg="#7B8C90",
                 font=("DejaVu Sans", 7, "bold"), padx=11, pady=6).pack(
                     side="right", padx=18)

    def open(self, mode):
        launcher = Path(__file__).resolve().parents[1] / "launch.sh"
        subprocess.Popen((str(launcher), mode))


def main():
    root = tk.Tk()
    ArgusLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
