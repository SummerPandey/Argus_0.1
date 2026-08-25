from __future__ import annotations

import math
import time
import tkinter as tk

from .session import Observation, SessionEngine, Stage


BG, CARD, INK = "#e9fbf7", "#ffffff", "#173b3f"
MINT, AQUA, CORAL, MUTED = "#42c7a5", "#56c9e9", "#ff806d", "#668184"


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Argus — Hand Hygiene Simulator")
        self.root.geometry("820x620")
        self.root.minsize(720, 560)
        self.root.configure(bg=BG)
        self.engine = SessionEngine(required_rub_seconds=20.0)
        self.hands = False
        self.rubbing = False
        self.soap_pulse = 0.0
        self.started = time.monotonic()
        self.last = self.started
        self._build()
        self._tick()

    def _build(self):
        tk.Label(self.root, text="ARGUS", bg=BG, fg=MINT,
                 font=("DejaVu Sans", 12, "bold")).pack(pady=(22, 0))
        tk.Label(self.root, text="Handwash Monitor", bg=BG, fg=INK,
                 font=("DejaVu Sans", 28, "bold")).pack()
        tk.Label(self.root, text="A smooth temporal-state demo — no camera required",
                 bg=BG, fg=MUTED, font=("DejaVu Sans", 11)).pack(pady=(2, 14))

        self.canvas = tk.Canvas(self.root, width=680, height=350, bg=CARD,
                                highlightbackground="#cce8e1", highlightthickness=1)
        self.canvas.pack(padx=30, fill="x")

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(pady=18)
        self.hand_button = self._button(controls, "Show hands", self.toggle_hands, AQUA)
        self.hand_button.grid(row=0, column=0, padx=6)
        self._button(controls, "Apply soap", self.apply_soap, MINT).grid(row=0, column=1, padx=6)
        self.rub_button = self._button(controls, "Start rubbing", self.toggle_rubbing, CORAL)
        self.rub_button.grid(row=0, column=2, padx=6)
        self._button(controls, "Reset", self.reset, "#dcebea").grid(row=0, column=3, padx=6)
        tk.Label(self.root, text="Tip: Start rubbing stays active until you click it again.",
                 bg=BG, fg=MUTED, font=("DejaVu Sans", 9)).pack()

    def _button(self, parent, text, command, color):
        return tk.Button(parent, text=text, command=command, bg=color, fg=INK,
                         activebackground=color, relief="flat", padx=17, pady=10,
                         font=("DejaVu Sans", 10, "bold"), cursor="hand2")

    def toggle_hands(self):
        self.hands = not self.hands
        if not self.hands:
            self.rubbing = False
        self.hand_button.configure(text="Hide hands" if self.hands else "Show hands")
        self.rub_button.configure(text="Stop rubbing" if self.rubbing else "Start rubbing")

    def toggle_rubbing(self):
        if not self.hands:
            self.hands = True
            self.hand_button.configure(text="Hide hands")
        self.rubbing = not self.rubbing
        self.rub_button.configure(text="Stop rubbing" if self.rubbing else "Start rubbing")

    def apply_soap(self):
        self.soap_pulse = time.monotonic() + 0.8

    def reset(self):
        self.engine.reset()
        self.hands = self.rubbing = False
        self.soap_pulse = 0.0
        self.hand_button.configure(text="Show hands")
        self.rub_button.configure(text="Start rubbing")

    def _tick(self):
        now = time.monotonic()
        snapshot = self.engine.update(Observation(
            timestamp=now,
            hands_confidence=0.96 if self.hands else 0.0,
            rubbing_confidence=0.94 if self.rubbing and self.hands else 0.0,
            soap_confidence=0.95 if now < self.soap_pulse else 0.0,
        ))
        self._draw(snapshot, now)
        self.root.after(50, self._tick)

    def _draw(self, state, now):
        c = self.canvas
        c.delete("all")
        width = max(680, c.winfo_width())
        c.create_text(width / 2, 42, text=state.message, fill=INK,
                      font=("DejaVu Sans", 20, "bold"))
        c.create_text(width / 2, 72, text=state.stage.value, fill=MUTED,
                      font=("DejaVu Sans", 10))

        # Friendly animated bubbles and simple hands.
        for i in range(5):
            phase = now * (0.7 + i * .08) + i
            x = width / 2 - 110 + i * 55 + math.sin(phase) * 8
            y = 155 - (phase * 18) % 95
            r = 7 + (i % 3) * 3
            c.create_oval(x-r, y-r, x+r, y+r, fill="#c9f5ee", outline=AQUA, width=2)
        if self.hands:
            shift = math.sin(now * 9) * 10 if self.rubbing else 0
            c.create_oval(width/2-125+shift, 125, width/2+10+shift, 235,
                          fill="#ffd7a8", outline="#edae72", width=3)
            c.create_oval(width/2-10-shift, 125, width/2+125-shift, 235,
                          fill="#ffd7a8", outline="#edae72", width=3)

        x1, x2, y = 90, width - 90, 278
        c.create_rectangle(x1, y, x2, y+24, fill="#dcefed", outline="")
        c.create_rectangle(x1, y, x1+(x2-x1)*state.progress, y+24,
                           fill=MINT if state.stage != Stage.FAILED else CORAL, outline="")
        c.create_text(width/2, y+12,
                      text=f"{state.rubbing_seconds:0.1f} / 20 seconds",
                      fill=INK, font=("DejaVu Sans", 10, "bold"))
        checks = f"Hands  {'✓' if state.hands_present else '○'}     Soap  {'✓' if state.soap_seen else '○'}     Rubbing  {'✓' if state.rubbing_seconds > 0 else '○'}"
        c.create_text(width/2, 328, text=checks, fill=MUTED,
                      font=("DejaVu Sans", 11, "bold"))


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
