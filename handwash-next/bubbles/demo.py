from __future__ import annotations

import math
import time
import tkinter as tk

from .session import Observation, SessionEngine, Stage


BG, CARD, INK = "#e9fbf7", "#ffffff", "#173b3f"
MINT, AQUA, CORAL, MUTED = "#42c7a5", "#56c9e9", "#ff806d", "#668184"
CARD_BORDER, CHIP_OFF = "#d3ece6", "#c9dbd8"

RESULT_ICON_RADIUS = 46

STAGE_TINTS = {
    Stage.IDLE: MUTED,
    Stage.READY: AQUA,
    Stage.SOAP: MINT,
    Stage.RUBBING: MINT,
    Stage.COMPLETE: MINT,
    Stage.FAILED: CORAL,
}


def _shade(hex_color, factor):
    """Darken (factor < 1) or lighten (factor > 1) a #rrggbb color - used
    for button hover states so pressing a control gives visible feedback
    instead of the flat, static look of a stock tk.Button."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, min(255, int(c * factor))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _rounded_points(x1, y1, x2, y2, radius):
    return [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]


def _rounded_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    return canvas.create_polygon(_rounded_points(x1, y1, x2, y2, radius), smooth=True, **kwargs)


def _pill(canvas, cx, cy, text, fill, text_color, font, padx=14, pady=7, tags=()):
    """Centers a rounded pill badge on (cx, cy), sized from the text's
    actual rendered extents rather than a guessed character width - keeps
    the pill correctly fitted regardless of font substitution or the
    particular letters in play."""
    text_id = canvas.create_text(cx, cy, text=text, fill=text_color, font=font, tags=tags)
    x1, y1, x2, y2 = canvas.bbox(text_id)
    half_w = (x2 - x1) / 2 + padx
    half_h = (y2 - y1) / 2 + pady
    _rounded_rect(canvas, cx - half_w, cy - half_h, cx + half_w, cy + half_h,
                  half_h, fill=fill, outline="", tags=tags)
    canvas.tag_raise(text_id)
    return half_w, half_h


class DemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Argus — Hand Hygiene Simulator")
        root.geometry("980x760")
        root.minsize(760, 620)
        # Root stays black so that in fullscreen (where the fixed-size
        # body below is smaller than the screen) the surrounding margin
        # reads as a letterbox bar instead of a near-white gap.
        root.configure(bg="black")
        self.body = tk.Frame(root, bg=BG)
        self.body.pack(expand=True)
        self.engine = SessionEngine(required_rub_seconds=20.0)
        self.hands = False
        self.rubbing = False
        self.soap_pulse = 0.0
        # (width, height) the card+shadow were last painted at - lets
        # _draw_card skip repainting on every 50ms tick when the window
        # hasn't actually resized, since that background never changes
        # otherwise.
        self._card_size = None
        self._build()
        self._tick()

    def _build(self):
        header = tk.Frame(self.body, bg=BG)
        header.pack(fill="x", padx=42, pady=(30, 4))

        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        logo = tk.Canvas(brand, width=34, height=34, bg=BG, highlightthickness=0)
        logo.pack(side="left")
        logo.create_arc(2, 2, 32, 32, start=18, extent=144, style="arc", outline=MINT, width=3)
        logo.create_arc(2, 2, 32, 32, start=198, extent=144, style="arc", outline=AQUA, width=3)
        logo.create_oval(13, 13, 21, 21, fill=MINT, outline="")
        wordmark = tk.Frame(brand, bg=BG)
        wordmark.pack(side="left", padx=(10, 0))
        tk.Label(wordmark, text="ARGUS", bg=BG, fg=INK,
                 font=("Ubuntu", 15, "bold")).pack(anchor="w")
        tk.Label(wordmark, text="HAND HYGIENE", bg=BG, fg=MUTED,
                 font=("Ubuntu", 8, "bold")).pack(anchor="w")

        tk.Label(header, text="  INTERACTIVE SIMULATOR  ", bg="#dff3ee", fg="#1f7d68",
                 font=("Ubuntu", 8, "bold"), pady=6).pack(side="right", pady=(4, 0))

        tk.Label(self.body, text="Handwash Monitor", bg=BG, fg=INK,
                 font=("Ubuntu", 30, "bold")).pack(pady=(8, 0))
        tk.Label(
            self.body,
            text="No camera required — walks the same WHO checkpoints the real sink test watches for",
            bg=BG, fg=MUTED, font=("DejaVu Sans", 11),
        ).pack(pady=(2, 16))

        self.canvas = tk.Canvas(self.body, width=1000, height=430, bg=BG, highlightthickness=0)
        self.canvas.pack(padx=30, fill="x")

        controls = tk.Frame(self.body, bg=BG)
        controls.pack(pady=(20, 6))
        self.hand_button = self._button(controls, "Show hands", self.toggle_hands, AQUA)
        self.hand_button.grid(row=0, column=0, padx=6)
        self._button(controls, "Apply soap", self.apply_soap, MINT).grid(row=0, column=1, padx=6)
        self.rub_button = self._button(controls, "Start rubbing", self.toggle_rubbing, CORAL)
        self.rub_button.grid(row=0, column=2, padx=6)
        self._button(controls, "Reset", self.reset, "#dcebea",
                     hover=_shade("#dcebea", 0.94)).grid(row=0, column=3, padx=6)
        tk.Label(self.body, text="Tip: Start rubbing stays active until you click it again.",
                 bg=BG, fg=MUTED, font=("DejaVu Sans", 9)).pack(pady=(2, 22))

    def _button(self, parent, text, command, color, hover=None):
        hover_color = hover or _shade(color, 0.88)
        btn = tk.Button(parent, text=text, command=command, bg=color, fg=INK,
                        activebackground=hover_color, activeforeground=INK,
                        relief="flat", bd=0, padx=18, pady=11,
                        font=("Ubuntu", 10, "bold"), cursor="hand2")
        btn.bind("<Enter>", lambda _e: btn.configure(bg=hover_color))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=color))
        return btn

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

    def _draw_card(self, c, width, height):
        """Rounded card + soft layered shadow as the canvas's own painted
        background - the same elevated-card language as the launcher and
        the real sink test's HUD, instead of a flat rectangle canvas
        widget. Six smoothed polygons is real work for a Tk canvas to
        redo every tick for a background that never actually changes, so
        this only repaints when the card's size changes (i.e. the window
        was resized), not on every 20fps frame."""
        if self._card_size == (width, height):
            return
        c.delete("card_bg")
        radius = 22
        for i in range(5, 0, -1):
            off = i * 2
            _rounded_rect(
                c, off + 4, off + 8, width - off + 4, height - off + 8,
                radius + i, fill=_shade("#bfe4dc", 1 - i * 0.02), outline="",
                tags="card_bg",
            )
        _rounded_rect(c, 0, 0, width, height, radius, fill=CARD, outline=CARD_BORDER,
                      width=1, tags="card_bg")
        self._card_size = (width, height)

    def _draw_stage_visual(self, c, cx, cy, state, now):
        """The center-stage animation: soap bubbles drifting up while
        idle/soaping, two hands that shift when rubbing, or a celebratory
        checkmark / gentle retry glyph once the session resolves - so the
        payoff of finishing a wash reads as an actual moment, not just a
        progress bar hitting 100%."""

        r = RESULT_ICON_RADIUS

        if state.stage == Stage.COMPLETE:
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=MINT, width=4, tags="dynamic")
            c.create_line(cx - 20, cy + 2, cx - 5, cy + 18, fill=MINT, width=6,
                          capstyle="round", tags="dynamic")
            c.create_line(cx - 5, cy + 18, cx + 24, cy - 16, fill=MINT, width=6,
                          capstyle="round", tags="dynamic")
            for i in range(10):
                angle = now * 1.6 + i * (2 * math.pi / 10)
                spread = (r + 20) + (math.sin(now * 3 + i) * 6)
                px, py = cx + math.cos(angle) * spread, cy + math.sin(angle) * spread
                c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=MINT, outline="", tags="dynamic")
            return

        if state.stage == Stage.FAILED:
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=CORAL, width=4, tags="dynamic")
            c.create_line(cx - 18, cy - 18, cx + 18, cy + 18, fill=CORAL, width=6,
                          capstyle="round", tags="dynamic")
            c.create_line(cx - 18, cy + 18, cx + 18, cy - 18, fill=CORAL, width=6,
                          capstyle="round", tags="dynamic")
            return

        for i in range(5):
            phase = now * (0.7 + i * 0.08) + i
            x = cx - 110 + i * 55 + math.sin(phase) * 8
            y = cy - 25 - (phase * 18) % 95
            r = 7 + (i % 3) * 3
            c.create_oval(x - r, y - r, x + r, y + r, fill="#c9f5ee", outline=AQUA,
                          width=2, tags="dynamic")

        if self.hands:
            shift = math.sin(now * 9) * 10 if self.rubbing else 0
            c.create_oval(cx - 125 + shift, cy - 55, cx + 10 + shift, cy + 55,
                          fill="#ffd7a8", outline="#edae72", width=3, tags="dynamic")
            c.create_oval(cx - 10 - shift, cy - 55, cx + 125 - shift, cy + 55,
                          fill="#ffd7a8", outline="#edae72", width=3, tags="dynamic")

    def _draw(self, state, now):
        c = self.canvas
        width = max(1000, c.winfo_width())
        height = max(430, c.winfo_height())

        # The card+shadow are static (only repainted on an actual resize);
        # everything else changes every tick and gets cleared + rebuilt
        # under its own tag, instead of a blanket delete("all") that would
        # force those six background polygons to be resplined 20x/sec for
        # no visual difference.
        self._draw_card(c, width, height)
        c.delete("dynamic")

        tint = STAGE_TINTS[state.stage]
        _pill(c, width / 2, 44, state.stage.value.upper(), tint, CARD, ("Ubuntu", 9, "bold"),
              tags="dynamic")

        c.create_text(width / 2, 90, text=state.message, fill=INK,
                      font=("Ubuntu", 22, "bold"), tags="dynamic")

        self._draw_stage_visual(c, width / 2, 210, state, now)

        # -----------------------------------------------------
        # PROGRESS
        # -----------------------------------------------------
        x1, x2, y = 110, width - 110, 300
        bar_h = 14
        _rounded_rect(c, x1, y, x2, y + bar_h, bar_h / 2, fill=CHIP_OFF, outline="", tags="dynamic")
        fill_w = (x2 - x1) * state.progress
        if fill_w >= bar_h:
            fill_color = MINT if state.stage != Stage.FAILED else CORAL
            _rounded_rect(c, x1, y, x1 + fill_w, y + bar_h, bar_h / 2, fill=fill_color,
                          outline="", tags="dynamic")
        c.create_text(width / 2, y + bar_h + 20,
                      text=f"{state.rubbing_seconds:0.1f} / 20 seconds",
                      fill=INK, font=("Ubuntu", 10, "bold"), tags="dynamic")

        # -----------------------------------------------------
        # STATUS CHIPS
        # -----------------------------------------------------
        chip_y = 372
        chips = (
            ("Hands", state.hands_present),
            ("Soap", state.soap_seen),
            ("Rubbing", state.rubbing_seconds > 0),
        )
        chip_font = ("Ubuntu", 10, "bold")
        gap = 14
        texts = [f"{'✓' if done else '○'}  {label}" for label, done in chips]
        # Measure each label's real rendered width before laying any of
        # them out, so the row centers correctly regardless of font
        # substitution instead of guessing from character count.
        text_ids = [c.create_text(0, 0, text=t, font=chip_font, tags="dynamic") for t in texts]
        widths = [(c.bbox(tid)[2] - c.bbox(tid)[0]) + 26 for tid in text_ids]
        total = sum(widths) + gap * (len(chips) - 1)
        cx = width / 2 - total / 2
        for (_, done), text_id, w in zip(chips, text_ids, widths):
            color = MINT if done else CHIP_OFF
            text_color = CARD if done else MUTED
            _rounded_rect(c, cx, chip_y - 15, cx + w, chip_y + 15, 15, fill=color,
                          outline="", tags="dynamic")
            c.coords(text_id, cx + w / 2, chip_y)
            c.itemconfigure(text_id, fill=text_color)
            c.tag_raise(text_id)
            cx += w + gap


def main():
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
