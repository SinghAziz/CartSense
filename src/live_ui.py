"""Tk live inspection console: feed, selectable detections, evidence anatomy, and manual review."""
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk


class LiveConsole:
    def __init__(self, catalog, on_action):
        self.catalog, self.on_action = catalog, on_action
        self.root = tk.Tk(); self.root.title("CartSense | Live inspection"); self.root.geometry("1920x1080")
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.on_action("quit", {}))
        self.root.bind("<KeyPress-q>", lambda _e: self.on_action("quit", {}))
        self.root.bind("<Escape>", lambda _e: self.on_action("quit", {}))
        self.selected_track = None; self.selected_index = None; self.rows = []; self.image_ref = None; self.scale = 1.0
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL); outer.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(outer, padding=8); right = ttk.Frame(outer, padding=8)
        outer.add(left, weight=3); outer.add(right, weight=2)
        ttk.Label(left, text="LIVE VIDEO FEED — click a box or select a row to inspect", font=("Arial", 13, "bold")).pack(anchor="w")
        self.video = tk.Label(left, bg="#101418", width=960, height=620)
        self.video.pack(fill=tk.BOTH, expand=True, pady=(6, 8)); self.video.bind("<Button-1>", self._click_video)
        self.info = tk.StringVar(value="Starting camera and detector…")
        ttk.Label(left, textvariable=self.info, justify=tk.LEFT, wraplength=980).pack(anchor="w")

        ttk.Label(right, text="DETECTION / TRACK / IDENTITY EVIDENCE  •  double-click a row for full anatomy", font=("Arial", 12, "bold")).pack(anchor="w")
        cart_box = ttk.LabelFrame(right, text="CART STATUS", padding=7); cart_box.pack(fill=tk.X, pady=(5, 4))
        self.cart_text = tk.StringVar(value="Cart is empty")
        ttk.Label(cart_box, textvariable=self.cart_text, justify=tk.LEFT, font=("Arial", 11, "bold"), wraplength=900).pack(anchor="w")
        quick = ttk.LabelFrame(right, text="ADD FROM THE IMAGE / REVIEW", padding=6); quick.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(quick, text="1. Click a box or select its row   2. Choose SKU   3. Add selected").grid(row=0, column=0, columnspan=6, sticky="w")
        self.sku = tk.StringVar()
        self.sku_box = ttk.Combobox(quick, textvariable=self.sku, state="readonly", width=33,
                                    values=[f"{p.sku_id} — {p.name}" for p in catalog.products.values()])
        self.sku_box.grid(row=1, column=0, padx=3, pady=4, sticky="w")
        ttk.Button(quick, text="Add selected image item", command=lambda: self._run("add")).grid(row=1, column=1, padx=3)
        ttk.Button(quick, text="Remove selected", command=lambda: self._run("remove")).grid(row=1, column=2, padx=3)
        ttk.Button(quick, text="Confirm selected review", command=lambda: self._run("confirm")).grid(row=1, column=3, padx=3)
        ttk.Button(quick, text="Reject review", command=lambda: self._run("reject")).grid(row=1, column=4, padx=3)
        ttk.Button(quick, text="Add SKU manually", command=lambda: self._run("manual_add")).grid(row=1, column=5, padx=3)
        cols = ("track", "model", "class", "det", "box", "barcode", "ocr", "sku", "identity", "state")
        self.table = ttk.Treeview(right, columns=cols, show="headings", height=19, selectmode="browse")
        headers = {"track": "Track", "model": "Model", "class": "COCO class", "det": "Det score",
                   "box": "Box xyxy", "barcode": "Barcode", "ocr": "OCR", "sku": "SKU", "identity": "ID evidence", "state": "Lifecycle"}
        widths = {"track": 50, "model": 85, "class": 90, "det": 65, "box": 125, "barcode": 95, "ocr": 70, "sku": 75, "identity": 105, "state": 105}
        for col in cols:
            self.table.heading(col, text=headers[col]); self.table.column(col, width=widths[col], anchor=tk.W)
        self.table.pack(fill=tk.BOTH, expand=True, pady=(6, 10)); self.table.bind("<<TreeviewSelect>>", self._select_row)
        self.table.bind("<Double-1>", self._show_anatomy)

        self.metrics = tk.StringVar(value="Component diagnostics loading…")
        ttk.Label(right, textvariable=self.metrics, justify=tk.LEFT, wraplength=900,
                  font=("Menlo", 10)).pack(fill=tk.X, anchor="w", pady=4)
        form = ttk.LabelFrame(right, text="BARCODE / CLASS LEARNING FOR NEXT TIME", padding=6); form.pack(fill=tk.X, pady=4)
        ttk.Label(form, text="Barcode read from selected item").grid(row=0, column=0, sticky="w")
        self.barcode = tk.StringVar(); ttk.Entry(form, textvariable=self.barcode, width=27).grid(row=0, column=1, sticky="ew", padx=4)
        form.columnconfigure(1, weight=1)
        actions = ttk.Frame(form); actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        ttk.Button(actions, text="Save barcode rule", command=lambda: self._run("learn_barcode")).pack(side=tk.LEFT, padx=2)
        self.learn_class_button = ttk.Button(actions, text="Learn class rule", command=lambda: self._run("learn_class"))
        self.learn_class_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Forget class rule", command=lambda: self._run("forget_class")).pack(side=tk.LEFT, padx=2)
        ttk.Label(form, text="A learned class rule maps a generic class (for example, bottle) to one SKU; it cannot distinguish brands or package variants.",
                  wraplength=545).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(right, text="Scores are raw detector outputs, not calibrated probabilities. Barcode/OCR evidence is shown separately.",
                  wraplength=570).pack(anchor="w", pady=4)

    def _sku_id(self):
        text = self.sku.get().strip()
        return text.split(" — ", 1)[0] if text else ""

    def _run(self, action):
        row = self._selected_row()
        args = {"track_id": self.selected_track, "label": row.get("label") if row else None,
                "barcode": self.barcode.get().strip(), "sku_id": self._sku_id(), "row": row}
        try:
            message = self.on_action(action, args)
            if message: self.info.set(message)
        except Exception as exc:
            messagebox.showerror("CartSense action", str(exc), parent=self.root)

    def _selected_row(self):
        if self.selected_index is not None and self.selected_index < len(self.rows):
            return self.rows[self.selected_index]
        return next((r for r in self.rows if r.get("track_id") == self.selected_track), None)

    def _select_row(self, _event=None):
        ids = self.table.selection()
        if ids:
            self.selected_index = int(ids[0]); row = self.rows[self.selected_index]; self.selected_track = row.get("track_id")
            if row.get("barcode"):
                self.barcode.set(row["barcode"])
            self.info.set(f"Selected: track {self.selected_track or 'not tracked'} | {row['label']} | {row['identity']}")
            self._update_rule_buttons(row)

    def _update_rule_buttons(self, row=None):
        row = row or self._selected_row()
        allowed = bool(row and row.get("cart_trackable", True))
        self.learn_class_button.configure(state=tk.NORMAL if allowed else tk.DISABLED)

    def _show_anatomy(self, _event=None):
        row = self._selected_row()
        if not row: return
        win = tk.Toplevel(self.root); win.title("Selected detection anatomy")
        body = tk.Text(win, width=105, height=28, wrap=tk.WORD)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        details = {k: v for k, v in row.items() if k != "index"}
        body.insert("1.0", __import__("json").dumps(details, indent=2, default=str)); body.configure(state=tk.DISABLED)

    def _click_video(self, event):
        x, y = event.x / self.scale, event.y / self.scale
        hits = [r for r in self.rows if r["bbox"][0] <= x <= r["bbox"][2] and r["bbox"][1] <= y <= r["bbox"][3]]
        if hits:
            row = min(hits, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
            self.selected_index = row["index"]; self.selected_track = row.get("track_id")
            children = self.table.get_children()
            if row["index"] < len(children):
                self.table.selection_set(children[row["index"]]); self.table.see(children[row["index"]])

    def update(self, frame, rows, metrics, cart_summary=""):
        self.rows = rows
        self.metrics.set(metrics)
        self.cart_text.set(cart_summary or "Cart is empty")
        self.table.delete(*self.table.get_children())
        for idx, row in enumerate(rows):
            row["index"] = idx
            box = ",".join(str(int(v)) for v in row["bbox"])
            values = (row.get("track_id") or "—", row["model"], row["label"], f"{row['confidence']:.3f}",
                      box, row.get("barcode") or "—", row.get("ocr") or "disabled",
                      row.get("sku") or "unknown", row["identity"], row.get("lifecycle", "observed"))
            self.table.insert("", tk.END, iid=str(idx), values=values)
        if self.selected_track is not None:
            for idx, row in enumerate(rows):
                if row.get("track_id") == self.selected_track:
                    self.selected_index = idx; self.table.selection_set(str(idx)); break
        elif self.selected_index is not None and self.selected_index < len(rows):
            self.table.selection_set(str(self.selected_index))
        self._update_rule_buttons()
        h, w = frame.shape[:2]
        max_w, max_h = 960, 620
        self.scale = min(max_w / w, max_h / h, 1.0)
        shown = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        self.image_ref = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=self.image_ref, width=int(w * self.scale), height=int(h * self.scale))
        self.root.update_idletasks(); self.root.update()

    def close(self):
        try: self.root.destroy()
        except tk.TclError: pass
