"""lora_container_gui: a window for lora_container.

Open a .safetensors, add up to six images (the first is the cover: preview and icon),
fill in the details, save. Opening a file that already has covers shows them.

Needs Python 3.8+ with tkinter, and Pillow. Dropping files onto the window works
if the optional tkinterdnd2 package is installed.
"""
import base64, io, os, sys, threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

import lora_container as lc

BG, PANEL, RAISED, LINE = "#1F1F1E", "#262626", "#3D3D3B", "#353534"
TEXT, MUTED, ACCENT, ACCENT_HI = "#ECE9E2", "#B5B0A7", "#A66BD6", "#B987E0"
SLOTS = lc.MAX_COVERS
THUMB = 72
LABELS = [("title", "Name"), ("author", "Author"), ("base_model", "Base model"),
          ("trigger_words", "Trigger words (comma separated)"), ("tags", "Tags (comma separated)")]
IMAGE_TYPES = [("Images", " ".join("*" + e for e in lc.IMAGE_EXT)), ("All files", "*")]
MODEL_TYPES = [("safetensors", "*.safetensors"), ("All files", "*")]


def photo(im):
    """A Tk image from a Pillow image, without needing Pillow's ImageTk."""
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return tk.PhotoImage(data=base64.b64encode(buf.getvalue()))


def fitted(im, w, h):
    im = im.copy()
    im.thumbnail((max(1, w), max(1, h)))
    return im


class Cover:
    """One image: src is a file path, or the JPEG bytes already inside the model."""

    def __init__(self, src):
        from PIL import Image, ImageOps
        self.src = src
        with Image.open(io.BytesIO(src) if isinstance(src, bytes) else src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1280, 1280))
            self.image = im
        self.thumb = photo(fitted(self.image, THUMB, THUMB))
        self.name = "in the file" if isinstance(src, bytes) else os.path.basename(src)
        self.embedded = isinstance(src, bytes)


class Btn(tk.Label):
    """A flat button that looks the same on every platform."""

    def __init__(self, parent, text, command, primary=False):
        self.base, self.hot = (ACCENT, ACCENT_HI) if primary else (RAISED, ACCENT)
        super().__init__(parent, text=text, bg=self.base, fg=TEXT, font=parent.winfo_toplevel().f_bold,
                         padx=12, pady=8, cursor="hand2", takefocus=1, highlightthickness=1,
                         highlightbackground=self.base, highlightcolor=TEXT)
        self.command, self.on = command, True
        self.bind("<Enter>", lambda e: self.on and self.config(bg=self.hot))
        self.bind("<Leave>", lambda e: self.config(bg=self.base if self.on else PANEL))
        for ev in ("<Button-1>", "<Return>", "<space>"):
            self.bind(ev, lambda e: self.on and self.command())

    def enable(self, on):
        self.on = bool(on)
        self.config(bg=self.base if on else PANEL, fg=TEXT if on else MUTED, cursor="hand2" if on else "arrow",
                    highlightbackground=self.base if on else LINE)


class App:
    def __init__(self, root):
        self.root = root
        self.model, self.covers, self.sel = None, [], 0
        self.dirty = self.busy = False
        self.loaded, self.error = {}, None
        self._fit_job = self._shown = None
        family = "Inter" if "Inter" in tkfont.families(root) else tkfont.nametofont("TkDefaultFont").actual("family")
        root.f = (family, -13)
        root.f_bold = (family, -13, "bold")
        root.f_head = (family, -15, "bold")
        root.title("LoRA Container")
        root.configure(bg=BG)
        root.geometry("1200x740")
        root.minsize(1160, 700)
        self._build()
        root.bind("<Left>", lambda e: self._arrow(e, -1))
        root.bind("<Right>", lambda e: self._arrow(e, 1))
        self._drop_setup()
        self.refresh()

    # ----- layout -----

    def _build(self):
        r = self.root
        top = tk.Frame(r, bg=PANEL, height=44)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="LoRA Container", bg=PANEL, fg=TEXT, font=(r.f[0], -14, "bold")).pack(side="left", padx=(16, 12))
        tk.Frame(top, bg=RAISED, width=1, height=16).pack(side="left")
        self.title = tk.Label(top, text="No file open", bg=PANEL, fg=MUTED, font=(r.f[0], -14))
        self.title.pack(side="left", padx=12)
        tk.Frame(r, bg=LINE, height=1).pack(fill="x")
        self.status = tk.Label(r, text="", bg=PANEL, fg=MUTED, font=r.f, anchor="w", padx=16, pady=6)
        self.status.pack(side="bottom", fill="x")
        body = tk.Frame(r, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=PANEL, width=290)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self._head(left, "Model")
        self.b_open = self._btn(left, "Open safetensors...", self.open_dialog)
        self.info = tk.Text(left, bg=BG, fg=MUTED, font=r.f, relief="flat", wrap="char", height=11, padx=10, pady=10,
                            highlightthickness=0, insertwidth=0, spacing3=4)
        self.info.pack(fill="x", padx=14, pady=(10, 4))
        self.b_all = self._btn(left, "All metadata...", self.metadata_window)
        self.b_verify = self._btn(left, "Verify", self.verify)
        self.b_extract = self._btn(left, "Extract images...", self.extract_dialog)
        self.b_strip = self._btn(left, "Remove embedded data...", self.strip_dialog)

        right = tk.Frame(body, bg=PANEL, width=330)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.b_inplace = self._btn(right, "Save in place", self.save_in_place, side="bottom", pady=(6, 14))
        self.b_save = self._btn(right, "Save as...", self.save_as, primary=True, side="bottom")
        self._head(right, "Details")
        self.entries = {}
        for key, label in LABELS:
            tk.Label(right, text=label, bg=PANEL, fg=MUTED, font=r.f, anchor="w").pack(fill="x", padx=14, pady=(6, 2))
            e = tk.Entry(right, **self._field())
            e.pack(fill="x", padx=14, ipady=6)
            self.entries[key] = e
        tk.Label(right, text="Description", bg=PANEL, fg=MUTED, font=r.f, anchor="w").pack(fill="x", padx=14, pady=(6, 2))
        self.description = tk.Text(right, height=3, wrap="word", padx=6, pady=6, **self._field())
        self.description.pack(fill="x", padx=14)
        tk.Label(right, text="Other metadata (key=value, one per line)", bg=PANEL, fg=MUTED, font=r.f,
                 anchor="w").pack(fill="x", padx=14, pady=(6, 2))
        self.other = tk.Text(right, height=2, wrap="none", padx=6, pady=6, **self._field())
        self.other.pack(fill="x", padx=14)
        self.sidecar = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Also update the .cm-info.json beside it", variable=self.sidecar, bg=PANEL, fg=TEXT,
                       selectcolor=BG, activebackground=PANEL, activeforeground=TEXT, highlightthickness=0, font=r.f,
                       anchor="w").pack(fill="x", padx=10, pady=(8, 0))

        mid = tk.Frame(body, bg=BG)
        mid.pack(side="left", fill="both", expand=True, padx=16, pady=16)
        tools = tk.Frame(mid, bg=BG)
        tools.pack(side="bottom", fill="x", pady=(10, 0))
        self.b_add = Btn(tools, "Add images...", self.add_dialog)
        self.b_left = Btn(tools, "Move left", lambda: self.move(-1))
        self.b_right = Btn(tools, "Move right", lambda: self.move(1))
        self.b_remove = Btn(tools, "Remove", self.remove_selected)
        for b in (self.b_add, self.b_left, self.b_right, self.b_remove):
            b.pack(side="left", padx=(0, 8))
        strip = tk.Frame(mid, bg=BG)
        strip.pack(side="bottom", pady=(8, 0))
        self.caption = tk.Label(mid, text="", bg=BG, fg=MUTED, font=r.f)
        self.caption.pack(side="bottom", pady=(10, 0))
        self.slots = []
        for i in range(SLOTS):
            cell = tk.Frame(strip, bg=BG)
            cell.pack(side="left", padx=4)
            box = tk.Frame(cell, bg=PANEL, width=THUMB + 4, height=THUMB + 4, highlightthickness=2, highlightbackground=BG)
            box.pack()
            box.pack_propagate(False)
            pic = tk.Label(box, bg=PANEL, fg=MUTED, text="+", font=(r.f[0], -22), cursor="hand2")
            pic.pack(fill="both", expand=True)
            pic.bind("<Button-1>", lambda e, i=i: self.click_slot(i))
            tk.Label(cell, text="Cover" if i == 0 else "Gallery %d" % i, bg=BG, fg=MUTED, font=r.f).pack(pady=(4, 0))
            self.slots.append((box, pic))
        self.stage = tk.Label(mid, bg=PANEL, fg=MUTED, font=r.f)
        self.stage.pack(fill="both", expand=True)
        self.stage.bind("<Configure>", lambda e: self._refit())

    def _field(self):
        return dict(bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat", highlightthickness=1,
                    highlightbackground=LINE, highlightcolor=ACCENT, font=self.root.f)

    def _head(self, parent, text):
        tk.Label(parent, text=text, bg=PANEL, fg=TEXT, font=self.root.f_head, anchor="w").pack(fill="x", padx=14, pady=(14, 8))

    def _btn(self, parent, text, command, primary=False, side="top", pady=(6, 0)):
        b = Btn(parent, text, command, primary)
        b.pack(side=side, fill="x", padx=14, pady=pady)
        return b

    def _drop_setup(self):
        try:
            from tkinterdnd2 import DND_FILES
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", lambda e: self.dropped(self.root.tk.splitlist(e.data)))
            self.can_drop = True
        except Exception:
            self.can_drop = False

    # ----- state to screen -----

    def say(self, text):
        self.status.config(text=text)

    def refresh(self):
        have = self.model is not None
        self.title.config(text=os.path.basename(self.model) if have else "No file open")
        for i, (box, pic) in enumerate(self.slots):
            if i < len(self.covers):
                pic.config(image=self.covers[i].thumb, text="")
            else:
                pic.config(image="", text="+")
            box.config(highlightbackground=ACCENT if self.covers and i == self.sel else BG)
        for b in (self.b_all, self.b_verify, self.b_save, self.b_inplace):
            b.enable(have and not self.busy)
        embedded = have and bool(self.loaded.get("embedded"))
        self.b_extract.enable(embedded and not self.busy)
        self.b_strip.enable(embedded and not self.busy)
        self.b_open.enable(not self.busy)
        self.b_add.enable(have and len(self.covers) < SLOTS)
        self.b_left.enable(self.covers and self.sel > 0)
        self.b_right.enable(self.covers and self.sel < len(self.covers) - 1)
        self.b_remove.enable(bool(self.covers))
        if self.covers:
            c = self.covers[self.sel]
            self.caption.config(text="%s: %s" % ("Cover (preview and icon)" if self.sel == 0 else "Gallery %d" % self.sel, c.name))
        else:
            self.caption.config(text="")
        self._shown = None
        self._refit()

    def _refit(self):
        if self._fit_job:
            self.root.after_cancel(self._fit_job)
        self._fit_job = self.root.after(60, self._fit)

    def _fit(self):
        self._fit_job = None
        if not self.covers:
            hint = "Open a .safetensors file" + (", or drop one here" if self.can_drop else "")
            self.stage.config(image="", text=hint if self.model is None else "No images yet: add up to six. The first is the cover.")
            self._big = None
            return
        w, h = self.stage.winfo_width() - 16, self.stage.winfo_height() - 16
        key = (id(self.covers[self.sel]), w, h)
        if key != self._shown:
            self._shown = key
            self._big = photo(fitted(self.covers[self.sel].image, w, h))
            self.stage.config(image=self._big, text="")

    def _set_info(self, text):
        self.info.config(state="normal")
        self.info.delete("1.0", "end")
        self.info.insert("1.0", text)
        self.info.config(state="disabled")

    def _arrow(self, event, step):
        if isinstance(event.widget, (tk.Entry, tk.Text)) or not self.covers:
            return
        self.sel = min(max(self.sel + step, 0), len(self.covers) - 1)
        self.refresh()

    # ----- background work, so a large file does not freeze the window -----

    def _work(self, label, fn, done):
        if self.busy:
            return
        self.busy, self.error = True, None
        self.say(label)
        self.refresh()
        box = {}

        def run():
            try:
                box["result"] = fn()
            except Exception as e:  # shown to the user below
                box["error"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()

        def poll():
            if t.is_alive():
                self.root.after(40, poll)
                return
            self.busy = False
            if "error" in box:
                self.fail(box["error"])
            else:
                done(box["result"])
            self.refresh()

        self.root.after(40, poll)

    def wait(self):
        """Block until background work is finished (for scripts and tests)."""
        while self.busy:
            self.root.update()
        self.root.update()

    def fail(self, e):
        self.error = str(e)
        self.say("Not done: %s" % e)
        if not getattr(self, "quiet", False):
            messagebox.showerror("LoRA Container", str(e), parent=self.root)

    # ----- model -----

    def open_dialog(self):
        p = filedialog.askopenfilename(parent=self.root, title="Open safetensors", filetypes=MODEL_TYPES)
        if p:
            self.open_model(p)

    def open_model(self, path):
        try:
            s = lc.show(path)
            covers = [Cover(c) for c in lc.get_covers(lc.read_header(path)[1].get("__metadata__", {}))]
        except Exception as e:
            self.fail(e)
            return False
        found = 0
        if not covers:  # nothing embedded yet: take the loose images beside the model, by their usual names
            try:
                loose = lc.find_covers(path)[:SLOTS]
            except OSError:
                loose = []
            for p in loose:
                try:
                    covers.append(Cover(p))
                except Exception:
                    pass
            found = len(covers)
        self.model, self.covers, self.sel, self.dirty = path, covers, 0, bool(found)
        beside = None if s["cm_info"] else lc._load_json(lc.sidecar_path(path))
        cm, md = s["cm_info"] or beside or {}, s["metadata"]
        self.loaded = {"title": cm.get("ModelName") or "", "author": cm.get("AuthorUsername") or "",
                       "base_model": cm.get("BaseModel") or "", "trigger_words": ", ".join(cm.get("TrainedWords") or []),
                       "tags": ", ".join(cm.get("Tags") or []), "description": cm.get("ModelDescription") or "",
                       "embedded": bool(covers or s["cm_info"] or any(k in md for k in lc.OWN_KEYS))}
        for key, e in self.entries.items():
            e.delete(0, "end")
            e.insert(0, self.loaded[key])
        self.description.delete("1.0", "end")
        self.description.insert("1.0", self.loaded["description"])
        self.other.delete("1.0", "end")
        lines = ["%.2f MiB, %d tensors" % (os.path.getsize(path) / 2**20, s["tensors"]),
                 "Header %.1f KiB" % (s["header_bytes"] / 1024),
                 "%d images found beside the file, not yet embedded" % found if found
                 else "%d embedded images" % len(covers) if covers else "No embedded images"]
        if beside:
            lines.append("Details taken from the .cm-info.json beside the file")
        if md.get("tensor_data_sha256"):
            lines.append("Identity (tensor data SHA-256):\n" + md["tensor_data_sha256"])
        if md.get("model_sha256_original"):
            lines.append("Original file SHA-256:\n" + md["model_sha256_original"])
        self._set_info("\n".join(lines))
        self.say("Opened %s%s" % (path, ". Found %d images%s beside it: ready to save" % (
            found, " and a cm-info" if beside else "") if found else ""))
        self.refresh()
        return True

    def metadata_window(self):
        s = lc.show(self.model)
        w = tk.Toplevel(self.root, bg=BG)
        w.title("All metadata: " + os.path.basename(self.model))
        w.geometry("760x560")
        bar = tk.Scrollbar(w)
        bar.pack(side="right", fill="y")
        t = tk.Text(w, bg=BG, fg=TEXT, font=self.root.f, relief="flat", wrap="word", padx=14, pady=12,
                    highlightthickness=0, yscrollcommand=bar.set, spacing3=4)
        t.pack(fill="both", expand=True)
        bar.config(command=t.yview)
        out = ["cover_%d: %.1f KiB" % (i, n * 0.75 / 1024) for i, n in enumerate(s["covers"], 1)]
        out += ["%s: %s" % kv for kv in s["metadata"].items()]
        out += ["cm_info.%s: %s" % (k, v) for k, v in (s["cm_info"] or {}).items()]
        t.insert("1.0", "\n".join(out) or "No metadata")
        t.config(state="disabled")
        return w

    def verify(self):
        def done(r):
            self.say("Verify: %s. %s" % ({"ok": "all good", "plain": "nothing embedded", "failed": "FAILED"}[r["status"]],
                                         "; ".join(r["problems"] + r["notes"])))
            if r["status"] == "failed" and not getattr(self, "quiet", False):
                messagebox.showerror("Verify", "\n".join(r["problems"]), parent=self.root)
        self._work("Checking...", lambda: lc.verify(self.model), done)

    def extract_dialog(self):
        d = filedialog.askdirectory(parent=self.root, title="Folder to write the images into")
        if d:
            self.extract(d)

    def extract(self, folder):
        self._work("Extracting...", lambda: lc.extract(self.model, folder, sm_names=True),
                   lambda paths: self.say("Wrote %d files to %s" % (len(paths), folder)))

    def strip_dialog(self):
        stem = os.path.splitext(self.model)[0]
        p = filedialog.asksaveasfilename(parent=self.root, title="Save a copy without the embedded data", filetypes=MODEL_TYPES,
                                         defaultextension=".safetensors", initialfile=os.path.basename(stem) + "_plain.safetensors",
                                         initialdir=os.path.dirname(self.model))
        if p:
            self.strip(p)

    def strip(self, out):
        def done(r):
            self.say("Wrote %s: %s" % (out, {True: "identical to the original file", None: "tensor data intact",
                                             False: "tensor data intact; header not identical to the original"}[r["exact"]]))
        self._work("Removing embedded data...", lambda: lc.strip(self.model, out, force=True), done)

    # ----- images -----

    def click_slot(self, i):
        if i < len(self.covers):
            self.sel = i
            self.refresh()
        elif self.model:
            self.add_dialog()

    def add_dialog(self):
        ps = filedialog.askopenfilenames(parent=self.root, title="Add images (the first is the cover)", filetypes=IMAGE_TYPES)
        if ps:
            self.add_images(list(ps))

    def add_images(self, paths):
        added = skipped = 0
        for p in paths:
            if len(self.covers) >= SLOTS:
                skipped += 1
                continue
            try:
                self.covers.append(Cover(p))
                added += 1
            except Exception as e:
                self.fail("cannot read image %s: %s" % (p, e))
        if added:
            self.dirty = True
            self.sel = len(self.covers) - 1
        self.say("Added %d image%s%s" % (added, "" if added == 1 else "s",
                                          "; %d left out, six is the limit" % skipped if skipped else ""))
        self.refresh()

    def move(self, step):
        j = self.sel + step
        if self.covers and 0 <= j < len(self.covers):
            self.covers[self.sel], self.covers[j] = self.covers[j], self.covers[self.sel]
            self.sel, self.dirty = j, True
            self.refresh()

    def remove_selected(self):
        if self.covers:
            del self.covers[self.sel]
            self.sel, self.dirty = max(0, min(self.sel, len(self.covers) - 1)), True
            self.refresh()

    def dropped(self, paths):
        models = [p for p in paths if p.lower().endswith(".safetensors")]
        images = [p for p in paths if p.lower().endswith(lc.IMAGE_EXT)]
        if models:
            self.open_model(models[0])
        if images and self.model:
            self.add_images(images)

    # ----- save -----

    def save_as(self):
        stem = os.path.splitext(self.model)[0]
        p = filedialog.asksaveasfilename(parent=self.root, title="Save as", filetypes=MODEL_TYPES, defaultextension=".safetensors",
                                         initialfile=os.path.basename(stem) + "_container.safetensors",
                                         initialdir=os.path.dirname(self.model))
        if p:
            self.save(p)

    def save_in_place(self):
        if messagebox.askyesno("Save in place", "Change %s itself?\n\nIts whole-file hash will change. The tensor data will not."
                               % os.path.basename(self.model), parent=self.root):
            self.save(self.model)

    def values(self):
        v = {key: e.get().strip() for key, e in self.entries.items()}
        v["description"] = self.description.get("1.0", "end").strip()
        return v

    def save(self, out):
        try:
            covers = None
            if self.dirty:
                if not self.covers:
                    raise lc.Error("There are no images left. To take them all out, use Remove embedded data.")
                covers = [c.src for c in self.covers]
            fields = {k: v for k, v in self.values().items() if v != self.loaded.get(k, "")}
            meta = {}
            for line in self.other.get("1.0", "end").splitlines():
                if line.strip():
                    k, sep, v = line.partition("=")
                    if not sep or not k.strip():
                        raise lc.Error("Other metadata must be key=value, got: %s" % line)
                    meta[k.strip()] = v.strip()
        except lc.Error as e:
            self.fail(e)
            return
        model, sidecar = self.model, self.sidecar.get()

        def done(rep):
            self.open_model(rep["out"])
            self.say("Saved %s: %d images, %.1f KiB of header" % (rep["out"], rep["covers"], rep["header_new"] / 1024))

        self._work("Saving...", lambda: lc.embed(model, covers, out, True, True, meta, None, fields, True, sidecar), done)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        import PIL  # noqa: F401
    except ImportError:
        sys.exit("Pillow is needed: pip install Pillow")
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    app = App(root)
    if argv:
        root.after(100, lambda: app.open_model(argv[0]))
    root.mainloop()


if __name__ == "__main__":
    main()
