"""
Advanced Image Restoration Tool
--------------------------------
Requirements:  pip install opencv-python numpy pillow
Run:           python image_restoration_tool.py
"""
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

MAX_HISTORY = 20
SAVE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ----------------------------------------------------------------------------
# Pure image-processing functions (never modify their input)
# ----------------------------------------------------------------------------
def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def to_float(img):
    return img.astype(np.float32) / 255.0


def to_u8(img):
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def estimate_noise(gray):
    """Immerkaer's fast noise sigma estimate (in gray levels)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(conv[1:-1, 1:-1])) / 6.0)


def blur_score(gray):
    """Variance of Laplacian, measured at a fixed resolution so it is comparable."""
    s = 800.0 / max(gray.shape[:2])
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return {
        "noise": estimate_noise(gray),
        "sharpness": blur_score(gray),
        "contrast": float(lab[:, :, 0].std()),
        "cast": float(np.hypot(lab[:, :, 1].mean() - 128, lab[:, :, 2].mean() - 128)),
    }


def gaussian_psf(sigma):
    k = int(2 * np.ceil(3 * sigma) + 1)
    g = cv2.getGaussianKernel(k, sigma)
    return (g @ g.T).astype(np.float32)


def unsharp(img, sigma, amount):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def wiener_deconvolve(img, sigma, k_noise):
    """Frequency-domain Wiener deconvolution assuming a Gaussian blur (per channel, colour kept)."""
    psf = gaussian_psf(sigma)
    ks = psf.shape[0]
    pad = ks
    src = cv2.copyMakeBorder(to_float(img), pad, pad, pad, pad, cv2.BORDER_REFLECT)
    h, w = src.shape[:2]
    kernel = np.zeros((h, w), np.float32)
    kernel[:ks, :ks] = psf
    kernel = np.roll(kernel, (-(ks // 2), -(ks // 2)), axis=(0, 1))
    H = np.fft.rfft2(kernel)
    G = np.conj(H) / (np.abs(H) ** 2 + k_noise)
    out = np.empty_like(src)
    for c in range(src.shape[2]):
        out[:, :, c] = np.fft.irfft2(np.fft.rfft2(src[:, :, c]) * G, s=(h, w))
    return to_u8(out[pad:-pad, pad:-pad])


def richardson_lucy(img, sigma, iterations):
    """Richardson-Lucy deconvolution (Gaussian PSF, colour kept)."""
    psf = gaussian_psf(sigma)  # symmetric, so no flip needed
    observed = np.maximum(to_float(img), 1e-4)
    est = observed.copy()
    for _ in range(int(iterations)):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        ratio = observed / np.maximum(conv, 1e-4)
        est = est * cv2.filter2D(ratio, -1, psf, borderType=cv2.BORDER_REFLECT)
        est = np.clip(est, 0, 1.5)
    return to_u8(est)


def hist_equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_filter(img, clip=2.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def white_balance(img):
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)
    gains = np.clip(means.mean() / np.maximum(means, 1e-6), 0.5, 2.0)
    return np.clip(f * gains, 0, 255).astype(np.uint8)


def edge_enhance(img, amount=0.5):
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    return np.clip(f - amount * lap, 0, 255).astype(np.uint8)


def adaptive_binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    block = odd(max(11, min(gray.shape[:2]) / 40))
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, block, 2)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def build_lut(brightness, contrast, gamma):
    x = np.arange(256, dtype=np.float32)
    y = np.clip((x + brightness - 128.0) * contrast + 128.0, 0, 255)
    y = 255.0 * np.power(y / 255.0, 1.0 / max(gamma, 1e-3))
    return np.clip(y, 0, 255).astype(np.uint8)


# BGR-ordered sepia matrix (rows = output B, G, R; columns = input B, G, R)
SEPIA_BGR = np.array([[0.131, 0.534, 0.272],
                      [0.168, 0.686, 0.349],
                      [0.189, 0.769, 0.393]], np.float32)


def auto_restore(img, level, style, report):
    """Analysis-driven restoration. level: 1-3, style: natural | vintage | bw."""
    out = img.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    sigma = estimate_noise(gray)
    sharp = blur_score(gray)
    report(f"Analyzed: noise sigma {sigma:.1f}, sharpness {sharp:.0f}")

    # 1. Denoise (strength follows the measured noise)
    mult = {1: 0.7, 2: 1.0, 3: 1.4}[level]
    h = float(np.clip(sigma * mult, 3.0 if level == 3 else 0.0, 15.0))
    if h >= 2.0:
        report("Reducing noise...")
        window = 21 if out.shape[0] * out.shape[1] < 8_000_000 else 15
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h, 7, window)

    # 2. Tone / colour / sharpening in LAB (touches luminance only for sharpening)
    report("Correcting tone and colour...")
    L, A, B = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
    if style == "natural":
        k = {1: 0.35, 2: 0.5, 3: 0.7}[level]
        da = float(np.clip(k * (A.mean() - 128), -10, 10))
        db = float(np.clip(k * (B.mean() - 128), -10, 10))
        A = np.clip(A.astype(np.float32) - da, 0, 255).astype(np.uint8)
        B = np.clip(B.astype(np.float32) - db, 0, 255).astype(np.uint8)
    if style != "bw" and (level >= 2 or L.std() < 45):
        clip = {1: 1.5, 2: 2.0, 3: 3.0}[level]
        w = {1: 0.5, 2: 0.7, 3: 1.0}[level]
        Lc = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(L)
        L = cv2.addWeighted(Lc, w, L, 1 - w, 0)

    report("Sharpening...")
    amount = {1: 0.5, 2: 0.9, 3: 1.4}[level]
    if sharp < 30:
        amount *= 1.4
    elif sharp > 300:
        amount *= 0.5
    radius = float(np.clip(max(img.shape[:2]) / 1200.0, 0.8, 3.0))
    L = cv2.addWeighted(L, 1 + amount, cv2.GaussianBlur(L, (0, 0), radius), -amount, 0)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)

    # 3. Style
    report(f"Applying {style} style...")
    if style == "natural":
        sat = {1: 1.0, 2: 1.05, 3: 1.15}[level]
        if sat != 1.0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif style == "vintage":
        f = out.astype(np.float32)
        sepia = cv2.transform(f, SEPIA_BGR)
        f = 0.5 * sepia + 0.5 * f
        rows, cols = f.shape[:2]
        vig = cv2.getGaussianKernel(rows, rows * 0.6) @ cv2.getGaussianKernel(cols, cols * 0.6).T
        vig = 0.45 + 0.55 * (vig / vig.max())
        f = f * vig[:, :, None].astype(np.float32) * 0.92 + 10
        out = np.clip(f, 0, 255).astype(np.uint8)
    elif style == "bw":
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return out


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
class ImagePane(ttk.Frame):
    """Shows a BGR image scaled to fit, with a caption. Redraws on resize."""

    def __init__(self, parent, title):
        super().__init__(parent)
        self.title = title
        self.caption = ttk.Label(self, text=title, anchor="center", font=("Helvetica", 10, "bold"))
        self.caption.pack(fill=tk.X, pady=(0, 2))
        self.canvas = tk.Canvas(self, bg="#262626", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.img = None
        self._photo = None
        self._job = None
        self.canvas.bind("<Configure>", self._on_configure)

    def set_image(self, img):
        self.img = img
        if img is None:
            self.caption.config(text=self.title)
        else:
            h, w = img.shape[:2]
            self.caption.config(text=f"{self.title}  ({w} x {h})")
        self._draw()

    def _on_configure(self, _event):
        if self._job:
            self.after_cancel(self._job)
        self._job = self.after(60, self._draw)

    def _draw(self):
        self._job = None
        c = self.canvas
        c.delete("all")
        cw, ch = c.winfo_width(), c.winfo_height()
        if cw < 20 or ch < 20:
            return
        if self.img is None:
            c.create_text(cw // 2, ch // 2, text="Open an image  (Ctrl+O)",
                          fill="#9a9a9a", font=("Helvetica", 12))
            return
        h, w = self.img.shape[:2]
        s = min(cw / w, ch / h, 3.0)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(self.img, (nw, nh),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        c.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")


def add_slider(parent, text, var, lo, hi, fmt="{:.1f}"):
    f = ttk.Frame(parent)
    f.pack(fill=tk.X, padx=5, pady=1)
    ttk.Label(f, text=text, width=13).pack(side=tk.LEFT)
    val = ttk.Label(f, width=6, anchor="e")
    val.pack(side=tk.RIGHT)
    ttk.Scale(f, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL).pack(
        side=tk.LEFT, fill=tk.X, expand=True)

    def update(*_):
        val.config(text=fmt.format(var.get()))
    var.trace_add("write", update)
    update()
    return f


def button_row(parent, *specs):
    f = ttk.Frame(parent)
    f.pack(fill=tk.X, padx=5, pady=2)
    for text, cmd in specs:
        ttk.Button(f, text=text, command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
    return f


# ----------------------------------------------------------------------------
# Inpainting dialog
# ----------------------------------------------------------------------------
class InpaintDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app
        self.title("Smart Inpainting - paint over what you want removed")
        self.geometry("1000x720")
        self.minsize(720, 500)
        self.transient(app.root)

        self.src = app.current_img.copy()
        self.mask = np.zeros(self.src.shape[:2], np.uint8)
        self.undo_stack = []
        self.brush = tk.DoubleVar(value=15)
        self.radius = tk.DoubleVar(value=3)
        self.method = tk.StringVar(value="telea")
        self.s = 1.0
        self.ox = self.oy = 0
        self.last = None
        self.last_canvas = None
        self.photo = None

        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(bar, text="Brush").pack(side=tk.LEFT)
        ttk.Scale(bar, from_=2, to=80, variable=self.brush, orient=tk.HORIZONTAL,
                  length=120).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(bar, text="Radius").pack(side=tk.LEFT)
        ttk.Scale(bar, from_=1, to=15, variable=self.radius, orient=tk.HORIZONTAL,
                  length=90).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Radiobutton(bar, text="Telea", variable=self.method, value="telea").pack(side=tk.LEFT)
        ttk.Radiobutton(bar, text="Navier-Stokes", variable=self.method, value="ns").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(bar, text="Undo stroke", command=self.undo_stroke).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Apply", command=self.apply).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Close", command=self.close).pack(side=tk.RIGHT, padx=2)

        self.canvas = tk.Canvas(self, bg="#262626", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))
        ttk.Label(self, text="Tip: Telea is best for thin scratches and small blemishes; "
                             "large regions will look smeared.").pack(pady=(0, 6))

        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Control-z>", lambda e: self.undo_stroke())
        self.protocol("WM_DELETE_WINDOW", self.close)
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass

    # -- display -----------------------------------------------------------
    def render(self):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 20 or ch < 20:
            return
        h, w = self.src.shape[:2]
        s = min(cw / w, ch / h, 3.0)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        self.s = nw / w
        small = cv2.resize(self.src, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(self.mask, (nw, nh),
                       interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST) > 0
        if m.any():
            small[m] = (0.45 * small[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.ox, self.oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, anchor="nw", image=self.photo)

    # -- painting ----------------------------------------------------------
    def _to_img(self, x, y):
        return (x - self.ox) / self.s, (y - self.oy) / self.s

    def _paint(self, p0, p1):
        r = max(1, int(round(self.brush.get() / self.s)))
        a = (int(p0[0]), int(p0[1]))
        b = (int(p1[0]), int(p1[1]))
        cv2.line(self.mask, a, b, 255, 2 * r)
        cv2.circle(self.mask, b, r, 255, -1)

    def on_press(self, e):
        self.undo_stack.append(self.mask.copy())
        del self.undo_stack[:-15]
        self.last = self._to_img(e.x, e.y)
        self.last_canvas = (e.x, e.y)
        self._paint(self.last, self.last)
        r = self.brush.get()
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                                fill="#ff3030", outline="", tags="stroke")

    def on_drag(self, e):
        if self.last is None:
            return
        pt = self._to_img(e.x, e.y)
        self._paint(self.last, pt)
        self.canvas.create_line(*self.last_canvas, e.x, e.y, width=self.brush.get() * 2,
                                fill="#ff3030", capstyle=tk.ROUND, tags="stroke")
        self.last, self.last_canvas = pt, (e.x, e.y)

    def on_release(self, _e):
        self.last = None
        self.render()

    def undo_stroke(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.render()

    def clear(self):
        if self.mask.any():
            self.undo_stack.append(self.mask.copy())
            self.mask[:] = 0
            self.render()

    # -- actions -----------------------------------------------------------
    def apply(self):
        if not self.mask.any():
            self.app.set_status("Paint over the area you want to remove first.")
            return
        mask = cv2.dilate(self.mask, np.ones((3, 3), np.uint8))
        flag = cv2.INPAINT_TELEA if self.method.get() == "telea" else cv2.INPAINT_NS
        radius = max(1, int(round(self.radius.get())))
        self.app.run_task("Inpainting", lambda img: cv2.inpaint(img, mask, radius, flag),
                          on_done=self._after_apply)

    def _after_apply(self):
        if not self.winfo_exists():
            return
        self.src = self.app.current_img.copy()
        self.mask[:] = 0
        self.undo_stack.clear()
        self.render()

    def close(self):
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.app.inpaint_win = None
        self.destroy()


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        root.title("Advanced Image Restoration Tool")
        root.geometry("1300x820")
        root.minsize(1100, 720)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.original_img = None
        self.current_img = None
        self.history = []
        self.history_index = -1
        self.is_processing = False
        self.ui_queue = queue.Queue()
        self.inpaint_win = None
        self._preview_job = None
        self._suspend_preview = False
        self._resize_guard = False

        # Tk variables
        self.level_var = tk.IntVar(value=2)
        self.style_var = tk.StringVar(value="natural")
        self.denoise_var = tk.DoubleVar(value=10)
        self.filter_size_var = tk.DoubleVar(value=5)
        self.psf_var = tk.DoubleVar(value=1.5)
        self.amount_var = tk.DoubleVar(value=1.0)
        self.iter_var = tk.DoubleVar(value=20)
        self.k_var = tk.DoubleVar(value=0.01)
        self.brightness_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.gamma_var = tk.DoubleVar(value=1.0)
        self.quality_var = tk.DoubleVar(value=90)
        self.width_var = tk.StringVar(value="")
        self.height_var = tk.StringVar(value="")
        self.keep_aspect = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready - open an image to begin")

        self._build_menu()
        self._build_layout()
        self._bind_keys()
        root.after(50, self._poll_queue)

    # ---- UI construction ----------------------------------------------------
    def _build_menu(self):
        bar = tk.Menu(self.root)
        self.root.config(menu=bar)
        f = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="File", menu=f)
        f.add_command(label="Open Image", command=self.open_image, accelerator="Ctrl+O")
        f.add_command(label="Save Image", command=self.save_image, accelerator="Ctrl+S")
        f.add_separator()
        f.add_command(label="Exit", command=self.root.destroy)
        e = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="Edit", menu=e)
        e.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        e.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        e.add_separator()
        e.add_command(label="Reset to Original", command=self.reset_image)
        h = tk.Menu(bar, tearoff=0)
        bar.add_cascade(label="Help", menu=h)
        h.add_command(label="About", command=self.show_about)

    def _bind_keys(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())

    def _build_layout(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))
        for text, cmd in [("Open", self.open_image), ("Save", self.save_image), (None, None),
                          ("Undo", self.undo), ("Redo", self.redo), ("Reset", self.reset_image)]:
            if text is None:
                ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
            else:
                ttk.Button(toolbar, text=text, command=cmd, width=8).pack(side=tk.LEFT, padx=2)

        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 6))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Label(status, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        panel = ttk.Frame(body, width=360)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        view = ttk.Frame(body)
        view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        view.columnconfigure(0, weight=1, uniform="p")
        view.columnconfigure(1, weight=1, uniform="p")
        view.rowconfigure(0, weight=1)
        self.pane_orig = ImagePane(view, "Original")
        self.pane_proc = ImagePane(view, "Processed")
        self.pane_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.pane_proc.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        nb = ttk.Notebook(panel)
        nb.pack(fill=tk.BOTH, expand=True)
        self._tab_auto(nb)
        self._tab_prepare(nb)
        self._tab_restore(nb)
        self._tab_adjust(nb)

    def _tab(self, nb, title):
        t = ttk.Frame(nb)
        nb.add(t, text=title)
        return t

    def _tab_auto(self, nb):
        t = self._tab(nb, "Auto")
        lf = ttk.LabelFrame(t, text="One-Click Restoration")
        lf.pack(fill=tk.X, padx=5, pady=5)
        row = ttk.Frame(lf)
        row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(row, text="Level:", width=7).pack(side=tk.LEFT)
        for text, v in [("Light", 1), ("Medium", 2), ("Heavy", 3)]:
            ttk.Radiobutton(row, text=text, variable=self.level_var, value=v).pack(side=tk.LEFT)
        row = ttk.Frame(lf)
        row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(row, text="Style:", width=7).pack(side=tk.LEFT)
        for text, v in [("Natural", "natural"), ("Vintage", "vintage"), ("B&W", "bw")]:
            ttk.Radiobutton(row, text=text, variable=self.style_var, value=v).pack(side=tk.LEFT)
        ttk.Button(lf, text="Restore", command=self.auto_restore_image).pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(lf, text="Analyze Image", command=self.show_analysis).pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Label(t, wraplength=320, foreground="#666",
                  text="Measures noise, sharpness, contrast and colour cast, then picks the "
                       "filter strengths automatically. It works on the current image, so "
                       "Undo steps back.").pack(padx=8, pady=4, anchor="w")

    def _tab_prepare(self, nb):
        t = self._tab(nb, "Prepare")
        lf = ttk.LabelFrame(t, text="Analysis")
        lf.pack(fill=tk.X, padx=5, pady=5)
        button_row(lf, ("Detect Blur", self.detect_blur), ("Detect Noise", self.detect_noise))
        lf = ttk.LabelFrame(t, text="Tone & Colour")
        lf.pack(fill=tk.X, padx=5, pady=5)
        for text, fn in [("Histogram Equalization", lambda: self.run_task("Histogram equalization", hist_equalize)),
                         ("CLAHE", lambda: self.run_task("CLAHE", clahe_filter)),
                         ("Auto White Balance", lambda: self.run_task("White balance", white_balance))]:
            ttk.Button(lf, text=text, command=fn).pack(fill=tk.X, padx=5, pady=2)

        lf = ttk.LabelFrame(t, text="Resize")
        lf.pack(fill=tk.X, padx=5, pady=5)
        row = ttk.Frame(lf)
        row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(row, text="Width").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.width_var, width=6).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(row, text="Height").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.height_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(lf, text="Keep aspect ratio", variable=self.keep_aspect).pack(anchor="w", padx=5)
        ttk.Button(lf, text="Resize", command=self.resize_image).pack(fill=tk.X, padx=5, pady=3)
        self.width_var.trace_add("write", lambda *_: self._on_size_edit("w"))
        self.height_var.trace_add("write", lambda *_: self._on_size_edit("h"))

    def _tab_restore(self, nb):
        t = self._tab(nb, "Restore")
        lf = ttk.LabelFrame(t, text="Noise Reduction")
        lf.pack(fill=tk.X, padx=5, pady=5)
        add_slider(lf, "Strength", self.denoise_var, 1, 30, "{:.0f}")
        button_row(lf, ("Non-Local Means", self.op_nlm), ("Bilateral", self.op_bilateral))
        add_slider(lf, "Filter size", self.filter_size_var, 3, 15, "{:.0f}")
        button_row(lf, ("Gaussian Blur", self.op_gaussian), ("Median Blur", self.op_median))

        lf = ttk.LabelFrame(t, text="Deblurring")
        lf.pack(fill=tk.X, padx=5, pady=5)
        add_slider(lf, "Blur radius", self.psf_var, 0.5, 6, "{:.1f}")
        add_slider(lf, "Amount", self.amount_var, 0.2, 3, "{:.1f}")
        add_slider(lf, "Iterations", self.iter_var, 5, 60, "{:.0f}")
        add_slider(lf, "Noise (K)", self.k_var, 0.001, 0.1, "{:.3f}")
        button_row(lf, ("Unsharp", self.op_unsharp), ("Wiener", self.op_wiener),
                   ("Lucy-Richardson", self.op_richardson))

        lf = ttk.LabelFrame(t, text="Smart Inpainting")
        lf.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(lf, text="Open Inpainting Tool...", command=self.open_inpaint).pack(
            fill=tk.X, padx=5, pady=4)

    def _tab_adjust(self, nb):
        t = self._tab(nb, "Adjust")
        lf = ttk.LabelFrame(t, text="Colour Correction (live preview)")
        lf.pack(fill=tk.X, padx=5, pady=5)
        add_slider(lf, "Brightness", self.brightness_var, -100, 100, "{:.0f}")
        add_slider(lf, "Contrast", self.contrast_var, 0.5, 2.0, "{:.2f}")
        add_slider(lf, "Gamma", self.gamma_var, 0.1, 3.0, "{:.2f}")
        button_row(lf, ("Apply", self.apply_adjustments), ("Reset sliders", self.reset_adjust))
        for v in (self.brightness_var, self.contrast_var, self.gamma_var):
            v.trace_add("write", self._on_adjust_change)

        lf = ttk.LabelFrame(t, text="Post-processing")
        lf.pack(fill=tk.X, padx=5, pady=5)
        button_row(lf, ("Edge Enhancement", lambda: self.run_task("Edge enhancement", edge_enhance)),
                   ("Adaptive Threshold", lambda: self.run_task("Adaptive threshold", adaptive_binarize)))

        lf = ttk.LabelFrame(t, text="Export")
        lf.pack(fill=tk.X, padx=5, pady=5)
        add_slider(lf, "JPEG quality", self.quality_var, 10, 100, "{:.0f}")
        ttk.Label(lf, text="Format is chosen by the file extension when saving.",
                  foreground="#666").pack(padx=5, pady=(0, 4), anchor="w")

    # ---- status / busy / background tasks -------------------------------------
    def set_status(self, text):
        self.status_var.set(text)

    def set_busy(self, busy):
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def run_task(self, label, func, on_done=None):
        """Run func(image)->image on a worker thread; UI is updated via a queue."""
        if self.current_img is None:
            self.set_status("Open an image first.")
            return
        if self.is_processing:
            self.set_status("Busy - wait for the current operation to finish.")
            return
        self.is_processing = True
        self.set_busy(True)
        self.set_status(f"{label}...")
        src = self.current_img.copy()

        def worker():
            try:
                self.ui_queue.put(("done", label, func(src), on_done))
            except Exception as exc:
                self.ui_queue.put(("error", label, str(exc), traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                msg = self.ui_queue.get_nowait()
                kind = msg[0]
                if kind == "status":
                    self.set_status(msg[1])
                elif kind == "done":
                    _, label, result, on_done = msg
                    self.is_processing = False
                    self.set_busy(False)
                    self.reset_adjust()
                    self.commit(result)
                    self.set_status(f"{label} - done")
                    if on_done:
                        on_done()
                elif kind == "error":
                    _, label, err, tb = msg
                    self.is_processing = False
                    self.set_busy(False)
                    print(tb, file=sys.stderr)
                    self.set_status(f"{label} failed: {err}")
                    messagebox.showerror(f"{label} failed", err)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    # ---- image state / history -------------------------------------------------
    def commit(self, img):
        self.current_img = img
        del self.history[self.history_index + 1:]
        self.history.append(img)
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)
        self.history_index = len(self.history) - 1
        self.refresh()
        self.sync_size_fields()

    def refresh(self):
        self.pane_orig.set_image(self.original_img)
        self.pane_proc.set_image(self.current_img)

    def undo(self):
        if self.is_processing or self.history_index <= 0:
            return
        self.history_index -= 1
        self._restore_from_history("Undo")

    def redo(self):
        if self.is_processing or self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        self._restore_from_history("Redo")

    def _restore_from_history(self, label):
        self.reset_adjust()
        self.current_img = self.history[self.history_index]
        self.refresh()
        self.sync_size_fields()
        self.set_status(label)

    def reset_image(self):
        if self.original_img is None or self.is_processing:
            return
        self.reset_adjust()
        self.commit(self.original_img.copy())
        self.set_status("Reset to original")

    # ---- file I/O -----------------------------------------------------------------
    def open_image(self):
        if self.is_processing:
            return
        path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = np.fromfile(path, dtype=np.uint8)  # works with non-ASCII paths on Windows
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Unsupported or corrupt image file.")
        except Exception as exc:
            self.set_status(f"Error opening image: {exc}")
            messagebox.showerror("Open failed", str(exc))
            return
        self.original_img = img
        self.current_img = img
        self.history = [img]
        self.history_index = 0
        self.reset_adjust()
        self.refresh()
        self.sync_size_fields()
        self.set_status(f"Opened {os.path.basename(path)}  ({img.shape[1]} x {img.shape[0]})")

    def save_image(self):
        img = self.pane_proc.img  # what you see is what you save (incl. live preview)
        if img is None:
            self.set_status("Nothing to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
            ("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("TIFF", "*.tif *.tiff"), ("BMP", "*.bmp")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in SAVE_EXTS:
            path += ".png"
            ext = ".png"
        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.quality_var.get())] if ext in (".jpg", ".jpeg") else []
        try:
            ok, buf = cv2.imencode(ext, img, params)
            if not ok:
                raise ValueError("Encoding failed.")
            buf.tofile(path)
            self.set_status(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            self.set_status(f"Error saving: {exc}")
            messagebox.showerror("Save failed", str(exc))

    def show_about(self):
        messagebox.showinfo("About", "Advanced Image Restoration Tool\n\n"
                            "One-click restoration, denoising, deblurring (Wiener / Richardson-Lucy),\n"
                            "inpainting, live colour adjustments, undo/redo.")

    # ---- analysis ---------------------------------------------------------------------
    def _need_image(self):
        if self.current_img is None:
            self.set_status("Open an image first.")
            return False
        return True

    def detect_blur(self):
        if not self._need_image():
            return
        s = analyze(self.current_img)["sharpness"]
        verdict = "very blurry" if s < 30 else "blurry" if s < 100 else "sharp"
        messagebox.showinfo("Blur Detection", f"Sharpness score: {s:.1f}\nImage looks {verdict}.")

    def detect_noise(self):
        if not self._need_image():
            return
        n = analyze(self.current_img)["noise"]
        verdict = "low" if n < 2 else "moderate" if n < 5 else "high"
        messagebox.showinfo("Noise Detection", f"Estimated noise sigma: {n:.2f}\nNoise level is {verdict}.")

    def show_analysis(self):
        if not self._need_image():
            return
        a = analyze(self.current_img)
        messagebox.showinfo("Image Analysis",
                            f"Noise sigma: {a['noise']:.2f}\nSharpness: {a['sharpness']:.0f}\n"
                            f"Contrast (L std): {a['contrast']:.0f}\nColour cast: {a['cast']:.1f}")

    # ---- restoration operations ---------------------------------------------------------
    def auto_restore_image(self):
        level, style = int(self.level_var.get()), self.style_var.get()

        def report(text):
            self.ui_queue.put(("status", text))
        self.run_task(f"Restoring ({style}, level {level})",
                      lambda img: auto_restore(img, level, style, report))

    def op_nlm(self):
        h = float(self.denoise_var.get())
        self.run_task("Non-local means denoising",
                      lambda img: cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21))

    def op_bilateral(self):
        sig = 30 + 4 * float(self.denoise_var.get())
        self.run_task("Bilateral filter", lambda img: cv2.bilateralFilter(img, 9, sig, sig))

    def op_gaussian(self):
        k = odd(self.filter_size_var.get())
        self.run_task("Gaussian blur", lambda img: cv2.GaussianBlur(img, (k, k), 0))

    def op_median(self):
        k = odd(self.filter_size_var.get())
        self.run_task("Median blur", lambda img: cv2.medianBlur(img, k))

    def op_unsharp(self):
        s, a = float(self.psf_var.get()), float(self.amount_var.get())
        self.run_task("Unsharp masking", lambda img: unsharp(img, s, a))

    def op_wiener(self):
        s, k = float(self.psf_var.get()), float(self.k_var.get())
        self.run_task("Wiener deconvolution", lambda img: wiener_deconvolve(img, s, k))

    def op_richardson(self):
        s, n = float(self.psf_var.get()), int(round(self.iter_var.get()))
        self.run_task("Richardson-Lucy deconvolution", lambda img: richardson_lucy(img, s, n))

    def open_inpaint(self):
        if not self._need_image() or self.is_processing:
            return
        if self.inpaint_win is not None and self.inpaint_win.winfo_exists():
            self.inpaint_win.lift()
            return
        self.inpaint_win = InpaintDialog(self)

    # ---- live colour adjustment -----------------------------------------------------------
    def _adjust_is_default(self):
        return (abs(self.brightness_var.get()) < 0.5 and abs(self.contrast_var.get() - 1) < 0.005
                and abs(self.gamma_var.get() - 1) < 0.005)

    def _on_adjust_change(self, *_):
        if self._suspend_preview or self.current_img is None:
            return
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(40, self._show_adjust_preview)

    def _lut(self):
        return build_lut(self.brightness_var.get(), self.contrast_var.get(), self.gamma_var.get())

    def _show_adjust_preview(self):
        self._preview_job = None
        if self.current_img is None:
            return
        if self._adjust_is_default():
            self.pane_proc.set_image(self.current_img)
        else:
            self.pane_proc.set_image(cv2.LUT(self.current_img, self._lut()))

    def reset_adjust(self):
        if self._preview_job:
            self.root.after_cancel(self._preview_job)
            self._preview_job = None
        self._suspend_preview = True
        self.brightness_var.set(0)
        self.contrast_var.set(1.0)
        self.gamma_var.set(1.0)
        self._suspend_preview = False
        if self.current_img is not None and self.pane_proc.img is not self.current_img:
            self.pane_proc.set_image(self.current_img)

    def apply_adjustments(self):
        if not self._need_image():
            return
        if self._adjust_is_default():
            self.set_status("Move a slider first.")
            return
        result = cv2.LUT(self.current_img, self._lut())
        self.reset_adjust()
        self.commit(result)
        self.set_status("Applied colour adjustments")

    # ---- resize ------------------------------------------------------------------------------
    def sync_size_fields(self):
        if self.current_img is None:
            return
        h, w = self.current_img.shape[:2]
        self._resize_guard = True
        self.width_var.set(str(w))
        self.height_var.set(str(h))
        self._resize_guard = False

    def _on_size_edit(self, which):
        if self._resize_guard or not self.keep_aspect.get() or self.current_img is None:
            return
        h0, w0 = self.current_img.shape[:2]
        try:
            if which == "w":
                new = max(1, round(int(self.width_var.get()) * h0 / w0))
                target = self.height_var
            else:
                new = max(1, round(int(self.height_var.get()) * w0 / h0))
                target = self.width_var
        except ValueError:
            return
        self._resize_guard = True
        target.set(str(new))
        self._resize_guard = False

    def resize_image(self):
        if not self._need_image() or self.is_processing:
            return
        try:
            w, h = int(self.width_var.get()), int(self.height_var.get())
        except ValueError:
            self.set_status("Width and height must be whole numbers.")
            return
        if not (1 <= w <= 20000 and 1 <= h <= 20000):
            self.set_status("Width and height must be between 1 and 20000.")
            return
        h0, w0 = self.current_img.shape[:2]
        interp = cv2.INTER_AREA if w * h < w0 * h0 else cv2.INTER_CUBIC
        result = cv2.resize(self.current_img, (w, h), interpolation=interp)
        self.reset_adjust()
        self.commit(result)
        self.set_status(f"Resized to {w} x {h}")


def main():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on HiDPI screens
        except Exception:
            pass
    root = tk.Tk()
    ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()