
"""
PTPF Lite v6.5 (HPQ + SOSA + S-FSO) — simplified, single-file module
====================================================================

Design goal:
- Keep the *same look* you approved (HPQ + palette64(purple) + half-block + SOSA A/B auto),
- but make the implementation simpler + faster + easier to reason about.

What changed vs the "long" experimental versions:
- No Floyd–Steinberg (FS). Replaced by S‑FSO (structure‑aware low‑entropy hole jitter).
- Fast quantization via precomputed bucket LUT (Hue+Sat+Val+Luma), with boundary-aware fallback.
- SOSA controls A/B and also controls S‑FSO amplitude (structure region vs fill region).

Public API:
- ptpf_render_hpq_sosa(image, cols=240, mode="auto", return_preview=True)
    -> preview PIL image (pixel-domain, already 2x rows for half-block)
- ptpf_render_ansi_hpq_sosa(image, cols=240, mode="auto")
    -> ANSI string using █ ▀ ▄ (24-bit truecolor).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageFilter
# =========================
# 全局缓存（避免重复建 LUT/调色盘）
# =========================

_GLOBAL_PAL64_PURPLE = None
_GLOBAL_LUT_CACHE = {}
def _retina_refine(rgb01: np.ndarray, E01: np.ndarray, strength: float = 0.35, blur_k: int = 5, gamma: float = 1.0) -> np.ndarray:
    """Edge-weighted highpass refinement (machine retina). Cheap & stable.
    E01: edge strength map in [0..1]
    """
    if not strength or strength <= 0:
        return rgb01
    w = np.clip(E01, 0, 1) ** max(gamma, 1e-6)
    base = _box_blur_float01(rgb01, k=blur_k)
    high = rgb01 - base
    return np.clip(rgb01 + strength * w[..., None] * high, 0, 1)



def ptpf_get_palette64_purple_cached() -> np.ndarray:
    """返回固定的 64 色紫盘（全局只构造一次）。"""
    global _GLOBAL_PAL64_PURPLE
    if _GLOBAL_PAL64_PURPLE is None:
        _GLOBAL_PAL64_PURPLE = ptpf_palette64_cartoon_with_purple()
    return _GLOBAL_PAL64_PURPLE

def ptpf_get_hsv_bucket_lut_cached(pal_u8: np.ndarray, cfg: "Config") -> "HSVBucketLUT":
    """HSV 分桶 LUT 全局缓存（palette + bins/权重/边界参数为 key）。"""
    pal_key = pal_u8.tobytes()
    key = (
        pal_key,
        int(cfg.h_bins), int(cfg.s_bins), int(cfg.v_bins), int(cfg.l_bins),
        float(cfg.w_h), float(cfg.w_s), float(cfg.w_v), float(cfg.w_l),
        float(cfg.hue_boundary_margin),
    )
    lut = _GLOBAL_LUT_CACHE.get(key)
    if lut is None:
        lut = HSVBucketLUT(pal_u8, cfg)
        _GLOBAL_LUT_CACHE[key] = lut
    return lut


# -------------------------
# Config
# -------------------------

@dataclass
class PTPFConfig:
    cols: int = 240
    char_aspect: float = 2.0          # terminal character height/width ratio
    blur_k: int = 3
    unsharp_amount: float = 0.50
    sat_k: float = 1.25
    gray_mix: float = 0.15            # stabilize highlights without killing saturation
    gray_gamma: float = 0.95

    # SOSA (structure map)
    sosa_edge_gain: float = 1.0
    sosa_thresh: float = 0.42         # mean edge energy threshold for auto A/B

    # S‑FSO (hole jitter), amplitudes in RGB space
    # A: conservative, B: more fill-region smoothing
    hole_amp_A: float = 0.018
    hole_amp_B: float = 0.030
    hole_amp_structure_scale: float = 0.25  # structure region uses amp * scale

    # Quantization distance weights (HSV)
    w_h: float = 3.0
    w_s: float = 1.0
    w_v: float = 1.0
    w_l: float = 0.60                 # extra luma term to reduce "wrong bucket" banding

    
    # Machine retina (optional, default OFF)
    retina_enabled: bool = False
    retina_strength: float = 0.35        # 0.2–0.6
    retina_blur_k: int = 5               # odd-ish; larger = softer highpass
    retina_gamma: float = 1.0            # edge exponent for weighting

# LUT bucketing resolution (trade speed vs quality)
    h_bins: int = 48
    s_bins: int = 12
    v_bins: int = 12
    l_bins: int = 12

    # boundary fallback: if close to hue boundary, also test neighbor hue-bins
    hue_boundary_margin: float = 0.015  # in [0..1] hue space


# -------------------------
# Utilities: resize to half-block pixel domain
# -------------------------

def resize_for_halfblock(im: Image.Image, cols: int, char_aspect: float) -> Image.Image:
    """
    Resize into half-block pixel domain:
    - width = cols (terminal columns)
    - height = 2 * rows (each char cell = 2 vertical pixels)
    """
    W, H = im.size
    rows = max(2, int(round(H / W * cols / char_aspect)))
    return im.resize((cols, rows * 2), resample=Image.Resampling.BICUBIC)


def _box_blur_float01(rgb01: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return rgb01
    r = max(1, k // 2)
    im = Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8))
    im = im.filter(ImageFilter.BoxBlur(r))
    return np.asarray(im).astype(np.float32) / 255.0


def _unsharp_like(rgb01: np.ndarray, amount: float) -> np.ndarray:
    blur = _box_blur_float01(rgb01, 3)
    return np.clip(rgb01 + amount * (rgb01 - blur), 0, 1)


def _boost_saturation_rgb(rgb01: np.ndarray, k: float) -> np.ndarray:
    mean = rgb01.mean(axis=-1, keepdims=True)
    return np.clip(mean + k * (rgb01 - mean), 0, 1)


def _to_gray_gamma(rgb01: np.ndarray, gamma: float) -> np.ndarray:
    gray = 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]
    gray = np.clip(gray, 0, 1)
    if gamma != 1.0:
        gray = np.clip(gray ** gamma, 0, 1)
    return gray


# -------------------------
# Palette (64 colors with purple anchors)
# -------------------------

def palette64_cartoon_with_purple() -> np.ndarray:
    skin  = [(255,222,207),(250,205,185),(240,190,170),(225,175,160)]
    hair  = [(40,32,24),(64,48,40),(96,72,64),(140,100,80)]
    vivid = [(240,90,90),(255,120,120),(255,150,150),(255,200,200)]
    orange= [(255,150,80),(255,180,110),(255,200,140),(255,220,170)]
    blue  = [(90,120,210),(110,145,225),(130,165,240),(165,190,255)]
    green = [(90,165,120),(110,185,140),(130,205,160),(160,225,185)]
    purple= [(82,62,130),(120,80,170),(160,90,200),(190,120,230)]
    misc  = [(240,240,200),(225,225,185),(210,210,210),(185,185,225)]
    rows = skin + hair + vivid + orange + blue + green + purple + misc
    while len(rows) < 64:
        rows.append(rows[-1])
    return np.array(rows[:64], dtype=np.uint8)


# -------------------------
# Color space & SOSA
# -------------------------

def rgb_to_hsv_np(rgb01: np.ndarray) -> np.ndarray:
    """Vectorized RGB->HSV, rgb01 shape (...,3), returns HSV in [0..1]."""
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    cmax = np.max(rgb01, axis=-1)
    cmin = np.min(rgb01, axis=-1)
    delta = cmax - cmin + 1e-8

    h = np.zeros_like(cmax)

    m = (cmax == r)
    h[m] = ((g - b)[m] / delta[m]) % 6.0
    m = (cmax == g)
    h[m] = (b - r)[m] / delta[m] + 2.0
    m = (cmax == b)
    h[m] = (r - g)[m] / delta[m] + 4.0

    h = (h / 6.0) % 1.0
    s = delta / (cmax + 1e-8)
    v = cmax
    return np.clip(np.stack([h, s, v], axis=-1), 0, 1)


def sosa_edge_map(rgb01: np.ndarray) -> np.ndarray:
    """
    Simple Sobel edge magnitude on luminance.
    Returns E in [0..1], same HxW.
    """
    L = 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]
    # sobel
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)

    # cheap conv via padding + manual sums (fast enough for terminal sizes)
    P = np.pad(L, ((1,1),(1,1)), mode="edge")
    gx = (
        Kx[0,0]*P[:-2,:-2] + Kx[0,1]*P[:-2,1:-1] + Kx[0,2]*P[:-2,2:] +
        Kx[1,0]*P[1:-1,:-2] + Kx[1,1]*P[1:-1,1:-1] + Kx[1,2]*P[1:-1,2:] +
        Kx[2,0]*P[2:,:-2] + Kx[2,1]*P[2:,1:-1] + Kx[2,2]*P[2:,2:]
    )
    gy = (
        Ky[0,0]*P[:-2,:-2] + Ky[0,1]*P[:-2,1:-1] + Ky[0,2]*P[:-2,2:] +
        Ky[1,0]*P[1:-1,:-2] + Ky[1,1]*P[1:-1,1:-1] + Ky[1,2]*P[1:-1,2:] +
        Ky[2,0]*P[2:,:-2] + Ky[2,1]*P[2:,1:-1] + Ky[2,2]*P[2:,2:]
    )

    mag = np.sqrt(gx*gx + gy*gy)
    # normalize robustly
    p95 = np.percentile(mag, 95) + 1e-8
    E = np.clip(mag / p95, 0, 1)
    return E.astype(np.float32)


# -------------------------
# S‑FSO: low-entropy holes (structure-aware jitter)
# -------------------------

def _low_discrepancy_noise(h: int, w: int) -> np.ndarray:
    """
    Deterministic "blue-noise-ish" scalar in [0..1] per pixel using golden ratio hashing.
    Low entropy: no RNG, stable across runs.
    """
    # (x*phi + y*pi) fractional part
    phi = 0.6180339887498949
    pi  = 3.141592653589793
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    v = (xs * phi + ys * pi)
    return (v - np.floor(v)).astype(np.float32)


def s_fso_hole_jitter(rgb01: np.ndarray, E: np.ndarray, amp_fill: float, amp_structure: float) -> np.ndarray:
    """
    Apply small RGB-space jitter (holes) to break banding without FS dithering.
    - E high -> structure region -> smaller amp
    - E low -> fill region -> larger amp
    """
    h, w, _ = rgb01.shape
    n = _low_discrepancy_noise(h, w)  # [0..1]
    # centered perturb in [-1,1]
    p = (n * 2.0 - 1.0)

    # blend amplitude by structure map
    amp = amp_structure * E + amp_fill * (1.0 - E)
    amp = amp[..., None]

    # 3-channel "polarization": rotate perturb per channel to avoid pure gray shifts
    pr = p
    pg = (p + 0.3333) - np.floor(p + 0.3333)
    pb = (p + 0.6666) - np.floor(p + 0.6666)
    pg = pg * 2.0 - 1.0
    pb = pb * 2.0 - 1.0

    jitter = np.stack([pr, pg, pb], axis=-1) * amp
    return np.clip(rgb01 + jitter, 0, 1)


# -------------------------
# Fast HSV quantization with buckets (Hue+Sat+Val+Luma)
# -------------------------

class HSVBucketLUT:
    def __init__(self, pal_u8: np.ndarray, cfg: PTPFConfig):
        self.cfg = cfg
        self.pal_u8 = pal_u8.astype(np.uint8)
        self.pal_rgb = self.pal_u8.astype(np.float32) / 255.0
        self.pal_hsv = rgb_to_hsv_np(self.pal_rgb.reshape(-1, 3))
        self.pal_l   = (0.2126*self.pal_rgb[:,0] + 0.7152*self.pal_rgb[:,1] + 0.0722*self.pal_rgb[:,2]).astype(np.float32)

        # build LUT: [h_bin, s_bin, v_bin, l_bin] -> best palette index
        hb, sb, vb, lb = cfg.h_bins, cfg.s_bins, cfg.v_bins, cfg.l_bins
        self.lut = np.zeros((hb, sb, vb, lb), dtype=np.uint8)
        self._build()

    def _build(self):
        cfg = self.cfg
        hb, sb, vb, lb = cfg.h_bins, cfg.s_bins, cfg.v_bins, cfg.l_bins

        # palette values for distance
        ph = self.pal_hsv[:, 0]
        ps = self.pal_hsv[:, 1]
        pv = self.pal_hsv[:, 2]
        pl = self.pal_l

        for ih in range(hb):
            h0 = (ih + 0.5) / hb
            # hue circular delta
            dh = np.minimum(np.abs(h0 - ph), 1.0 - np.abs(h0 - ph))

            for is_ in range(sb):
                s0 = (is_ + 0.5) / sb
                ds = np.abs(s0 - ps)

                for iv in range(vb):
                    v0 = (iv + 0.5) / vb
                    dv = np.abs(v0 - pv)

                    for il in range(lb):
                        l0 = (il + 0.5) / lb
                        dl = np.abs(l0 - pl)

                        d2 = (cfg.w_h*dh)**2 + (cfg.w_s*ds)**2 + (cfg.w_v*dv)**2 + (cfg.w_l*dl)**2
                        self.lut[ih, is_, iv, il] = int(np.argmin(d2))

    def _bucket_indices(self, hsv: np.ndarray, luma: np.ndarray):
        cfg = self.cfg
        h = hsv[..., 0]; s = hsv[..., 1]; v = hsv[..., 2]
        ih = np.clip((h * cfg.h_bins).astype(np.int32), 0, cfg.h_bins - 1)
        is_ = np.clip((s * cfg.s_bins).astype(np.int32), 0, cfg.s_bins - 1)
        iv = np.clip((v * cfg.v_bins).astype(np.int32), 0, cfg.v_bins - 1)
        il = np.clip((luma * cfg.l_bins).astype(np.int32), 0, cfg.l_bins - 1)
        return ih, is_, iv, il

    def quantize(self, rgb01: np.ndarray) -> np.ndarray:
        """
        Fast quantization:
        - LUT lookup with (H,S,V,L)
        - boundary-aware: if hue is near boundary, also test neighbor hue-bins with exact distance.
        """
        cfg = self.cfg
        h, w, _ = rgb01.shape
        hsv = rgb_to_hsv_np(rgb01.reshape(-1, 3)).reshape(h, w, 3)
        luma = (0.2126*rgb01[...,0] + 0.7152*rgb01[...,1] + 0.0722*rgb01[...,2]).astype(np.float32)

        ih, is_, iv, il = self._bucket_indices(hsv, luma)
        idx = self.lut[ih, is_, iv, il].astype(np.int32)

        # optional boundary fix: only on pixels near hue-bin borders
        if cfg.hue_boundary_margin > 0:
            # distance to nearest bin border
            hpos = hsv[..., 0] * cfg.h_bins
            frac = np.abs(hpos - np.round(hpos))
            mask = frac < (cfg.hue_boundary_margin * cfg.h_bins)
            if np.any(mask):
                # evaluate exact distance for current & neighbor hue bins
                ph = self.pal_hsv[:, 0]; ps = self.pal_hsv[:, 1]; pv = self.pal_hsv[:, 2]; pl = self.pal_l
                # gather pixel hsv/luma
                hm = hsv[...,0][mask]; sm = hsv[...,1][mask]; vm = hsv[...,2][mask]; lm = luma[mask]
                # current & neighbor bin centers
                ihm = ih[mask]
                cand_bins = np.stack([ihm, (ihm-1)%cfg.h_bins, (ihm+1)%cfg.h_bins], axis=1)  # (n,3)
                best = idx[mask].copy()
                best_d2 = np.full(best.shape, 1e9, dtype=np.float32)

                for j in range(3):
                    h0 = (cand_bins[:,j] + 0.5) / cfg.h_bins
                    dh = np.minimum(np.abs(h0[:,None]-ph[None,:]), 1.0-np.abs(h0[:,None]-ph[None,:]))
                    ds = np.abs(sm[:,None]-ps[None,:])
                    dv = np.abs(vm[:,None]-pv[None,:])
                    dl = np.abs(lm[:,None]-pl[None,:])
                    d2 = (cfg.w_h*dh)**2 + (cfg.w_s*ds)**2 + (cfg.w_v*dv)**2 + (cfg.w_l*dl)**2
                    k = np.argmin(d2, axis=1).astype(np.int32)
                    dmin = d2[np.arange(d2.shape[0]), k]
                    upd = dmin < best_d2
                    best[upd] = k[upd]
                    best_d2[upd] = dmin[upd]

                idx[mask] = best

        out = self.pal_u8[idx.reshape(-1)].reshape(h, w, 3)
        return out.astype(np.uint8)


# -------------------------
# Half-block encode & previews
# -------------------------

def halfblock_char_map(top_u8: np.ndarray, bot_u8: np.ndarray, luma_thresh: float = 0.08):
    """
    Decide █ / ▀ / ▄ for each cell.
    Returns:
      chars: (rows, cols) unicode array
      fg:    (rows, cols, 3) uint8
      bg:    (rows, cols, 3) uint8
    """
    t = top_u8.astype(np.float32)/255.0
    b = bot_u8.astype(np.float32)/255.0
    lt = 0.2126*t[...,0] + 0.7152*t[...,1] + 0.0722*t[...,2]
    lb = 0.2126*b[...,0] + 0.7152*b[...,1] + 0.0722*b[...,2]

    d = np.abs(lt - lb)

    chars = np.empty(d.shape, dtype=object)
    fg = np.zeros((*d.shape, 3), dtype=np.uint8)
    bg = np.zeros((*d.shape, 3), dtype=np.uint8)

    same = d < luma_thresh
    up = lt > lb

    chars[same] = "█"
    fg[same] = top_u8[same]
    bg[same] = top_u8[same]

    chars[~same & up] = "▀"
    fg[~same & up] = top_u8[~same & up]
    bg[~same & up] = bot_u8[~same & up]

    chars[~same & ~up] = "▄"
    fg[~same & ~up] = bot_u8[~same & ~up]
    bg[~same & ~up] = top_u8[~same & ~up]

    return chars, fg, bg


def render_preview_from_halfblock(top_u8: np.ndarray, bot_u8: np.ndarray, scale: int = 3) -> Image.Image:
    """
    Smooth "what terminal would show" preview:
    Simply stacks top/bot into pixel domain then upscales with NEAREST (keeps block feel).
    """
    h, w, _ = top_u8.shape
    img = np.zeros((h*2, w, 3), dtype=np.uint8)
    img[0::2] = top_u8
    img[1::2] = bot_u8
    im = Image.fromarray(img)
    if scale != 1:
        im = im.resize((w*scale, h*2*scale), resample=Image.Resampling.NEAREST)
    return im


def ansi_truecolor(fg, bg, ch) -> str:
    return f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m{ch}"


def render_ansi(chars, fg, bg) -> str:
    rows, cols = chars.shape
    out = []
    last_fg = None
    last_bg = None
    for y in range(rows):
        for x in range(cols):
            f = fg[y, x]; b = bg[y, x]; ch = chars[y, x]
            if last_fg is None or (f != last_fg).any():
                out.append(f"\x1b[38;2;{int(f[0])};{int(f[1])};{int(f[2])}m")
                last_fg = f.copy()
            if last_bg is None or (b != last_bg).any():
                out.append(f"\x1b[48;2;{int(b[0])};{int(b[1])};{int(b[2])}m")
                last_bg = b.copy()
            out.append(ch)
        out.append("\x1b[0m\n")
        last_fg = None
        last_bg = None
    return "".join(out)


# -------------------------
# Main pipeline
# -------------------------

def _hpq_preprocess(rgb01: np.ndarray, cfg: PTPFConfig) -> np.ndarray:
    wet = _box_blur_float01(rgb01, cfg.blur_k)
    wet = _unsharp_like(wet, cfg.unsharp_amount)
    wet = _boost_saturation_rgb(wet, cfg.sat_k)
    if cfg.gray_mix > 0:
        gray = _to_gray_gamma(wet, cfg.gray_gamma)
        wet = np.clip(wet * (1.0 - cfg.gray_mix) + gray[..., None] * cfg.gray_mix, 0, 1)
    return wet


def ptpf_render_hpq_sosa(
    image: Image.Image,
    cols: int = 240,
    mode: str = "auto",
    return_preview: bool = True,
    preview_scale: int = 3,
    cfg: PTPFConfig | None = None,
):
    """
    Render pipeline producing half-block colored result.

    mode:
      - "A": conservative holes
      - "B": stronger fill-region holes
      - "auto": choose by SOSA (edge density)

    Returns:
      - if return_preview: (preview_image, meta_dict)
      - else: (top_u8, bot_u8, meta_dict)
    """
    cfg = cfg or PTPFConfig(cols=cols)
    im = resize_for_halfblock(image.convert("RGB"), cols=cfg.cols, char_aspect=cfg.char_aspect)
    rgb01 = np.asarray(im).astype(np.float32) / 255.0

    # HPQ base
    base = _hpq_preprocess(rgb01, cfg)

    # SOSA edge map on base
    E = sosa_edge_map(base) * cfg.sosa_edge_gain
    E = np.clip(E, 0, 1)
    if getattr(cfg, 'retina_enabled', False):
        base = _retina_refine(base, E, strength=cfg.retina_strength, blur_k=cfg.retina_blur_k, gamma=cfg.retina_gamma)
        E = sosa_edge_map(base) * cfg.sosa_edge_gain
        E = np.clip(E, 0, 1)
    edge_mean = float(E.mean())

    if mode == "auto":
        mode_sel = "B" if edge_mean >= cfg.sosa_thresh else "A"
    else:
        mode_sel = mode.upper()

    # S‑FSO holes (no FS)
    amp_fill = cfg.hole_amp_B if mode_sel == "B" else cfg.hole_amp_A
    amp_structure = amp_fill * cfg.hole_amp_structure_scale
    jittered = s_fso_hole_jitter(base, E, amp_fill=amp_fill, amp_structure=amp_structure)

    # Quantization (fast)
    pal = palette64_cartoon_with_purple()
    lut = ptpf_get_hsv_bucket_lut_cached(pal, cfg)
    q = lut.quantize(jittered)  # uint8 HxWx3

    # Half-block split
    top_u8 = q[0::2]
    bot_u8 = q[1::2]

    meta = {
        "mode": mode_sel,
        "cols": cfg.cols,
        "rows": int(q.shape[0] // 2),
        "edge_mean": edge_mean,
        "cfg": cfg,
    }

    if return_preview:
        preview = render_preview_from_halfblock(top_u8, bot_u8, scale=preview_scale)
        return preview, meta

    return top_u8, bot_u8, meta


def ptpf_render_ansi_hpq_sosa(
    image: Image.Image,
    cols: int = 240,
    mode: str = "auto",
    cfg: PTPFConfig | None = None,
) -> str:
    """
    Return ANSI string (24-bit) using █ ▀ ▄.
    """
    cfg = cfg or PTPFConfig(cols=cols)
    top_u8, bot_u8, meta = ptpf_render_hpq_sosa(image, cols=cols, mode=mode, return_preview=False, cfg=cfg)
    chars, fg, bg = halfblock_char_map(top_u8, bot_u8)
    return render_ansi(chars, fg, bg)


# -------------------------
# CLI demo
# -------------------------

def _demo():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input image path")
    ap.add_argument("--cols", type=int, default=240)
    ap.add_argument("--mode", type=str, default="auto", choices=["auto","A","B"])
    ap.add_argument("--out", type=str, default="ptpf_preview.png")
    args = ap.parse_args()

    im = Image.open(args.image).convert("RGB")
    preview, meta = ptpf_render_hpq_sosa(im, cols=args.cols, mode=args.mode, return_preview=True)
    preview.save(args.out)
    print("saved:", args.out)
    print("meta:", {k: v for k, v in meta.items() if k != "cfg"})

if __name__ == "__main__":
    _demo()