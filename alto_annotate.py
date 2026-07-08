#!/usr/bin/env python3
"""
alto_annotate.py — Stamp alto trombone slide positions onto trombone sheet music.

Takes one or more images of standard (bass clef) trombone sheet music, detects
the noteheads, works out each note's pitch from its staff position, the key
signature you supply, AND any sharp/flat/natural accidentals printed in the
music (scoped to their measure via barline detection), then writes an
annotated PDF with the slide position printed under every note.

Three annotation methods (matching the three charts):
  octave  (default)  Same key, played one octave up — positions for note + 8va.
  pitch              Play at written pitch — standard Eb alto positions.
  fourth             Read as written with tenor positions — sounds a 4th up.

Usage:
  python alto_annotate.py score.png                          # octave method, C major
  python alto_annotate.py score.jpg -m octave -k Eb          # specify key
  python alto_annotate.py p1.png p2.png -k F -o out.pdf      # multi-page PDF
  python alto_annotate.py scan.png -k Eb --debug             # save detection overlay

Notes & limitations (lightweight OpenCV approach):
  * Bass clef is assumed. The key signature is read from the image by
    default (the printed flats/sharps after each clef, majority-voted
    across staves). If it is read incorrectly, override it with -k/--key
    (major keys: C F Bb Eb Ab Db Gb G D A E B F#). Detection is per page,
    so mid-piece key changes are not handled.
  * Accidentals (sharp / flat / natural) in front of notes are detected and
    applied for the rest of their measure, like a human reader would. Glyphs
    that can't be classified confidently are flagged with an orange '?' so
    you can fill the position in by eye. Double sharps/flats, courtesy
    accidentals in parentheses, and ties carrying an accidental across a
    barline are NOT handled. Use --no-accidentals to fall back to
    key-signature-only reading.
  * Works best on clean scans. Phone photos are deskewed and threshold-
    adapted automatically, but page curvature will still bend staff lines
    and shift detected pitches — flatten the page or use a scanning app.
  * Chord symbols, lyrics and rests can occasionally be mistaken for notes
    (or faint noteheads missed). Run with --debug to see exactly what was
    detected, and adjust --ledger-range / --sensitivity if needed.
  * Only annotate music you have the rights to.

Dependencies:  pip install opencv-python numpy Pillow
"""

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Position tables (MIDI note number -> primary slide position)
# --------------------------------------------------------------------------

# Eb alto trombone, sounding pitch. A2 (midi 45) is the lowest non-pedal note.
ALTO = {
    45: "7", 46: "6", 47: "5", 48: "4", 49: "3", 50: "2", 51: "1",   # A2–Eb3
    52: "7", 53: "6", 54: "5", 55: "4", 56: "3", 57: "2", 58: "1",   # E3–Bb3
    59: "5", 60: "4", 61: "3", 62: "2", 63: "1",                     # B3–Eb4
    64: "4", 65: "3", 66: "2", 67: "1",                              # E4–G4
    68: "3", 69: "2", 70: "1",                                       # Ab4–Bb4
    71: "5", 72: "4", 73: "3", 74: "2", 75: "1",                     # B4–Eb5
    76: "2", 77: "1", 78: "2", 79: "1",                              # E5–G5
}

# Tenor (Bb) trombone positions for the written note — used by the
# up-a-fourth method, where written notes are read with tenor positions.
TENOR = {
    40: "7", 41: "6", 42: "5", 43: "4", 44: "3", 45: "2", 46: "1",   # E2–Bb2
    47: "7", 48: "6", 49: "5", 50: "4", 51: "3", 52: "2", 53: "1",   # B2–F3
    54: "5", 55: "4", 56: "3", 57: "2", 58: "1",                     # Gb3–Bb3
    59: "4", 60: "3", 61: "2", 62: "1",                              # B3–D4
    63: "3", 64: "2", 65: "1",                                       # Eb4–F4
    66: "5", 67: "4", 68: "3", 69: "2", 70: "1",                     # Gb4–Bb4
    71: "2", 72: "1", 73: "2", 74: "1",                              # B4–D5
}

METHOD_INFO = {
    "octave": ("Octave-up method: same key, sounds one octave above written", (10, 105, 60)),
    "pitch":  ("Written-pitch method: standard Eb alto positions",            (35, 55, 140)),
    "fourth": ("Up-a-fourth method: tenor positions, sounds a 4th higher",    (140, 40, 60)),
}
UNKNOWN_COLOR = (230, 130, 0)

# Major key signatures: letter -> semitone alteration
FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]
SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
KEYS = {"C": 0, "F": -1, "BB": -2, "EB": -3, "AB": -4, "DB": -5, "GB": -6,
        "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "FS": 6}

# Where key-signature glyphs sit in bass clef, as diatonic steps above the
# bottom staff line (G2 = 0), and the key each accidental count spells.
FLAT_STEPS  = [2, 5, 1, 4, 0, 3]        # Bb2 Eb3 Ab2 Db3 Gb2 Cb3
SHARP_STEPS = [6, 3, 7, 4, 1, 5]        # F#3 C#3 G#3 D#3 A#2 E#3
FLAT_KEYS   = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
SHARP_KEYS  = ["C", "G", "D", "A", "E", "B", "F#"]

SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
LETTER_SEQ = ["G", "A", "B", "C", "D", "E", "F"]  # diatonic steps up from G
ACC_ALTER = {"flat": -1, "sharp": +1, "natural": 0}
ACC_MARK = {-1: "b", 0: "", +1: "#"}


def key_accidentals(key):
    k = KEYS.get(key.strip().upper().replace("♭", "B").replace("♯", "#"))
    if k is None:
        sys.exit(f"Unknown key '{key}'. Use one of: C F Bb Eb Ab Db Gb G D A E B F#")
    if k < 0:
        return {ltr: -1 for ltr in FLAT_ORDER[:-k]}
    return {ltr: +1 for ltr in SHARP_ORDER[:k]}


def step_to_letter(step):
    """Diatonic step above the bass-clef bottom line (G2 = 0) -> (letter, octave)."""
    return LETTER_SEQ[step % 7], 2 + (step + 4) // 7


def midi_of(letter, octave, alter):
    return 12 * (octave + 1) + SEMITONE[letter] + alter


def position_for(midi, method):
    if method == "pitch":
        return ALTO.get(midi)
    if method == "octave":
        return ALTO.get(midi + 12)
    return TENOR.get(midi)


# --------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------

def deskew(gray, color):
    """Estimate global tilt from long near-horizontal lines and rotate it out."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=150,
                            minLineLength=gray.shape[1] // 3, maxLineGap=20)
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 10:
                angles.append(a)
    if not angles:
        return gray, color, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.15:
        return gray, color, 0.0
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    color = cv2.warpAffine(color, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return gray, color, angle


def binarize(gray):
    """Adaptive threshold (ink -> 255) to cope with photos' uneven lighting."""
    block = max(31, (min(gray.shape) // 24) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 12)


def find_staves(bw):
    """Return list of staves: dict(lines, top, bottom, space, xstart)."""
    w = bw.shape[1]
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 10), 1)))
    rowsum = horiz.sum(axis=1) / 255.0
    is_line = rowsum > 0.25 * w

    centres = []
    y = 0
    while y < len(is_line):
        if is_line[y]:
            y0 = y
            while y < len(is_line) and is_line[y]:
                y += 1
            centres.append((y0 + y - 1) / 2.0)
        else:
            y += 1

    staves = []
    i = 0
    while i + 4 < len(centres):
        five = centres[i:i + 5]
        gaps = np.diff(five)
        if gaps.min() > 3 and gaps.max() < 1.35 * gaps.min():
            xs = np.where(horiz[int(five[2])] > 0)[0]
            staves.append({"lines": five, "top": five[0], "bottom": five[4],
                           "space": float(np.mean(gaps)),
                           "xstart": int(xs.min()) if len(xs) else 0})
            i += 5
        else:
            i += 1
    return staves, horiz


def fill_small_holes(bw, max_area, bridge=0):
    """Fill enclosed holes (hollow half/whole noteheads) below max_area.
    With bridge > 0, outlines broken by up-to-bridge-sized gaps (a common
    casualty of staff line removal) still count as enclosing."""
    base = bw
    if bridge:
        k = bridge | 1
        base = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    inv = cv2.bitwise_not(base)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = bw.copy()
    h, w = base.shape
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + ww >= w or y + hh >= h
        if not touches_border and area < max_area:
            out[lab == i] = 255
    return out


def clean_staff_roi(bw, staff, ledger_range):
    """Strip staff+ledger lines from this staff's region and heal the cuts.
    Returns (healed_roi, y0). The healed image keeps hollow noteheads and
    accidental glyphs intact (holes NOT filled) for later analysis."""
    ss = staff["space"]
    h = bw.shape[0]
    y0 = max(0, int(staff["top"] - (ledger_range + 1.2) * ss))
    y1 = min(h, int(staff["bottom"] + (ledger_range + 1.2) * ss))

    # Removing ledger lines too means the healing close can't weld a notehead
    # to the ledger line above/below it, which corrupts pitch and shape.
    all_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(1.5 * ss)), 1)))
    lines_fat = cv2.dilate(all_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    clean = cv2.bitwise_and(bw, cv2.bitwise_not(lines_fat))
    base = clean[y0:y1].copy()
    roi = cv2.morphologyEx(base, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(0.8 * ss)))))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    light = cv2.morphologyEx(base, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(4, int(0.65 * ss)))))
    return roi, light, y0


def ink_run(img, y, x, axis):
    """Length of the contiguous ink run through (y, x) along a row or column."""
    h, w = img.shape
    best = 0
    for d in (-1, 0, 1):
        if axis == 1:
            yy = min(max(y + d, 0), h - 1)
            line = img[yy, :]
            p = x
        else:
            xx = min(max(x + d, 0), w - 1)
            line = img[:, xx]
            p = y
        if p < 0 or p >= len(line) or not line[p]:
            continue
        a = p
        while a > 0 and line[a - 1]:
            a -= 1
        b = p
        while b < len(line) - 1 and line[b + 1]:
            b += 1
        best = max(best, b - a + 1)
    return best


def upward_run(healed, ss, cx, cy_local):
    """Longest contiguous ink run straight up from a point, over 3 columns."""
    h, w = healed.shape
    best = 0
    for dx in (-int(0.35 * ss), 0, int(0.35 * ss)):
        x = int(cx) + dx
        if not (0 <= x < w):
            continue
        run, y = 0, int(cy_local)
        while y >= 0 and healed[y, x]:
            run += 1
            y -= 1
        best = max(best, run)
    return best


def detect_noteheads(healed, light, y0, staff, ledger_range, sensitivity, skip_left):
    """Return [(cx, cy)] of notehead centres from a healed staff ROI."""
    ss = staff["space"]
    filled = fill_small_holes(healed, max_area=1.6 * ss * ss)

    # Erode with a notehead-sized ellipse: stems, beams, barlines and text
    # strokes vanish, noteheads shrink to compact cores. Deliberately NO
    # dilation afterwards — re-dilating can weld a notehead core back onto a
    # nearby eighth-note flag core, producing an oversized blob.
    kw = max(3, int(0.85 * ss * sensitivity))
    kh = max(3, int(0.60 * ss * sensitivity))
    eroded = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh)))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(eroded, 8)
    heads = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        est_w, est_h = ww + kw - 1, hh + kh - 1          # pre-erosion size estimate
        if area < 2 or not (0.55 * ss < est_w < 2.6 * ss and 0.45 * ss < est_h < 4.5 * ss):
            continue
        if est_h < 1.75 * ss:
            cx, cy = cent[i][0], cent[i][1] + y0
            tall = False
        else:
            # tall core: locate the notehead lobe among the thin slab rows
            sub = (lab[y:y + hh, x:x + ww] > 0) & (lab[y:y + hh, x:x + ww] == i)
            widths = sub.sum(axis=1)
            fat = widths >= max(3, 0.28 * ss)
            bands = []
            r = 0
            while r < hh:
                if fat[r]:
                    r0 = r
                    while r < hh and fat[r]:
                        r += 1
                    bands.append((widths[r0:r].sum(), r0, r))
                else:
                    r += 1
            candidate = None
            for _, r0, r1 in sorted(bands, reverse=True):
                if not (0.45 * ss < (r1 - r0) + kh - 1 < 1.75 * ss):
                    continue
                ys, xs = np.nonzero(sub[r0:r1])
                bcx, bcy = x + xs.mean(), y + r0 + ys.mean()
                if ink_run(filled, int(bcy), int(bcx), axis=1) >= 0.95 * ss:
                    candidate = (bcx, bcy + y0)
                    break
            if candidate is None:
                # uniform slab (flag welded to head): the notehead is the END
                # that is solid in the lightly-healed image; a flag end is a
                # sparse curl plus stem
                ends = []
                for r0, r1 in ((0, min(hh, int(1.0 * ss))), (max(0, hh - int(1.0 * ss)), hh)):
                    ys, xs = np.nonzero(sub[r0:r1])
                    if len(xs) == 0:
                        continue
                    bcx, bcy = x + xs.mean(), y + r0 + ys.mean()
                    by0, by1 = int(bcy - 0.5 * ss), int(bcy + 0.5 * ss)
                    bx0, bx1 = int(bcx - 0.65 * ss), int(bcx + 0.65 * ss)
                    box = light[max(0, by0):by1, max(0, bx0):bx1]
                    density = box.mean() / 255.0 if box.size else 0.0
                    ends.append((density, bcx, bcy))
                if len(ends) == 2 and max(e[0] for e in ends) > 0.45:
                    ends.sort(reverse=True)
                    if ends[0][0] > 1.35 * max(ends[1][0], 0.01):
                        bcx, bcy = ends[0][1], ends[0][2]
                        # snap to the vertical centre of the solid head blob
                        xx, yy = int(bcx), int(bcy)
                        if 0 <= xx < light.shape[1] and 0 <= yy < light.shape[0] and light[yy, xx]:
                            a = yy
                            while a > 0 and light[a - 1, xx]:
                                a -= 1
                            b = yy
                            while b < light.shape[0] - 1 and light[b + 1, xx]:
                                b += 1
                            if 0.5 * ss < b - a + 1 < 1.6 * ss:
                                bcy = (a + b) / 2.0
                        candidate = (bcx, bcy + y0)
            if candidate is None:
                continue
            cx, cy = candidate
            tall = True
        if cx < staff["xstart"] + skip_left * ss:
            continue  # clef / key signature / time signature zone
        if not (staff["top"] - ledger_range * ss - 0.6 * ss < cy < staff["bottom"] + ledger_range * ss + 0.6 * ss):
            continue
        # ink-profile validation: a notehead is WIDE and SHORT at its centre
        hrun = ink_run(filled, int(cy - y0), int(cx), axis=1)
        vrun = ink_run(filled, int(cy - y0), int(cx), axis=0)
        if tall:
            if hrun < 0.95 * ss or hrun > 2.7 * ss:
                continue
        elif not (0.8 * ss <= hrun <= 2.7 * ss) or vrun > 1.9 * ss:
            continue  # flag curl, stem, rest, or welded accidental slab
        if upward_run(healed, ss, cx, cent[i][1]) > 1.15 * ss:
            continue  # tall strokes above the centre: an accidental glyph, not a notehead
        heads.append((cx, cy))
    heads.sort(key=lambda p: p[0])

    merged = []
    for cx, cy in heads:
        if merged and abs(cx - merged[-1][0]) < 0.5 * ss and abs(cy - merged[-1][1]) < 0.6 * ss:
            merged[-1] = ((cx + merged[-1][0]) / 2, (cy + merged[-1][1]) / 2)
        else:
            merged.append((cx, cy))
    return merged


def looks_like_hollow_head(bw, cx, cy, ss):
    """Raw-image shape check: ring of ink around a hollow centre."""
    x0, x3 = int(cx - 1.1 * ss), int(cx + 1.1 * ss)
    yy0, yy3 = int(cy - 0.7 * ss), int(cy + 0.7 * ss)
    if x0 < 0 or yy0 < 0 or x3 >= bw.shape[1] or yy3 >= bw.shape[0]:
        return False
    ring = bw[yy0:yy3, x0:x3].mean() / 255.0
    inner = bw[int(cy - 0.22 * ss):int(cy + 0.22 * ss),
               int(cx - 0.3 * ss):int(cx + 0.3 * ss)]
    hole = inner.mean() / 255.0 if inner.size else 1.0
    return 0.2 < ring < 0.62 and hole < 0.35


def rescue_whole_notes(healed, bw, y0, staff, heads, ss, skip_left, ledger_range):
    """Recover whole noteheads that erosion annihilated: hollow rings whose
    outline break stopped the hole from filling, or rings reduced to a pair
    of side stubs by line extraction."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(healed, 8)
    stubs, found = [], []

    def plausible(cx, cy):
        if cx < staff["xstart"] + skip_left * ss:
            return False
        if not (staff["top"] - ledger_range * ss < cy < staff["bottom"] + ledger_range * ss):
            return False
        return not any(abs(cx - hx) < 1.2 * ss and abs(cy - hy) < 1.2 * ss
                       for hx, hy in heads + found)

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if 1 <= ww <= 0.5 * ss and 0.2 * ss <= hh <= 0.9 * ss:
            stubs.append((cent[i][0], cent[i][1]))
        elif 1.55 * ss <= ww <= 2.45 * ss and 0.55 * ss <= hh <= 1.25 * ss:
            cx, cy = cent[i][0], cent[i][1] + y0
            if plausible(cx, cy) and looks_like_hollow_head(bw, cx, cy, ss):
                found.append((cx, cy))

    for a in range(len(stubs)):
        for b in range(a + 1, len(stubs)):
            (x1, y1), (x2, y2) = stubs[a], stubs[b]
            if abs(y1 - y2) > 0.3 * ss or not (1.1 * ss < abs(x1 - x2) < 2.3 * ss):
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + y0
            if plausible(cx, cy) and looks_like_hollow_head(bw, cx, cy, ss):
                found.append((cx, cy))
    return found


def detect_barlines(bw, staff, skip_left):
    """Barlines: thin verticals whose ends coincide with the staff's top and
    bottom lines. Note stems are tall too, but rarely end on both."""
    ss = staff["space"]
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(6, int(3.7 * ss)))))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(vert, 8)
    bars = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if ww > 0.35 * ss or hh < 3.7 * ss or hh > 4.5 * ss:
            continue
        if abs(y - staff["top"]) > 0.5 * ss or abs(y + hh - staff["bottom"]) > 0.5 * ss:
            continue
        if cent[i][0] < staff["xstart"] + skip_left * ss * 0.5:
            continue
        bars.append(cent[i][0])
    bars.sort()
    merged = [b for j, b in enumerate(bars) if j == 0 or b - bars[j - 1] > ss]
    return merged


# --------------------------------------------------------------------------
# Accidental detection & classification
# --------------------------------------------------------------------------

def stroke_columns(mask, ss):
    """Return [(col_centre, top_row, bottom_row)] for each tall vertical stroke."""
    h, w = mask.shape
    cols = []
    for x in range(w):
        rows = np.where(mask[:, x] > 0)[0]
        cols.append((len(rows), rows[0] if len(rows) else 0, rows[-1] if len(rows) else 0))
    thresh = 0.55 * h
    groups = []
    x = 0
    while x < w:
        if cols[x][0] > thresh:
            x0 = x
            while x < w and cols[x][0] > thresh:
                x += 1
            xs = range(x0, x)
            top = min(cols[c][1] for c in xs)
            bot = max(cols[c][2] for c in xs)
            groups.append(((x0 + x - 1) / 2.0, top, bot))
        else:
            x += 1
    # merge groups closer than a stroke width apart (anti-aliasing splits)
    merged = []
    for g in groups:
        if merged and g[0] - merged[-1][0] < max(2, 0.18 * ss):
            merged[-1] = ((g[0] + merged[-1][0]) / 2, min(g[1], merged[-1][1]), max(g[2], merged[-1][2]))
        else:
            merged.append(g)
    return merged


def classify_accidental(mask, ss):
    """Classify a candidate glyph mask as 'flat' / 'sharp' / 'natural' / 'unknown'."""
    h, w = mask.shape
    strokes = stroke_columns(mask, ss)
    if len(strokes) == 1:
        # flat: one full-height stem at the LEFT, loop bulging right in the lower half
        col, top, bot = strokes[0]
        if col > 0.55 * w or top > 0.08 * h or bot < 0.92 * h:
            return "unknown"
        lower_right = mask[int(0.45 * h):, int(0.45 * w):]
        upper_right = mask[:int(0.4 * h), int(0.5 * w):]
        if lower_right.sum() > 3 * max(upper_right.sum(), 1):
            return "flat"
        return "unknown"
    if len(strokes) == 2:
        (x1, t1, b1), (x2, t2, b2) = strokes
        if x2 - x1 < 0.25 * ss:
            return "unknown"
        # natural: left stroke reaches the glyph top, right stroke reaches the
        # bottom, each offset from the other by a substantial fraction of the
        # height. Sharp: both strokes span nearly the full glyph height.
        top_off, bot_off = t2 - t1, b2 - b1
        if top_off > 0.12 * h and bot_off > 0.12 * h:
            return "natural"
        if abs(top_off) <= 0.12 * h and abs(bot_off) <= 0.12 * h:
            return "sharp"
        return "unknown"
    return "unknown"


def strip_long_runs(win, max_run):
    """Zero horizontal ink runs longer than max_run (ledger-line segments)."""
    for r in range(win.shape[0]):
        row = win[r] > 0
        x = 0
        w = len(row)
        while x < w:
            if row[x]:
                x0 = x
                while x < w and row[x]:
                    x += 1
                if x - x0 > max_run:
                    win[r, x0:x] = 0
            else:
                x += 1


def find_accidental(light, y0, ss, cx, cy):
    """Look for an accidental glyph immediately left of a notehead.
    Returns 'flat' / 'sharp' / 'natural' / 'unknown' / None.
    Staff-line removal can fragment a glyph, so components are first
    clustered by proximity (via dilation) and each cluster is judged whole."""
    h, w = light.shape
    wx0 = max(0, int(cx - 2.4 * ss))
    wx1 = max(0, int(cx - 0.45 * ss))
    wy0 = max(0, int(cy - y0 - 2.1 * ss))
    wy1 = min(h, int(cy - y0 + 2.1 * ss))
    if wx1 - wx0 < 3 or wy1 - wy0 < 3:
        return None
    win = light[wy0:wy1, wx0:wx1].copy()
    strip_long_runs(win, int(1.5 * ss))
    # Tiered fragment gluing: try untouched components first; only if nothing
    # valid is found, glue vertically-cut fragments; as a last resort glue
    # horizontally too. Stronger gluing risks lassoing a neighbouring stem
    # into the cluster and oversizing it, so gentler tiers get priority.
    for glue in (None, (1, 7), (3, 7)):
        glued = win if glue is None else cv2.dilate(
            win, cv2.getStructuringElement(cv2.MORPH_RECT, glue))
        best = try_accidental_clusters(win, glued, wx0, wy0, y0, ss, cx, cy)
        if best is not None:
            return classify_accidental(best, ss)
    return None


def try_accidental_clusters(win, glued, wx0, wy0, y0, ss, cx, cy):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(glued, 8)
    best = None
    for i in range(1, n):
        cluster = (lab == i) & (win > 0)          # original pixels of this cluster
        ys, xs = np.nonzero(cluster)
        if len(xs) == 0:
            continue
        x, y = xs.min(), ys.min()
        ww, hh = xs.max() - x + 1, ys.max() - y + 1
        if not (1.5 * ss < hh < 3.3 * ss):        # accidentals are ~2-3 spaces tall
            continue
        if not (0.35 * ss < ww < 1.4 * ss):       # wider than a stem, narrower than 1.5 sp
            continue
        right_edge = wx0 + x + ww
        if right_edge < cx - 1.6 * ss:            # too far left: previous note's flag/rest
            continue
        gy = wy0 + y + hh / 2 + y0                # glyph vertical centre in page coords
        if not (cy - 1.4 * ss < gy < cy + 0.9 * ss):   # flats sit high; allow asymmetry
            continue
        if best is None or right_edge > best[0]:
            best = (right_edge, cluster[y:y + hh, x:x + ww])
    return None if best is None else best[1].astype(np.uint8) * 255


# --------------------------------------------------------------------------
# Key signature detection
# --------------------------------------------------------------------------

def detect_staff_key(light, y0, staff):
    """Read the key signature printed after the clef on one staff.
    Returns ('flat' | 'sharp', count) or (None, 0) for no signature.
    Candidate glyphs are validated against the vertical positions engraving
    puts them at (FLAT_STEPS / SHARP_STEPS), which also keeps time-signature
    digits from being mistaken for accidentals."""
    ss = staff["space"]
    h, w = light.shape
    wx0 = min(w, max(0, int(staff["xstart"] + 2.6 * ss)))     # skip the clef
    wx1 = min(w, int(staff["xstart"] + 10.5 * ss))
    wy0 = max(0, int(staff["top"] - y0 - 1.5 * ss))
    wy1 = min(h, int(staff["bottom"] - y0 + 1.5 * ss))
    if wx1 - wx0 < 3 or wy1 - wy0 < 3:
        return None, 0
    win = light[wy0:wy1, wx0:wx1].copy()
    strip_long_runs(win, int(1.5 * ss))

    # Tiered fragment gluing as in find_accidental: these glyphs sit ON the
    # staff lines, so line removal fragments them worse than anywhere else.
    glyphs = []
    for glue in (None, (1, 7), (3, 7)):
        glued = win if glue is None else cv2.dilate(
            win, cv2.getStructuringElement(cv2.MORPH_RECT, glue))
        n, lab, _, _ = cv2.connectedComponentsWithStats(glued, 8)
        for i in range(1, n):
            cluster = (lab == i) & (win > 0)
            ys, xs = np.nonzero(cluster)
            if len(xs) == 0:
                continue
            x, y = xs.min(), ys.min()
            ww, hh = xs.max() - x + 1, ys.max() - y + 1
            if not (1.5 * ss < hh < 3.3 * ss and 0.35 * ss < ww < 1.4 * ss):
                continue
            kind = classify_accidental(
                cluster[y:y + hh, x:x + ww].astype(np.uint8) * 255, ss)
            glyphs.append((x, y, ww, hh, kind))
        if any(g[4] in ("flat", "sharp") for g in glyphs):
            break
        glyphs = []
    if not glyphs:
        return None, 0

    # Accept the leading left-to-right run of same-type accidentals whose
    # steps follow the expected sequence. A flat is anchored at its lower
    # loop (the stem rises ~1.5 steps above the notated position); a sharp
    # at its centre.
    expect = {"flat": FLAT_STEPS, "sharp": SHARP_STEPS}
    run_type, count, last_x = None, 0, 0
    for x, y, ww, hh, kind in sorted(glyphs, key=lambda g: g[0]):
        if kind not in ("flat", "sharp"):
            if run_type:
                break            # time signature or other clutter: run is over
            continue             # sliver of the clef edge: keep looking
        anchor = y + hh - 0.5 * ss if kind == "flat" else y + hh / 2.0
        step = 2 * (staff["bottom"] - (wy0 + anchor + y0)) / ss
        if run_type is None:
            if abs(step - expect[kind][0]) <= 0.7:
                run_type, count, last_x = kind, 1, x
            continue
        if (kind != run_type or count >= 6
                or not (0.5 * ss < x - last_x < 2.2 * ss)
                or abs(step - expect[kind][count]) > 0.7):
            break
        count, last_x = count + 1, x
    return (run_type, count) if run_type else (None, 0)


def detect_key_signature(staves, rois):
    """Majority-vote the page's key across its staves.
    rois is the [(healed, light, y0)] list parallel to staves.
    Returns (key_name, votes_for_winner, staff_count)."""
    votes = []
    for staff, (_, light, y0) in zip(staves, rois):
        kind, n = detect_staff_key(light, y0, staff)
        votes.append((SHARP_KEYS if kind == "sharp" else FLAT_KEYS)[n])
    tally = Counter(votes)
    # A staff whose glyphs were unreadable votes C, so ties favour a real
    # signature over an empty reading.
    key, n = max(tally.items(), key=lambda kv: (kv[1], kv[0] != "C"))
    return key, n, len(votes)


# --------------------------------------------------------------------------
# Annotation
# --------------------------------------------------------------------------

def load_font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                 "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate_page(path, args, key_acc):
    color = cv2.imread(path)
    if color is None:
        sys.exit(f"Could not read image: {path}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray, color, angle = deskew(gray, color)
    bw = binarize(gray)
    staves, horiz = find_staves(bw)
    if not staves:
        sys.exit(f"No staves found in {path} — try a cleaner/flatter image.")

    rois = [clean_staff_roi(bw, staff, args.ledger_range) for staff in staves]
    if key_acc is None:
        key_name, agree, nstaves = detect_key_signature(staves, rois)
        k = KEYS[key_name.upper()]
        desc = ("no sharps or flats" if k == 0
                else f"{abs(k)} {'flat' if k < 0 else 'sharp'}{'s' if abs(k) > 1 else ''}")
        print(f"  detected key: {key_name} major ({desc}; {agree}/{nstaves} staves agree)")
        key_acc = key_accidentals(key_name)

    caption, rgb = METHOD_INFO[args.method]
    img = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    dbg = color.copy() if args.debug else None

    total, unknown = 0, 0
    for si, staff in enumerate(staves):
        ss = staff["space"]
        font = load_font(max(12, int(1.5 * ss)))
        healed, light, y0 = rois[si]
        heads = detect_noteheads(healed, light, y0, staff, args.ledger_range,
                                 args.sensitivity, args.skip_left)
        heads.extend(rescue_whole_notes(healed, bw, y0, staff, heads, ss,
                                        args.skip_left, args.ledger_range))
        heads.sort(key=lambda p: p[0])
        bars = detect_barlines(bw, staff, args.skip_left) if not args.no_accidentals else []

        active = {}          # (letter, octave) -> alteration, cleared at barlines
        bar_i = 0
        names = []
        for cx, cy in heads:
            while bar_i < len(bars) and bars[bar_i] < cx - 0.4 * ss:
                active.clear()
                bar_i += 1
            step = int(round(2 * (staff["bottom"] - cy) / ss))
            letter, octave = step_to_letter(step)

            glyph = None if args.no_accidentals else find_accidental(light, y0, ss, cx, cy)
            starred = ""
            verify = False
            if glyph == "unknown":
                alter = active.get((letter, octave), key_acc.get(letter, 0))
                verify = True
            elif glyph:
                alter = ACC_ALTER[glyph]
                active[(letter, octave)] = alter
                starred = "*"
            elif (letter, octave) in active:
                alter = active[(letter, octave)]
                starred = "*"
            else:
                alter = key_acc.get(letter, 0)

            total += 1
            name = f"{letter}{ACC_MARK[alter]}{octave}{starred}"
            pos = position_for(midi_of(letter, octave, alter), args.method)
            if pos and not verify:
                label, col = pos, rgb
            elif pos:
                label, col = pos, UNKNOWN_COLOR   # unreadable glyph nearby: verify by eye
                name += "?"
                unknown += 1
            else:
                label, col = "?", UNKNOWN_COLOR   # outside the position table
                unknown += 1
            names.append(f"{name}:{label}")

            if args.placement == "below":
                ty = max(cy + 1.1 * ss, staff["bottom"] + 1.5 * ss)
            else:
                ty = min(cy - 2.4 * ss, staff["top"] - 2.8 * ss)
            tw = draw.textlength(label, font=font)
            draw.text((cx - tw / 2, ty), label, fill=col, font=font)
            if args.debug:
                cv2.circle(dbg, (int(cx), int(cy)), int(0.6 * ss), (0, 200, 0), 2)
        print(f"  staff {si + 1}: {len(heads)} notes, {len(bars)} barlines  " + " ".join(names))
        if args.debug:
            for ly in staff["lines"]:
                cv2.line(dbg, (0, int(ly)), (color.shape[1], int(ly)), (0, 0, 255), 1)
            for bx in bars:
                cv2.line(dbg, (int(bx), int(staff["top"])), (int(bx), int(staff["bottom"])),
                         (255, 0, 200), 2)

    ss0 = staves[0]["space"]
    strip = int(3 * ss0)
    out = Image.new("RGB", (img.width, img.height + strip), "white")
    out.paste(img, (0, 0))
    ImageDraw.Draw(out).text((int(ss0), img.height + int(0.7 * ss0)),
                             f"Alto trombone (Eb) — {caption}",
                             fill=rgb, font=load_font(max(12, int(1.3 * ss0))))
    if args.debug:
        dpath = os.path.splitext(path)[0] + "_debug.png"
        cv2.imwrite(dpath, dbg)
        print(f"  debug overlay -> {dpath}")
    if angle:
        print(f"  (deskewed {angle:+.2f} degrees)")
    if unknown:
        print(f"  {unknown}/{total} notes could not be resolved (marked '?').")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Three annotation"):])
    ap.add_argument("images", nargs="+", help="sheet music image file(s), in page order")
    ap.add_argument("-m", "--method", choices=["octave", "pitch", "fourth"],
                    default="octave", help="annotation method (default: octave)")
    ap.add_argument("-k", "--key", default="auto",
                    help='major key signature, e.g. Eb, F, G, or "auto" to read it '
                         "from the image (default: auto). Pass a key explicitly if "
                         "auto-detection reads it wrong.")
    ap.add_argument("-o", "--output", help="output PDF path")
    ap.add_argument("--placement", choices=["below", "above"], default="below",
                    help="print numbers below (default) or above the notes")
    ap.add_argument("--ledger-range", type=float, default=4.5,
                    help="how many staff spaces beyond the staff to look for notes (default 4.5)")
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help="notehead detector strictness; lower (<1) finds more/faint noteheads "
                         "but risks false positives (try 0.85 for photos)")
    ap.add_argument("--skip-left", type=float, default=8.0,
                    help="staff spaces to skip at the left of each staff (clef/key/time "
                         "signature zone, default 8.0; lower it if pickup notes are missed)")
    ap.add_argument("--no-accidentals", action="store_true",
                    help="ignore printed accidentals; use the key signature only")
    ap.add_argument("--debug", action="store_true",
                    help="save a *_debug.png overlay showing detected staves, noteheads "
                         "and barlines")
    args = ap.parse_args()

    if args.key.strip().lower() == "auto":
        key_acc = None
    else:
        key_acc = key_accidentals(args.key)
        if args.key.upper() == "C":
            print("Note: using key of C (no sharps/flats). Use -k auto to read it from the image.")

    pages = []
    for p in args.images:
        key_desc = "auto-detect" if key_acc is None else f"key of {args.key}"
        print(f"Processing {p} [{args.method}, {key_desc}] ...")
        pages.append(annotate_page(p, args, key_acc))

    outpath = args.output or os.path.splitext(args.images[0])[0] + f"_alto_{args.method}.pdf"
    pages[0].save(outpath, save_all=True, append_images=pages[1:], resolution=150)
    print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
