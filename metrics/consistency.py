"""No-GT wide-view consistency: CBSR and PD.

Ported from UniSplat_infer/metrics/consistency.py. The formulas are the
measurement definition; do not retune them here.

CBSR is the cross-band seam ratio. It asks whether the left and right third
boundaries are brightness steps that run from sky to road. PD is panel detail.
It asks whether the image still has local contrast. A low CBSR only counts as
a better seam when PD has not collapsed.

The ranking number for a set of frames is the mean. Median and P90 are
descriptive. Both metrics are defined for any wide render whose width is at
least 160 pixels; the usual driving width is 1554.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

BANDS = 6
WINDOW = 40
GAP = 8
MARGIN = 64
REF_START = 80
REF_STEP = 12
REF_EXCLUDE = 80


def luminance(rgb):
    """Linear luminance in the same 0-255 units as ``rgb``."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def seam_columns(width):
    third = width // 3
    return (third, 2 * third)


def reference_columns(width):
    """Columns away from both seams, used as the ordinary-column baseline."""
    blocked = [(seam - REF_EXCLUDE, seam + REF_EXCLUDE) for seam in seam_columns(width)]
    columns = []
    for column in range(REF_START, width - REF_START, REF_STEP):
        if any(left <= column <= right for left, right in blocked):
            continue
        columns.append(column)
    return columns


def interiors(width):
    third = width // 3
    return (
        (0, third - MARGIN),
        (third + MARGIN, 2 * third - MARGIN),
        (2 * third + MARGIN, width),
    )


def _band_jumps(field, column):
    """Right-minus-left log-luminance step in each horizontal band."""
    height = field.shape[0]
    band_h = height // BANDS
    if band_h < 1:
        raise ValueError(f"image height {height} is shorter than {BANDS} bands")
    jumps = []
    for band in range(BANDS):
        rows = slice(band * band_h, (band + 1) * band_h)
        left = field[rows, column - GAP - WINDOW:column - GAP]
        right = field[rows, column + GAP:column + GAP + WINDOW]
        jumps.append(float(right.mean() - left.mean()))
    return np.asarray(jumps, dtype=np.float64)


def column_score(field, column):
    """Absolute cross-band step, down-weighted when the bands disagree.

    A car or pole moves one or two bands. An exposure seam moves sky, horizon,
    and road together, so the cross-band median stays large and the signs agree.
    A zero median is treated as positive only for the sign comparison.
    """
    jumps = _band_jumps(field, column)
    median = float(np.median(jumps))
    reference_sign = np.sign(median if median != 0.0 else 1.0)
    agreement = float(np.mean(np.sign(jumps) == reference_sign))
    return abs(median) * agreement


def cbsr(rgb):
    """Cross-band seam ratio. Lower is better. One means an ordinary column."""
    field = np.log(luminance(rgb) + 1.0)
    width = field.shape[1]
    if width < REF_START * 2 + WINDOW + GAP:
        raise ValueError(f"image width {width} is too small for CBSR")
    seams = seam_columns(width)
    refs = reference_columns(width)
    if not refs:
        raise ValueError(f"image width {width} has no reference columns")
    seam_score = float(np.mean([column_score(field, column) for column in seams]))
    baseline = float(np.median([column_score(field, column) for column in refs]))
    return seam_score / (baseline + 1e-3)


def panel_detail(rgb):
    """Median interior horizontal contrast after a one-pixel blur.

    Higher means more local contrast. This is a companion to CBSR, not a seam
    score: a flat gray image can have a CBSR near one.
    """
    gradient = np.abs(np.diff(gaussian_filter(luminance(rgb), sigma=1), axis=1))
    scores = []
    for left, right in interiors(gradient.shape[1] + 1):
        patch = gradient[:, left:right - 1]
        if patch.size == 0:
            continue
        scores.append(float(np.median(patch.mean(axis=1))))
    if not scores:
        raise ValueError("image has no interior panel for PD")
    return float(np.median(scores))


def score_image(rgb):
    return {"cbsr": cbsr(rgb), "pd": panel_detail(rgb)}
