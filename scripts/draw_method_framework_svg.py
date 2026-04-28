#!/usr/bin/env python
"""Draw the PlantSeg-RIS method framework as editable SVG plus PDF/PNG.

Code-to-paper mapping used by this figure:
- The paper term HAPWAM is used in the figure as requested. In this checkout,
  lib/backbone.py currently routes align_module in ("spam", "hapwam") to the
  SPAM implementation, and HAPWAM is a compatibility alias.
- Healthy-Suppressed Language Gate (HLG) maps to HealthySuppressedLanguageGate.
  The code stores a suppression map and applies (1 - alpha * G_i) to F_i; the
  figure uses the compact paper label E_i = V_i + G_i odot F_i.
- The training auxiliary gate loss is computed only for hlg_stage3/hlg_stage4.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"
SVG_PATH = OUT_DIR / "method_framework.svg"
PDF_PATH = OUT_DIR / "method_framework.pdf"
PNG_PATH = OUT_DIR / "method_framework.png"

W, H = 3200, 1800


COLORS = {
    "bg": "#ffffff",
    "ink": "#202938",
    "muted": "#5f6b7a",
    "line": "#586272",
    "language": "#e3f1ff",
    "language_stroke": "#77a9d7",
    "visual": "#e1f4ee",
    "visual_stroke": "#72b7a6",
    "hapwam": "#ffe5c2",
    "hapwam_stroke": "#d28b43",
    "hlg": "#f8d5dc",
    "hlg_stroke": "#c86f7e",
    "decoder": "#e7e4f5",
    "decoder_stroke": "#887fba",
    "train": "#f4f5f7",
    "train_stroke": "#9aa3ad",
    "white": "#ffffff",
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    fill: str
    stroke: str
    font: int = 30
    radius: float = 18
    stroke_width: float = 3
    dashed: bool = False
    id: str = ""

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.y + self.h / 2

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.w, self.y + self.h / 2

    @property
    def top(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y

    @property
    def bottom(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


boxes: list[Box] = []
arrows: list[dict] = []
labels: list[dict] = []
small_lines: list[dict] = []


def add_box(*args, **kwargs) -> Box:
    box = Box(*args, **kwargs)
    boxes.append(box)
    return box


def add_label(x: float, y: float, text: str, size: int = 26, color: str = COLORS["ink"], anchor: str = "middle") -> None:
    labels.append({"x": x, "y": y, "text": text, "size": size, "color": color, "anchor": anchor})


def add_arrow(points: Iterable[tuple[float, float]], color: str = COLORS["line"], width: float = 4,
              dashed: bool = False, end: bool = True) -> None:
    arrows.append({"points": list(points), "color": color, "width": width, "dashed": dashed, "end": end})


def add_line(points: Iterable[tuple[float, float]], color: str = COLORS["line"], width: float = 2,
             dashed: bool = False) -> None:
    small_lines.append({"points": list(points), "color": color, "width": width, "dashed": dashed})


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def build_layout() -> None:
    add_label(W / 2, 78, "PlantSeg-RIS Method Framework", 48)

    # Inputs and language branch.
    text_in = add_box(90, 180, 310, 88, "Referring\nExpression (T)", COLORS["language"], COLORS["language_stroke"], 30)
    deberta = add_box(490, 172, 250, 104, "DeBERTa", COLORS["language"], COLORS["language_stroke"], 34)
    lang_l = add_box(850, 140, 420, 86, "Token-level Language\nFeatures (L)", COLORS["language"], COLORS["language_stroke"], 28)
    mask_m = add_box(850, 252, 420, 78, "Text Valid\nMask (M)", COLORS["language"], COLORS["language_stroke"], 28)

    add_arrow([text_in.right, deberta.left], COLORS["language_stroke"], 5)
    add_arrow([deberta.right, lang_l.left], COLORS["language_stroke"], 4)
    add_arrow([(deberta.x + deberta.w, deberta.y + 72), (805, deberta.y + 72), mask_m.left],
              COLORS["language_stroke"], 3.5)

    bus_y_l, bus_y_m = 415, 465
    bus_x0, bus_x1 = 980, 2220
    add_label((bus_x0 + bus_x1) / 2, bus_y_l - 24, "shared language inputs to each HAPWAM", 24, COLORS["muted"])
    add_arrow([lang_l.bottom, (lang_l.bottom[0], bus_y_l), (bus_x1, bus_y_l)], COLORS["language_stroke"], 3.5, end=False)
    add_arrow([mask_m.bottom, (mask_m.bottom[0], bus_y_m), (bus_x1, bus_y_m)], COLORS["language_stroke"], 3.0, dashed=True, end=False)

    # Visual branch and four language-aware stages.
    image_in = add_box(90, 760, 310, 100, "Image (I)", COLORS["visual"], COLORS["visual_stroke"], 34)
    swin = add_box(490, 744, 250, 132, "Swin-B", COLORS["visual"], COLORS["visual_stroke"], 36)
    add_arrow([image_in.right, swin.left], COLORS["visual_stroke"], 6)

    stage_x = [860, 1210, 1560, 1910]
    stage_outputs: dict[int, Box] = {}
    stage_boxes: dict[int, Box] = {}
    f_boxes: dict[int, Box] = {}
    gate_points: list[tuple[float, float, str]] = []

    previous_stage_entry = swin.right
    for idx, x in enumerate(stage_x, start=1):
        stage = add_box(x, 620, 280, 86, f"Stage {idx}\n(V_{idx})", COLORS["visual"], COLORS["visual_stroke"], 28)
        stage_boxes[idx] = stage
        add_arrow([previous_stage_entry, (stage.x - 38, previous_stage_entry[1]), stage.left],
                  COLORS["visual_stroke"], 6)

        hap = add_box(x, 770, 280, 240, "", COLORS["hapwam"], COLORS["hapwam_stroke"], 17)
        add_label(hap.x + hap.w / 2, hap.y + 45, f"HAPWAM_{idx}", 27, COLORS["ink"])
        add_label(hap.x + hap.w / 2, hap.y + 76, "Hierarchically Adaptive\nPixel-Word Alignment", 17, COLORS["muted"])
        add_label(hap.x + hap.w / 2, hap.y + 125, "dynamic token routing", 18, COLORS["muted"])
        add_label(hap.x + hap.w / 2, hap.y + 164, "shared pixel-word alignment", 18, COLORS["muted"])
        add_label(hap.x + hap.w / 2, hap.y + 203, "adaptive branch fusion", 18, COLORS["muted"])
        add_arrow([stage.bottom, hap.top], COLORS["visual_stroke"], 4.5)
        add_arrow([(stage.x + stage.w * 0.30, bus_y_l), (stage.x + stage.w * 0.30, hap.y)], COLORS["language_stroke"], 3.0)
        add_arrow([(stage.x + stage.w * 0.70, bus_y_m), (stage.x + stage.w * 0.70, hap.y)], COLORS["language_stroke"], 2.8, dashed=True)
        add_label(stage.x + stage.w * 0.30 - 22, hap.y - 15, "L", 23, COLORS["language_stroke"])
        add_label(stage.x + stage.w * 0.70 + 24, hap.y - 15, "M", 23, COLORS["language_stroke"])

        f_box = add_box(x + 68, 1060, 145, 64, f"F_{idx}", COLORS["white"], COLORS["hapwam_stroke"], 30, radius=16)
        f_boxes[idx] = f_box
        add_arrow([hap.bottom, f_box.top], COLORS["hapwam_stroke"], 4.5)

        if idx in (1, 2):
            add_label(f_box.x + f_box.w / 2, f_box.y + f_box.h + 44, "early language-aware\nalignment", 20, COLORS["muted"])
            stage_outputs[idx] = f_box
            previous_stage_entry = f_box.right
        else:
            hlg = add_box(x - 4, 1190, 288, 170, "", COLORS["hlg"], COLORS["hlg_stroke"], 22)
            e_box = add_box(x + 68, 1410, 145, 64, f"E_{idx}", COLORS["white"], COLORS["hlg_stroke"], 30, radius=16)
            add_arrow([f_box.bottom, hlg.top], COLORS["hlg_stroke"], 4.0)
            add_arrow([(stage.x + stage.w * 0.17, stage.y + stage.h), (stage.x + stage.w * 0.17, 1168),
                       (hlg.x, 1168), hlg.left], COLORS["visual_stroke"], 3.0)
            add_label(hlg.x + hlg.w / 2, hlg.y + 42, "HLG", 28, COLORS["ink"])
            add_label(hlg.x + hlg.w / 2, hlg.y + 76, "Healthy-Suppressed\nLanguage Gate", 20, COLORS["ink"])
            add_label(hlg.x + hlg.w / 2, hlg.y + 126, f"suppression gate map (G_{idx})", 17, COLORS["hlg_stroke"])
            add_label(hlg.x + hlg.w / 2, hlg.y + 153, f"E_{idx} = V_{idx} + G_{idx} \u2299 F_{idx}", 18, COLORS["muted"])
            add_arrow([hlg.bottom, e_box.top], COLORS["hlg_stroke"], 4.5)
            stage_outputs[idx] = e_box
            gate_points.append((hlg.x + hlg.w / 2, hlg.y + 126, f"G_{idx}"))
            previous_stage_entry = e_box.right

    # Decoder and output head.
    decoder = add_box(2350, 700, 360, 490, "", COLORS["decoder"], COLORS["decoder_stroke"], 30)
    add_label(decoder.x + decoder.w / 2, decoder.y + 70, "Top-down\nLightweight Decoder", 30, COLORS["ink"])
    decoder_slots = {
        4: (decoder.x, decoder.y + 150),
        3: (decoder.x, decoder.y + 240),
        2: (decoder.x, decoder.y + 330),
        1: (decoder.x, decoder.y + 420),
    }
    for sid, y in [(4, decoder.y + 150), (3, decoder.y + 240), (2, decoder.y + 330), (1, decoder.y + 420)]:
        add_line([(decoder.x + 52, y), (decoder.x + 130, y)], COLORS["decoder_stroke"], 5)
        label = f"E_{sid}" if sid in (3, 4) else f"F_{sid}"
        add_label(decoder.x + 220, y + 8, label, 24, COLORS["muted"])
    add_line([(decoder.x + 68, decoder.y + 150), (decoder.x + 68, decoder.y + 420)], COLORS["decoder_stroke"], 5)

    for sid in (4, 3, 2, 1):
        start = stage_outputs[sid].right
        end = decoder_slots[sid]
        mid_x = end[0] - 80
        add_arrow([start, (mid_x, start[1]), (mid_x, end[1]), end], COLORS["decoder_stroke"], 4.0)

    head = add_box(2825, 790, 270, 116, "Prediction\nHead", COLORS["decoder"], COLORS["decoder_stroke"], 30)
    pred = add_box(2825, 1000, 270, 82, "Final Prediction (P)", COLORS["white"], COLORS["decoder_stroke"], 27)
    mask = add_box(2825, 1175, 270, 82, "Lesion Mask", COLORS["white"], COLORS["visual_stroke"], 31)
    add_arrow([decoder.right, head.left], COLORS["decoder_stroke"], 6)
    add_arrow([head.bottom, pred.top], COLORS["decoder_stroke"], 4.5)
    add_arrow([pred.bottom, mask.top], COLORS["visual_stroke"], 4.5)

    # Training-only objective panel. Dashed arrows avoid implying an inference path.
    train = add_box(650, 1530, 2050, 220, "", COLORS["train"], COLORS["train_stroke"], 27, radius=22,
                    stroke_width=2.8, dashed=True)
    add_label(train.x + 180, train.y + 42, "Training Objectives", 31, COLORS["ink"])
    p_train = add_box(760, 1628, 230, 70, "Final Prediction\n(P)", COLORS["white"], COLORS["train_stroke"], 22)
    gt = add_box(1035, 1628, 230, 70, "Ground Truth\nMask (Y)", COLORS["white"], COLORS["train_stroke"], 22)
    seg = add_box(1310, 1628, 280, 70, "Segmentation\nLoss L_seg", COLORS["white"], COLORS["train_stroke"], 22)
    g_train = add_box(1640, 1628, 185, 70, "G_3, G_4", COLORS["white"], COLORS["train_stroke"], 24)
    omega = add_box(1870, 1614, 250, 98, "Omega_dis\n+\nOmega_fh", COLORS["white"], COLORS["train_stroke"], 22)
    gate = add_box(2160, 1628, 260, 70, "Auxiliary Gate\nLoss L_gate", COLORS["white"], COLORS["train_stroke"], 21)
    total = add_box(2510, 1628, 145, 70, "Total\nLoss L", COLORS["white"], COLORS["train_stroke"], 22)
    add_arrow([p_train.right, seg.left], COLORS["train_stroke"], 3.0)
    add_arrow([gt.right, seg.left], COLORS["train_stroke"], 3.0)
    add_arrow([seg.right, total.left], COLORS["train_stroke"], 3.0)
    add_arrow([g_train.right, omega.left], COLORS["train_stroke"], 3.0)
    add_arrow([omega.right, gate.left], COLORS["train_stroke"], 3.0)
    add_arrow([gate.right, total.left], COLORS["train_stroke"], 3.0)
    for gx, gy, _ in gate_points:
        add_arrow([(gx, gy + 12), (gx + 65, gy + 12), (gx + 65, 1578), g_train.top],
                  COLORS["train_stroke"], 2.6, dashed=True)


def svg_text(x: float, y: float, text: str, size: int, color: str, anchor: str = "middle") -> str:
    lines = text.split("\n")
    line_h = size * 1.18
    start_y = y - (len(lines) - 1) * line_h / 2
    out = []
    for i, line in enumerate(lines):
        out.append(
            f'<text x="{x:.1f}" y="{start_y + i * line_h:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}">{html.escape(line)}</text>'
        )
    return "\n".join(out)


def render_svg() -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<marker id="arrow" markerWidth="14" markerHeight="10" refX="12" refY="5" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L14,5 L0,10 z" fill="#5a6575"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
    ]

    for line in small_lines:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in line["points"])
        dash = ' stroke-dasharray="9 7"' if line["dashed"] else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{line["color"]}" stroke-width="{line["width"]}" '
                     f'stroke-linecap="round" stroke-linejoin="round"{dash}/>')

    for arrow in arrows:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in arrow["points"])
        dash = ' stroke-dasharray="12 9"' if arrow["dashed"] else ""
        marker = ' marker-end="url(#arrow)"' if arrow["end"] else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{arrow["color"]}" stroke-width="{arrow["width"]}" '
                     f'stroke-linecap="round" stroke-linejoin="round"{dash}{marker}/>')

    for box in boxes:
        dash = ' stroke-dasharray="12 9"' if box.dashed else ""
        parts.append(
            f'<rect x="{box.x:.1f}" y="{box.y:.1f}" width="{box.w:.1f}" height="{box.h:.1f}" '
            f'rx="{box.radius}" ry="{box.radius}" fill="{box.fill}" stroke="{box.stroke}" '
            f'stroke-width="{box.stroke_width}"{dash}/>'
        )
        if box.label:
            parts.append(svg_text(box.x + box.w / 2, box.y + box.h / 2 + box.font * 0.35, box.label, box.font, COLORS["ink"]))

    for label in labels:
        parts.append(svg_text(label["x"], label["y"], label["text"], label["size"], label["color"], label["anchor"]))

    parts.append("</svg>")
    return "\n".join(parts)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_center_text(draw: ImageDraw.ImageDraw, box: Box, text: str, size: int, color: str) -> None:
    font = load_font(size)
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    line_h = int(size * 1.18)
    total_h = line_h * (len(lines) - 1) + max(heights or [size])
    y = box.y + (box.h - total_h) / 2 - 2
    for i, line in enumerate(lines):
        x = box.x + (box.w - widths[i]) / 2
        draw.text((x, y + i * line_h), line, fill=color, font=font)


def draw_label_text(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int, color: str, anchor: str) -> None:
    font = load_font(size)
    text = text.replace("\u2299", "x").replace("\u03a9", "Omega")
    lines = text.split("\n")
    line_h = int(size * 1.18)
    total_h = line_h * (len(lines) - 1) + size
    yy = y - total_h / 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        width = bbox[2] - bbox[0]
        if anchor == "middle":
            xx = x - width / 2
        elif anchor == "end":
            xx = x - width
        else:
            xx = x
        draw.text((xx, yy + i * line_h), line, fill=color, font=font)


def arrow_head(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], color: str, scale: float = 1.0) -> None:
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    length = 16 * scale
    spread = 0.42
    pts = [
        p2,
        (p2[0] - length * math.cos(ang - spread), p2[1] - length * math.sin(ang - spread)),
        (p2[0] - length * math.cos(ang + spread), p2[1] - length * math.sin(ang + spread)),
    ]
    draw.polygon(pts, fill=color)


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: str, width: int,
                  dashed: bool = False) -> None:
    if not dashed:
        draw.line(points, fill=color, width=width, joint="curve")
        return

    dash_len = 18
    gap_len = 12
    for p1, p2 in zip(points, points[1:]):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            seg_end = min(pos + dash_len, length)
            draw.line([(x1 + ux * pos, y1 + uy * pos), (x1 + ux * seg_end, y1 + uy * seg_end)],
                      fill=color, width=width)
            pos += dash_len + gap_len


def render_png() -> None:
    img = Image.new("RGB", (W, H), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    for line in small_lines:
        draw_polyline(draw, line["points"], line["color"], int(line["width"]), line.get("dashed", False))

    for arrow in arrows:
        draw_polyline(draw, arrow["points"], arrow["color"], int(arrow["width"]), arrow.get("dashed", False))
        if arrow["end"] and len(arrow["points"]) >= 2:
            arrow_head(draw, arrow["points"][-2], arrow["points"][-1], arrow["color"], max(0.8, arrow["width"] / 4))

    for box in boxes:
        draw.rounded_rectangle([box.x, box.y, box.x + box.w, box.y + box.h], radius=box.radius,
                               fill=box.fill, outline=box.stroke, width=int(box.stroke_width))
        if box.label:
            draw_center_text(draw, box, box.label, box.font, COLORS["ink"])

    for label in labels:
        draw_label_text(draw, label["x"], label["y"], label["text"], label["size"], label["color"], label["anchor"])

    img.save(PNG_PATH, dpi=(300, 300))


def pdf_escape(text: str) -> str:
    normalized = (
        text.replace("\u2299", "x")
        .replace("\u03a9", "Omega")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )
    return normalized


def pdf_color(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


def pdf_stream() -> str:
    cmds: list[str] = []

    def rg(hex_color: str, stroke: bool = False) -> str:
        r, g, b = pdf_color(hex_color)
        return f"{r:.4f} {g:.4f} {b:.4f} {'RG' if stroke else 'rg'}"

    cmds.append(rg(COLORS["bg"]))
    cmds.append(f"0 0 {W} {H} re f")

    def yflip(y: float) -> float:
        return H - y

    def pdf_text_x(text: str, x: float, size: int, anchor: str = "middle") -> float:
        # Lightweight centering for the fallback vector PDF. The SVG is the
        # exact editable master; this keeps the PDF visually usable without
        # external converters.
        width = len(pdf_escape(text)) * size * 0.27
        if anchor == "middle":
            return x - width / 2
        if anchor == "end":
            return x - width
        return x

    for line in small_lines:
        pts = line["points"]
        cmds.append(rg(line["color"], True))
        cmds.append(f"{line['width']:.2f} w")
        if line.get("dashed"):
            cmds.append("[12 9] 0 d")
        else:
            cmds.append("[] 0 d")
        first = pts[0]
        cmds.append(f"{first[0]:.1f} {yflip(first[1]):.1f} m")
        for x, y in pts[1:]:
            cmds.append(f"{x:.1f} {yflip(y):.1f} l")
        cmds.append("S")

    for arrow in arrows:
        pts = arrow["points"]
        cmds.append(rg(arrow["color"], True))
        cmds.append(f"{arrow['width']:.2f} w")
        cmds.append("[12 9] 0 d" if arrow.get("dashed") else "[] 0 d")
        first = pts[0]
        cmds.append(f"{first[0]:.1f} {yflip(first[1]):.1f} m")
        for x, y in pts[1:]:
            cmds.append(f"{x:.1f} {yflip(y):.1f} l")
        cmds.append("S")
        if arrow.get("end") and len(pts) >= 2:
            p1, p2 = pts[-2], pts[-1]
            ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            length = 13 * max(0.8, arrow["width"] / 4)
            spread = 0.42
            tri = [
                p2,
                (p2[0] - length * math.cos(ang - spread), p2[1] - length * math.sin(ang - spread)),
                (p2[0] - length * math.cos(ang + spread), p2[1] - length * math.sin(ang + spread)),
            ]
            cmds.append(rg(arrow["color"]))
            cmds.append(f"{tri[0][0]:.1f} {yflip(tri[0][1]):.1f} m "
                        f"{tri[1][0]:.1f} {yflip(tri[1][1]):.1f} l "
                        f"{tri[2][0]:.1f} {yflip(tri[2][1]):.1f} l h f")

    for box in boxes:
        cmds.append(rg(box.fill))
        cmds.append(rg(box.stroke, True))
        cmds.append(f"{box.stroke_width:.2f} w")
        cmds.append("[12 9] 0 d" if box.dashed else "[] 0 d")
        cmds.append(f"{box.x:.1f} {yflip(box.y + box.h):.1f} {box.w:.1f} {box.h:.1f} re B")
        if box.label:
            for i, line in enumerate(box.label.split("\n")):
                lines = box.label.split("\n")
                yy = box.y + box.h / 2 + (i - (len(lines) - 1) / 2) * box.font * 1.12 + box.font * 0.28
                xx = pdf_text_x(line, box.x + box.w / 2, box.font)
                escaped = pdf_escape(line)
                cmds.append(rg(COLORS["ink"]))
                cmds.append(f"BT /F1 {box.font} Tf 1 0 0 1 {xx:.1f} {yflip(yy):.1f} Tm ({escaped}) Tj ET")

    for label in labels:
        for i, line in enumerate(label["text"].split("\n")):
            lines = label["text"].split("\n")
            yy = label["y"] + (i - (len(lines) - 1) / 2) * label["size"] * 1.12
            escaped = pdf_escape(line)
            xx = pdf_text_x(line, label["x"], label["size"], label.get("anchor", "middle"))
            cmds.append(rg(label["color"]))
            cmds.append(f"BT /F1 {label['size']} Tf 1 0 0 1 {xx:.1f} {yflip(yy):.1f} Tm ({escaped}) Tj ET")

    return "\n".join(cmds).encode("latin-1", errors="replace").decode("latin-1")


def write_pdf() -> None:
    stream = pdf_stream().encode("latin-1", errors="replace")
    objects = []
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(
        f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objects.append("4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    objects.append(f"5 0 obj << /Length {len(stream)} >> stream\n{stream.decode('latin-1')}\nendstream endobj\n")

    data = ["%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"]
    offsets = [0]
    current = len(data[0].encode("latin-1"))
    for obj in objects:
        offsets.append(current)
        data.append(obj)
        current += len(obj.encode("latin-1"))
    xref_pos = current
    xref = ["xref\n", f"0 {len(objects) + 1}\n", "0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n")
    trailer = f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    PDF_PATH.write_bytes(("".join(data) + "".join(xref) + trailer).encode("latin-1", errors="replace"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_layout()
    SVG_PATH.write_text(render_svg(), encoding="utf-8")
    render_png()
    write_pdf()
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
