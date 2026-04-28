#!/usr/bin/env python
"""Generate an OpenAI Images API style-reference PNG for the method figure.

This script is intentionally separate from the deterministic SVG/PDF renderer.
It uses the prompt in assets/method_figure_prompt_images_v2.txt and writes a
raster sketch for visual inspiration only.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "assets" / "method_figure_prompt_images_v2.txt"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "method_framework_images_v2.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PlantSeg-RIS method figure PNG with OpenAI Images API.")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT, help="Prompt text file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5"),
        help="OpenAI image model. Override with OPENAI_IMAGE_MODEL or this flag.",
    )
    parser.add_argument(
        "--size",
        default=os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
        help="Image size supported by the selected model, e.g. 1536x1024.",
    )
    parser.add_argument(
        "--quality",
        default=os.getenv("OPENAI_IMAGE_QUALITY", "high"),
        help="Image quality for GPT image models: low, medium, high, or auto.",
    )
    return parser.parse_args()


def fail(message: str, exc: BaseException | None = None) -> int:
    print("ERROR: " + message, file=sys.stderr)
    if exc is not None:
        print("DETAIL: " + str(exc), file=sys.stderr)
    return 1


def write_png_with_dpi(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.save(path, dpi=(300, 300))
    except Exception:
        # DPI metadata is useful but not required for the generated sketch.
        pass


def main() -> int:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        return fail("OPENAI_API_KEY is not set. Set it in the environment before running this script.")

    if not args.prompt.exists():
        return fail(f"Prompt file not found: {args.prompt}")

    try:
        from openai import OpenAI
    except Exception as exc:
        return fail("The Python package `openai` is not installed in this environment.", exc)

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        return fail(f"Prompt file is empty: {args.prompt}")

    try:
        client = OpenAI()
        result = client.images.generate(
            model=args.model,
            prompt=prompt,
            size=args.size,
            quality=args.quality,
            n=1,
            output_format="png",
        )
    except TypeError as exc:
        return fail(
            "The installed OpenAI SDK does not accept the Images API arguments used here. "
            "Upgrade the `openai` Python package or adjust --model/--size/--quality.",
            exc,
        )
    except Exception as exc:
        return fail("OpenAI Images API request failed.", exc)

    try:
        item = result.data[0]
    except Exception as exc:
        return fail("OpenAI response did not include image data.", exc)

    image_b64 = getattr(item, "b64_json", None)
    if image_b64:
        try:
            write_png_with_dpi(args.output, base64.b64decode(image_b64))
        except Exception as exc:
            return fail(f"Failed to decode or write generated image to {args.output}", exc)
    else:
        image_url = getattr(item, "url", None)
        if not image_url:
            return fail("OpenAI response contained neither b64_json nor url image data.")
        try:
            with urllib.request.urlopen(image_url, timeout=60) as response:
                write_png_with_dpi(args.output, response.read())
        except Exception as exc:
            return fail(f"Failed to download image URL into {args.output}", exc)

    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
