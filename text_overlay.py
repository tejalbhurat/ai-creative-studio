import re
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


FONT_CANDIDATES = {
    "handwritten": (
        "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    ),
    "editorial": (
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
    "default": (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ),
}


def _load_font(size: int, mode: str = "default") -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES.get(mode, ()) + FONT_CANDIDATES["default"]:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _mode_for_plan(style: str, placement: str, integration: str) -> str:
    words = set(re.findall(r"[a-z]+", f"{style} {placement} {integration}".lower()))
    for keywords, mode in (
        ({"handwritten", "note", "notebook", "paper"}, "handwritten"),
        ({"signage", "sign", "billboard"}, "signage"),
        ({"label", "packaging"}, "label"),
        ({"screen", "interface", "display"}, "screen"),
        ({"caption", "corner", "margin"}, "subtle_caption"),
        ({"editorial", "magazine", "newspaper"}, "editorial"),
        ({"poster", "headline", "advertisement"}, "poster"),
    ):
        if words.intersection(keywords):
            return mode
    return "editorial"


def _wrap_lines(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines or [text]


def _sample_tone(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    sample = image.convert("RGB").crop(box).resize((1, 1))
    return sample.getpixel((0, 0))


def _text_position(
    mode: str,
    placement: str,
    image_size: tuple[int, int],
    text_size: tuple[int, int],
) -> tuple[int, int]:
    width, height = image_size
    text_width, text_height = text_size
    words = set(re.findall(r"[a-z]+", placement.lower()))
    margin = max(20, width // 16)
    if "top" in words:
        return ((width - text_width) // 2, margin)
    if "bottom" in words:
        return ((width - text_width) // 2, height - text_height - margin)
    if "left" in words:
        return (margin, (height - text_height) // 2)
    if "right" in words:
        return (width - text_width - margin, (height - text_height) // 2)
    if mode == "handwritten":
        return (margin, margin)
    if mode == "subtle_caption":
        return (margin, height - text_height - margin)
    if mode == "poster":
        return (width // 12, height // 8)
    # Editorial default: offset from center so it does not look like a centered card.
    return (max(margin, width // 12), max(margin, (height - text_height) // 3))


def _render_text_layer(
    image: Image.Image,
    text: str,
    mode: str,
    placement: str,
) -> Image.Image:
    width, height = image.size
    max_width = int(width * (0.78 if mode in {"poster", "editorial"} else 0.58))
    size = max(22, min(width // (9 if mode == "poster" else 14), height // 6))
    scratch = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(scratch)
    while size > 18:
        font = _load_font(size, "handwritten" if mode == "handwritten" else "editorial")
        lines = _wrap_lines(draw, text, font, max_width)
        if len(lines) <= 3:
            break
        size -= 2
    font = _load_font(size, "handwritten" if mode == "handwritten" else "editorial")
    lines = _wrap_lines(draw, text, font, max_width)
    spacing = max(5, size // 5)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(box[2] - box[0] for box in boxes)
    text_height = sum(box[3] - box[1] for box in boxes) + spacing * (len(lines) - 1)
    left, top = _text_position(mode, placement, image.size, (text_width, text_height))

    # Use tones from the local composition instead of a universal white card.
    tone = _sample_tone(image, (max(0, left), max(0, top), min(width, left + text_width), min(height, top + text_height)))
    luminance = (tone[0] * 299 + tone[1] * 587 + tone[2] * 114) / 1000
    foreground = (245, 240, 226, 245) if luminance < 145 else (24, 24, 28, 235)
    accent = (255, 185, 72, 220) if mode in {"poster", "editorial"} else foreground
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    y = top
    for line, box in zip(lines, boxes):
        line_width = box[2] - box[0]
        x = left if mode in {"handwritten", "subtle_caption"} else left + max(0, (text_width - line_width) // 2)
        mask_draw.text((x, y), line, font=font, fill=255, stroke_width=1)
        y += box[3] - box[1] + spacing

    # A restrained blurred shadow and outline create depth without a box or banner.
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_mask = mask.filter(ImageFilter.GaussianBlur(max(2, size // 16)))
    shadow.putalpha(shadow_mask.point(lambda value: int(value * 0.42)))
    shadow_color = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_color.paste((0, 0, 0, 180), (0, 0, width, height), shadow)
    layer = Image.new("RGBA", image.size, foreground)
    if mode in {"poster", "editorial"}:
        gradient = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gradient_pixels = gradient.load()
        for y_pixel in range(max(0, top), min(height, top + text_height)):
            ratio = (y_pixel - top) / max(1, text_height)
            color = tuple(int(foreground[index] * (1 - ratio) + accent[index] * ratio) for index in range(3))
            for x_pixel in range(max(0, left), min(width, left + text_width)):
                gradient_pixels[x_pixel, y_pixel] = (*color, foreground[3])
        layer = gradient
    layer.putalpha(mask)
    result = Image.alpha_composite(image.convert("RGBA"), shadow_color)
    return Image.alpha_composite(result, layer)


def apply_text_overlay(
    image_path: str | Path,
    text: str | None,
    text_style: str = "",
    text_placement: str = "",
    text_integration: str = "",
    screen_region: tuple[int, int, int, int] | None = None,
) -> Path:
    """Render exact text with image-aware typography; no generic text card is used."""
    source_path = Path(image_path)
    if not text:
        return source_path
    try:
        with Image.open(source_path) as source:
            image = source.convert("RGBA")
        mode = _mode_for_plan(text_style, text_placement, text_integration)
        if mode == "screen" and not _valid_screen_region(screen_region, image.size):
            mode = "editorial"
        rendered = _render_text_layer(image, text, mode, text_placement)
        output_path = source_path.with_name(f"{source_path.stem}_integrated.png")
        rendered.convert("RGB").save(output_path, format="PNG")
        return output_path
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Integrated typography rendering failed: {error}") from error


def _valid_screen_region(
    region: tuple[int, int, int, int] | None,
    image_size: tuple[int, int],
) -> bool:
    if not region or len(region) != 4:
        return False
    left, top, right, bottom = region
    width, height = image_size
    return (
        0 <= left < right <= width
        and 0 <= top < bottom <= height
        and (right - left) >= width * 0.12
        and (bottom - top) >= height * 0.08
    )
