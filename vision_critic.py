import base64
import json
import os

from mistralai.client import Mistral


VISION_MODEL = os.getenv("MISTRAL_VISION_MODEL", "mistral-small-latest")


class VisionRateLimitError(RuntimeError):
    """Raised when the Vision API asks the caller to stop due to rate limits."""


CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "prompt_adherence": {"type": "integer", "minimum": 1, "maximum": 10},
        "composition": {"type": "integer", "minimum": 1, "maximum": 10},
        "visual_quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "brand_fit": {"type": "integer", "minimum": 1, "maximum": 10},
        "realism_naturalness": {"type": "integer", "minimum": 1, "maximum": 10},
        "text_integration": {"type": "integer", "minimum": 1, "maximum": 10},
        "typography_quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "text_visible": {"type": "boolean"},
        "text_readable": {"type": "boolean"},
        "text_exact": {"type": "boolean"},
        "ai_artificiality": {"type": "integer", "minimum": 1, "maximum": 10},
        "issues": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "decision": {"type": "string", "enum": ["ACCEPT", "REGENERATE"]},
    },
    "required": [
        "overall_score",
        "prompt_adherence",
        "composition",
        "visual_quality",
        "brand_fit",
        "realism_naturalness",
        "text_integration",
        "typography_quality",
        "text_visible",
        "text_readable",
        "text_exact",
        "ai_artificiality",
        "issues",
        "improvements",
        "decision",
    ],
}


def _validate_result(result: dict) -> dict:
    required = set(CRITIC_SCHEMA["required"])
    if not required.issubset(result):
        raise RuntimeError("Mistral critic returned an incomplete evaluation.")
    if result["decision"] not in {"ACCEPT", "REGENERATE"}:
        raise RuntimeError("Mistral critic returned an invalid decision.")
    return result


def critique_image(
    original_brief: str,
    image_prompt: str,
    image_bytes: bytes,
    text_required: bool = False,
    requested_text: str = "",
    text_style: str = "",
    text_placement: str = "",
    text_integration: str = "",
    text_rendering_mode: str = "model",
) -> dict:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing. Add it to a .env file before running the app."
        )

    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    client = Mistral(api_key=api_key)
    try:
        response = client.chat.complete(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict visual quality critic. Evaluate the image "
                        "against the original brief and prompt. Assess realism and "
                        "naturalness when realistic imagery is requested, including "
                        "materials, lighting, proportions, optics, and AI-like artifacts. "
                        "Return only JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Original brief:\n{original_brief}\n\n"
                                f"Optimized prompt:\n{image_prompt}\n\n"
                                "Score prompt adherence, composition, visual quality, "
                                "brand fit, realism_naturalness, and ai_artificiality "
                                "(10 means natural, not artificial). "
                                + (
                                    f"The requested text is exactly: {requested_text!r}. "
                                    "The image model was asked to create one complete "
                                    "artwork including it. Evaluate whether the text is "
                                    "visible, readable, and exact, whether typography "
                                    "belongs naturally to the composition, and whether "
                                    "it looks pasted on. Return text_visible, "
                                    "text_readable, and text_exact. "
                                    f"Evaluate its typography and integration against this "
                                    f"direction: style={text_style}; placement={text_placement}; "
                                    f"integration={text_integration}; rendering={text_rendering_mode}. "
                                    "Score text_integration and typography_quality."
                                    if text_required
                                    else "No text was requested; do not penalize or score text."
                                )
                                + " Use ACCEPT only when overall_score >= 7 and the image "
                                "is useful. Otherwise use REGENERATE."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded_image}",
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
        )
    except Exception as error:
        if getattr(error, "status_code", None) == 429 or "429" in str(error):
            raise VisionRateLimitError(
                "Vision Critic is temporarily rate-limited."
            ) from error
        raise
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Mistral critic returned an empty evaluation.")
    try:
        result = _validate_result(json.loads(content))
    except json.JSONDecodeError as error:
        raise RuntimeError("Mistral critic returned malformed JSON.") from error
    if text_required and not all(
        result[field] for field in ("text_visible", "text_readable", "text_exact")
    ):
        result["decision"] = "REGENERATE"
        result["issues"].append(
            "The mandatory requested phrase was missing, unreadable, or not exact."
        )
        result["improvements"].append(
            "Make the exact requested phrase a prominent, readable graphic element."
        )
    return result
