import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from creative_director import create_creative_plan, improve_image_prompt
from image_generator import generate_image
from text_overlay import apply_text_overlay
from vision_critic import VisionRateLimitError, critique_image


MAX_ATTEMPTS = 3
LOG_PATH = Path("logs/runs.jsonl")
logger = logging.getLogger(__name__)


def _log_run(data: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(data) + "\n")


def run_creative_pipeline(user_brief: str) -> dict:
    if not user_brief.strip():
        raise ValueError("The creative brief cannot be empty.")

    plan = create_creative_plan(user_brief)
    prompt = plan["image_prompt"]
    text_content = plan["text_content"] if plan["text_required"] else ""
    text_style = plan["text_style"] if plan["text_required"] else ""
    text_placement = plan["text_placement"] if plan["text_required"] else ""
    text_integration = plan["text_integration"] if plan["text_required"] else ""
    # Exact text must use the compositor so model-generated copy cannot add a
    # second, malformed banner or duplicate phrase.
    text_rendering_mode = "hybrid" if plan["text_required"] else "model"
    attempts = []
    image_bytes = None
    critic_result = None

    with tempfile.TemporaryDirectory(prefix="creative_studio_") as directory:
        for attempt_number in range(1, MAX_ATTEMPTS + 1):
            image_bytes = generate_image(prompt)
            if text_content and text_rendering_mode in {"hybrid", "pillow_fallback"}:
                source_path = Path(directory) / f"attempt_{attempt_number}.png"
                source_path.write_bytes(image_bytes)
                final_path = apply_text_overlay(
                    source_path,
                    text_content,
                    text_style,
                    text_placement,
                    text_integration,
                )
                image_bytes = final_path.read_bytes()
            try:
                critic_result = critique_image(
                    user_brief,
                    prompt,
                    image_bytes,
                    text_required=bool(text_content),
                    requested_text=text_content,
                    text_style=text_style,
                    text_placement=text_placement,
                    text_integration=text_integration,
                    text_rendering_mode=text_rendering_mode,
                )
            except VisionRateLimitError:
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "prompt": prompt,
                        "critic": None,
                    }
                )
                result = {
                    "original_brief": user_brief,
                    "final_image": image_bytes,
                    "final_prompt": prompt,
                    "text_content": text_content,
                    "attempts": attempts,
                    "critic_result": None,
                    "status": "vision_rate_limited",
                }
                _log_run(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "original_brief": user_brief,
                        "optimized_prompt": attempts[0]["prompt"],
                        "text_content": text_content,
                        "attempts": attempts,
                        "final_status": result["status"],
                    }
                )
                return result
            attempts.append(
                {
                    "attempt": attempt_number,
                    "prompt": prompt,
                    "critic": critic_result,
                }
            )
            if critic_result["decision"] == "ACCEPT":
                break
            if attempt_number < MAX_ATTEMPTS:
                prompt = improve_image_prompt(
                    user_brief,
                    prompt,
                    critic_result,
                    text_required=bool(text_content),
                    requested_text=text_content,
                    text_style=text_style,
                    text_placement=text_placement,
                    text_integration=text_integration,
                    text_rendering_mode=text_rendering_mode,
                )

    status = (
        "accepted"
        if critic_result and critic_result["decision"] == "ACCEPT"
        else "maximum_attempts_reached"
    )
    result = {
        "original_brief": user_brief,
        "final_image": image_bytes,
        "final_prompt": prompt,
        "text_content": text_content,
        "attempts": attempts,
        "critic_result": critic_result,
        "status": status,
    }
    _log_run(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_brief": user_brief,
            "optimized_prompt": attempts[0]["prompt"],
            "text_content": text_content,
            "attempts": attempts,
            "final_status": status,
        }
    )
    return result
