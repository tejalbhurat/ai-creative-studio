import json
import os

from groq import Groq


MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SYSTEM_PROMPT = """
You are the Creative Director for an AI image-generation studio.
Analyze the user's request and return only a valid JSON object with these keys:
image_prompt, text_required, text_content, text_style, text_placement,
text_integration, text_rendering_mode.
image_prompt must be a detailed, production-ready prompt for the visual.
Preserve the user's intent and add useful composition, lighting, color palette,
camera/style, audience, aspect-ratio, and art-direction details. Respect the
requested medium: do not turn illustrations, anime, watercolor, 3D, or graphic
design into photography. Decide whether the brief calls for realistic
photography, a designed graphic, or another medium. For realistic imagery,
prioritize natural lighting, believable shadows, realistic materials and
proportions, authentic environmental details, realistic camera perspective,
moderate depth of field, restrained color grading, subtle photographic grain
when appropriate, and imperfect but intentional details. Avoid plastic-looking
surfaces or skin, excessive smoothness, fake HDR, excessive glow or neon,
impossible reflections or shadows, perfect symmetry, floating objects, overly
saturated colors, excessive blur, generic futuristic CGI, overly clean
environments, and generic AI-art aesthetics. For designed graphics, art-direct
a coherent layout, hierarchy, spacing, and visual system instead of simulating
a photograph. If text_required is true, make the image model create one complete
artwork including the exact requested phrase. Include the phrase in image_prompt,
require readable spelling, capitalization, and punctuation, and describe how the
typography belongs to the composition. Set text_rendering_mode to "hybrid" by default. In hybrid mode, the image model
creates the visual artwork and the application creates the exact typography using
an image-aware compositor; do not render readable text in the image model. Use "model" only when explicitly requested, although the application may enforce
hybrid rendering to prevent duplicate or malformed model text. If text_required
is false, text_content must be an empty string and
text_style, text_placement, and text_integration must also be empty strings; do
not add that instruction. Only set text_required true when the user explicitly
asks for visible words, a phrase, a title, a sign, a caption, a logo name, or
similar text. text_content must contain only the exact words the user requested,
without invented slogans or explanations. Choose a text_style, text_placement,
and text_integration that fit the brief. Typography must be a primary
graphic-design element of the finished artwork, with intentional scale,
hierarchy, spacing, overlap, perspective, layering, texture, and interaction
with the subject and environment. Do not use a generic rectangle, banner,
sticker, watermark, label, card, or separate text layer. Do not omit, hide,
replace, paraphrase, or use meaningless pseudo-lettering for the requested
phrase.
Do not invent event facts, logos, dates, or readable text beyond the requested
phrase.
""".strip()


def create_creative_plan(user_request: str) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to a .env file before running the app."
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ],
        temperature=0.7,
        max_completion_tokens=600,
        reasoning_effort="low",
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty creative plan.")
    try:
        plan = json.loads(content)
    except json.JSONDecodeError:
        return {
            "image_prompt": user_request.strip(),
            "text_required": False,
            "text_content": "",
            "text_style": "",
            "text_placement": "",
            "text_integration": "",
            "text_rendering_mode": "hybrid",
        }
    image_prompt = plan.get("image_prompt")
    text_required = plan.get("text_required") is True
    text_content = plan.get("text_content") if text_required else ""
    text_style = plan.get("text_style") if text_required else ""
    text_placement = plan.get("text_placement") if text_required else ""
    text_integration = plan.get("text_integration") if text_required else ""
    text_rendering_mode = (
        plan.get("text_rendering_mode", "hybrid") if text_required else "model"
    )
    if not isinstance(image_prompt, str) or not image_prompt.strip():
        return {
            "image_prompt": user_request.strip(),
            "text_required": False,
            "text_content": "",
            "text_style": "",
            "text_placement": "",
            "text_integration": "",
            "text_rendering_mode": "hybrid",
        }
    if not all(
        isinstance(value, str)
        for value in (
            text_content,
            text_style,
            text_placement,
            text_integration,
            text_rendering_mode,
        )
    ):
        text_content = ""
        text_required = False
        text_style = text_placement = text_integration = ""
        text_rendering_mode = "hybrid"
    if text_rendering_mode not in {"model", "hybrid", "pillow_fallback"}:
        text_rendering_mode = "hybrid"
    if text_required and text_rendering_mode == "hybrid":
        image_prompt = (
            f"{image_prompt.strip()}\n\n"
            "Create the visual artwork and composition without readable words, "
            "letters, logos, captions, or pseudo-text. The application will render "
            "the exact typography as an image-aware design treatment afterward; "
            "provide strong composition, lighting, texture, and visual hierarchy "
            "for that treatment. Do not create a fake sign, screen, banner, card, "
            "or text box."
        )
    elif text_required and text_rendering_mode == "model":
        image_prompt = (
            f"{image_prompt.strip()}\n\n"
            f'REQUIRED TEXT: "{text_content}"\n'
            "The requested phrase MUST be visibly rendered and readable in the "
            "final artwork as a mandatory primary graphic-design element. Build "
            "the imagery and typography as one cohesive composition. Preserve "
            "exact spelling, capitalization, and punctuation. Do not omit, hide, "
            "replace, paraphrase, or use placeholder text."
        )
    return {
        "image_prompt": image_prompt.strip(),
        "text_required": text_required,
        "text_content": text_content,
        "text_style": text_style.strip(),
        "text_placement": text_placement.strip(),
        "text_integration": text_integration.strip(),
        "text_rendering_mode": text_rendering_mode,
    }


def create_image_prompt(user_request: str) -> str:
    """Return the visual prompt while preserving the original public API."""
    return create_creative_plan(user_request)["image_prompt"]


def improve_image_prompt(
    original_brief: str,
    current_prompt: str,
    critic_result: dict,
    text_required: bool = False,
    requested_text: str = "",
    text_style: str = "",
    text_placement: str = "",
    text_integration: str = "",
    text_rendering_mode: str = "hybrid",
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to a .env file before running the app."
        )

    feedback = "\n".join(
        [
            f"Overall score: {critic_result['overall_score']}",
            f"Prompt adherence: {critic_result['prompt_adherence']}",
            f"Composition: {critic_result['composition']}",
            f"Visual quality: {critic_result['visual_quality']}",
            f"Brand fit: {critic_result['brand_fit']}",
            f"Issues: {', '.join(critic_result['issues'])}",
            f"Improvements: {', '.join(critic_result['improvements'])}",
            f"Text visible: {critic_result.get('text_visible', 'not evaluated')}",
            f"Text readable: {critic_result.get('text_readable', 'not evaluated')}",
            f"Text exact: {critic_result.get('text_exact', 'not evaluated')}",
        ]
    )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior image-prompt editor. Rewrite the current "
                    "prompt to address every relevant critic issue while preserving "
                    "the original brief. Preserve the requested medium and do not "
                    "add generic AI-art styling. Return only the complete replacement "
                    "prompt."
                    + (
                        f' The complete artwork MUST include the exact phrase '
                        f'"{requested_text}" as a visible primary graphic-design '
                        f"element. "
                        f"with this direction: {text_style}; {text_placement}; "
                        f"{text_integration}. Keep it readable and do not use "
                        "placeholder text."
                        if text_required
                        else ""
                    ),
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original brief:\n{original_brief}\n\n"
                    f"Current prompt:\n{current_prompt}\n\n"
                    f"Critic feedback:\n{feedback}"
                ),
            },
        ],
        temperature=0.5,
        max_completion_tokens=500,
        reasoning_effort="low",
    )
    prompt = response.choices[0].message.content
    if not prompt:
        raise RuntimeError("Groq returned an empty improved prompt.")
    improved_prompt = prompt.strip()
    if text_required:
        improved_prompt += (
            f'\n\nREQUIRED TEXT: "{requested_text}"\n'
            "The previous attempt failed to satisfy the mandatory typography "
            "requirement. The exact phrase MUST be visibly rendered and readable "
            "in the final artwork. Preserve spelling, capitalization, and "
            "punctuation. Make it a primary graphic-design element integrated with "
            "the imagery. Do not omit, hide, replace, paraphrase, or use placeholder "
            "text. Do not use a generic rectangle, banner, sticker, watermark, "
            "label, card, or separate text layer."
        )
    return improved_prompt
