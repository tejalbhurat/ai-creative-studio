# AI Creative Studio

A multimodal Agentic AI system that transforms creative briefs into generated
visuals, evaluates them with vision AI, and iteratively improves them.

## Architecture

```text
User brief
    -> Groq Creative Director
    -> optimized image prompt, realism direction, and text decision
    -> Mistral image_generation tool
    -> image-aware hybrid typography compositor
    -> generated image
    -> Mistral vision critic
    -> structured evaluation
    -> Groq prompt improver when rejected
    -> bounded regeneration (maximum 3 attempts)
    -> final creative
```

## Tech stack

- Python
- Streamlit
- Groq chat completions for creative direction and prompt improvement
- Mistral Agents/Conversations image-generation tool
- Pillow for exact requested text overlays
- Mistral vision-capable Chat Completions model
- Structured JSON critic output
- Local JSONL observability logs

Mistral image generation is an account-controlled feature. Mistral documents
Studio Free mode with limited usage and no credit card required for API access,
but image-generation entitlement and limits can vary by account. This project
does not claim unlimited or universally free image generation.

## Project structure

```text
Project4/
├── app.py                # Thin Streamlit UI
├── creative_director.py  # Groq Creative Director and prompt improver
├── image_generator.py    # Mistral image_generation adapter
├── text_overlay.py       # Contextual exact text overlay with Pillow
├── vision_critic.py      # Mistral vision evaluation adapter
├── creative_pipeline.py  # Bounded orchestration and logging
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup on macOS

```bash
cd /Users/tejalbhurat/Documents/Project4
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set these values in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_VISION_MODEL=mistral-small-latest
```

Start the application:

```bash
streamlit run app.py
```

## Agent workflow

1. Groq returns a structured plan containing `image_prompt`, `text_required`,
   exact `text_content`, and contextual `text_style`, `text_placement`, and
   `text_integration` directions. It preserves requested mediums and adds
   realism guidance only when appropriate.
2. Mistral's documented `image_generation` built-in tool generates the visual.
   When text is required, the prompt explicitly tells Mistral not to render it.
3. The generated file is downloaded through Mistral's Files API.
4. When text is requested, the default hybrid path asks the image model for the
   visual artwork without unreliable readable copy, then uses an image-aware
   typography compositor. It samples local tones, applies contrast-aware
   gradient fills, restrained shadow/depth, responsive typography, and contextual
   placement without a generic rectangle or banner. The model-rendered path
   remains available explicitly, and no-text images are unchanged.
5. Mistral's vision model receives the original brief, optimized prompt, and
   final overlaid image as a base64 data URL.
6. The critic returns JSON scores, issues, improvements, and either `ACCEPT` or
   `REGENERATE`.
7. A rejected image is sent to the Groq prompt improver, which rewrites the
   prompt instead of appending feedback.
8. The pipeline regenerates, reapplies the same exact text overlay, and critiques
   until accepted or three attempts are
   reached.

The local workflow log is written to `logs/runs.jsonl`. It contains briefs,
prompts, attempt numbers, scores, feedback, and status, but never API keys.

## Validation criteria

- Missing API keys show a concise Streamlit error.
- A blank brief is rejected before any API call.
- The image-generation adapter uses only Mistral's documented agent tool and
  file-download flow.
- Critic output is parsed as JSON and validated for required fields.
- Regeneration is bounded by `MAX_ATTEMPTS = 3`.
- The app displays the brief, final prompt, image, scores, feedback, and status.
- Requested text is overlaid exactly by Pillow; briefs without requested text
  receive no application-added text.
- No Hugging Face dependency or configuration is used.

## Official API references

- [Mistral image-generation agent tool](https://docs.mistral.ai/studio/agents/agent-tools/image_generation)
- [Mistral agent tools](https://docs.mistral.ai/studio/agents/agent-tools)
- [Mistral vision](https://docs.mistral.ai/studio/conversations/vision/)
- [Mistral structured outputs](https://docs.mistral.ai/studio/conversations/structured-output/json_mode)
- [Mistral Studio setup](https://docs.mistral.ai/getting-started/quickstarts/studio/activate-and-generate-api-key)
