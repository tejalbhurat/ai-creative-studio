import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral


IMAGE_AGENT_MODEL = "mistral-medium-latest"
logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def _find_file_id(response) -> str | None:
    """Find a tool-file chunk across SDK models and serialized response shapes."""
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    if isinstance(response, dict):
        if response.get("type") == "tool_file" and response.get("file_id"):
            return response["file_id"]
        for value in response.values():
            file_id = _find_file_id(value)
            if file_id:
                return file_id
        return None
    if isinstance(response, (list, tuple)):
        for value in response:
            file_id = _find_file_id(value)
            if file_id:
                return file_id
        return None
    if getattr(response, "type", None) == "tool_file":
        return getattr(response, "file_id", None)
    if hasattr(response, "outputs"):
        return _find_file_id(response.outputs)
    if hasattr(response, "content"):
        return _find_file_id(response.content)
    return None


def _mistral_error_details(error: Exception) -> str:
    """Return safe status/body diagnostics without exposing authentication data."""
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    body = getattr(error, "body", None)
    if body is None and response is not None:
        try:
            body = response.text
        except Exception:
            body = None

    detail = str(body or error).strip()
    api_key = os.getenv("MISTRAL_API_KEY")
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    if len(detail) > 1000:
        detail = detail[:1000] + "..."
    return f"HTTP {status_code}: {detail}" if status_code else detail


def generate_image(image_prompt: str):
    """Generate an image with Mistral's documented image_generation tool."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing. Add it to a .env file before running the app."
        )

    logger.warning(
        "Mistral image request: model=%s method=beta.agents.create+"
        "beta.conversations.start+files.download api_key_present=%s",
        IMAGE_AGENT_MODEL,
        bool(api_key),
    )
    client = Mistral(api_key=api_key)
    try:
        image_agent = client.beta.agents.create(
            model=IMAGE_AGENT_MODEL,
            name="AI Creative Studio Image Generator",
            description="Generates a single image from an optimized creative prompt.",
            instructions="Use the image generation tool to create the requested image.",
            tools=[{"type": "image_generation"}],
        )
        response = client.beta.conversations.start(
            agent_id=image_agent.id,
            inputs=image_prompt,
        )
        file_id = _find_file_id(response)
        if not file_id:
            raise RuntimeError("Mistral returned no generated image file.")
        return client.files.download(file_id=file_id).read()
    except RuntimeError:
        raise
    except Exception as error:
        logger.error(
            "Mistral image request failed: exception_class=%s %s",
            type(error).__name__,
            _mistral_error_details(error),
        )
        raise RuntimeError(
            f"Mistral image generation failed: {_mistral_error_details(error)}"
        ) from error
