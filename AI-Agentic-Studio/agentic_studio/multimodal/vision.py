"""Image input.

Images travel through the same `Message` type as text (`Message.user(text,
images=[...])`), so vision is a provider capability rather than a separate code
path. When no vision-capable provider is configured, the functions return real
image metadata instead of pretending to see the picture.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from agentic_studio.agents.tools.registry import tool
from agentic_studio.core.types import Message
from agentic_studio.observability.logs import get_logger
from agentic_studio.observability.tracing import get_tracer

logger = get_logger("multimodal.vision")

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_BYTES = 8 * 1024 * 1024


def to_data_url(path: Path | str) -> str:
    """Encode a local image as a data URL for provider APIs."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"unsupported image type: {path.suffix}")
    payload = path.read_bytes()
    if len(payload) > MAX_BYTES:
        raise ValueError(f"image is {len(payload)} bytes, limit is {MAX_BYTES}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def image_metadata(path: Path | str) -> dict[str, Any]:
    """Dimensions, mode, and format, using Pillow when it is installed."""
    path = Path(path)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "suffix": path.suffix.lower(),
    }
    try:
        from PIL import Image

        with Image.open(path) as image:
            info.update({"width": image.width, "height": image.height, "mode": image.mode,
                         "format": image.format})
    except Exception as exc:
        info["metadata_error"] = str(exc)
    return info


def supports_vision(router: Any = None) -> bool:
    from agentic_studio.llm.router import get_router

    router = router or get_router()
    return any(
        provider.supports_vision and provider.available() and provider.name != "echo"
        for provider in router.providers
    )


def describe_image(
    source: str,
    question: str = "Describe this image in detail.",
    router: Any = None,
) -> dict[str, Any]:
    """Ask a vision model about an image; `source` is a local path or an http(s) URL."""
    from agentic_studio.llm.router import get_router

    router = router or get_router()
    is_url = source.startswith(("http://", "https://", "data:"))

    with get_tracer().span("vision.describe", kind="llm", remote=is_url) as span:
        if is_url:
            reference = source
            metadata: dict[str, Any] = {"url": source}
        else:
            metadata = image_metadata(source)
            if not metadata.get("exists"):
                return {"ok": False, "error": f"image not found: {source}"}
            reference = to_data_url(source)

        if not supports_vision(router):
            span.set(vision=False)
            return {
                "ok": True,
                "vision_model_used": False,
                "description": (
                    "No vision-capable provider is configured, so the image was not analysed. "
                    "Metadata only. Set STUDIO_LLM_PROVIDERS to openai, anthropic, gemini, or "
                    "ollama with a vision model."
                ),
                "metadata": metadata,
            }

        response = router.generate(
            [
                Message.system("You are a careful visual analyst. Describe only what is visible."),
                Message.user(question, images=[reference]),
            ],
            use_cache=False,
        )
        span.set(vision=True, tokens=response.usage.total_tokens)
        return {
            "ok": True,
            "vision_model_used": True,
            "description": response.text,
            "metadata": metadata,
            "provider": response.provider,
        }


@tool(name="describe_image", tags=("multimodal",))
def describe_image_tool(source: str, question: str = "Describe this image in detail.") -> dict[str, Any]:
    """Analyse an image and answer a question about it.

    Args:
        source: Local image path or an http(s) image URL.
        question: What to ask about the image.
    """
    return describe_image(source, question)
