"""Streamlit prototype for generating comic panels from text descriptions."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import streamlit as st
from openai import OpenAI
from PIL import Image


GPT_MODEL = "gpt-4o-mini"
IMAGE_MODEL = "gpt-image-1"


@dataclass
class Panel:
    """Represents a single comic panel."""

    title: str
    prompt: str
    image: Image.Image | None



def _get_client() -> OpenAI:
    """Create a cached OpenAI client using the environment API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the OPENAI_API_KEY environment variable to use the demo."
        )
    return OpenAI(api_key=api_key)


def generate_panel_descriptions(client: OpenAI, idea: str) -> List[Panel]:
    """Use the LLM to break the idea into panel prompts."""
    system_prompt = (
        "You design short four-panel comic strips. Return four bullet points. "
        "Each bullet contains a short panel title followed by a colon and an "
        "image prompt describing the scene."
    )
    response = client.responses.create(
        model=GPT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Idea: {idea}\nReturn format: Title: image prompt",
            },
        ],
    )
    text = response.output_text
    panels: List[Panel] = []
    for line in text.splitlines():
        line = line.strip(" -*")
        if not line:
            continue
        if ":" in line:
            title, prompt = line.split(":", 1)
            panels.append(Panel(title=title.strip(), prompt=prompt.strip(), image=None))
    if not panels:
        raise RuntimeError(
            "The language model did not return any panel descriptions."
        )
    return panels[:4]


def generate_panel_image(client: OpenAI, panel: Panel) -> Panel:
    """Generate an image for a single panel using the image model."""
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=(
            "Comic book style, colorful ink outlines, panel illustration. "
            + panel.prompt
        ),
        size="512x512",
    )
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    return Panel(title=panel.title, prompt=panel.prompt, image=image)


def render_panels(panels: List[Panel]) -> None:
    """Display panels in a 2x2 layout."""
    cols = st.columns(2)
    for idx, panel in enumerate(panels):
        column = cols[idx % 2]
        with column:
            st.subheader(panel.title)
            if panel.image is not None:
                st.image(panel.image, caption=panel.prompt, use_container_width=True)
            else:
                st.info(panel.prompt)
        if idx % 2 == 1 and idx != len(panels) - 1:
            cols = st.columns(2)


def main() -> None:
    st.set_page_config(page_title="AI Comic Generator", layout="wide")
    st.title("AI Comic Strip Prototype")
    st.write(
        "Describe a story idea in one or two sentences and generate a four-panel comic."
    )

    with st.form("comic-form"):
        description = st.text_area(
            "Comic idea",
            placeholder="e.g. A cat invents a robot vacuum that chases lasers.",
        )
        submitted = st.form_submit_button("Generate comic")

    if submitted:
        if not description.strip():
            st.warning("Please enter a short description before generating a comic.")
            return
        try:
            client = _get_client()
        except RuntimeError as exc:  # pragma: no cover - requires configuration
            st.error(str(exc))
            return

        with st.spinner("Drafting comic panels..."):
            try:
                panels = generate_panel_descriptions(client, description)
            except Exception as exc:  # pragma: no cover - handles API errors gracefully
                st.error(f"Failed to generate panel descriptions: {exc}")
                return

        generated_panels: List[Panel] = []
        for panel in panels:
            with st.spinner(f"Rendering panel: {panel.title}"):
                try:
                    generated_panels.append(generate_panel_image(client, panel))
                except Exception as exc:  # pragma: no cover - handles API errors gracefully
                    st.warning(
                        f"Could not create an image for '{panel.title}'. Showing prompt instead."
                    )
                    generated_panels.append(panel)
                    st.exception(exc)

        render_panels(generated_panels)


if __name__ == "__main__":
    main()
