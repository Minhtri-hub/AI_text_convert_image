"""AI Comic Creator: Streamlit app for turning text into comic panels."""
from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI


MIN_SCENES = 3
MAX_SCENES = 6


@dataclass
class Scene:
    """Structured representation of a comic scene."""

    scene_id: int
    scene_summary: str
    scene_text: str
    dialogues: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    prompt: str | None = None
    image: Image.Image | None = None


OPENAI_IMAGE_MODEL = "gpt-image-1"


@st.cache_resource(show_spinner=False)
def get_openai_client() -> Optional[OpenAI]:
    """Create a cached OpenAI client if credentials are present."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:  # pragma: no cover - defensive safety net
        st.warning(f"Could not initialise OpenAI client: {exc}")
        return None


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(paragraph: str) -> List[str]:
    """Split the paragraph into rough sentences."""

    paragraph = normalise_whitespace(paragraph)
    if not paragraph:
        return []
    sentence_candidates = re.split(r"(?<=[.!?])\s+", paragraph)
    sentences = [normalise_whitespace(s) for s in sentence_candidates if s.strip()]
    return sentences


def ensure_min_sentences(sentences: List[str], target: int) -> List[str]:
    """Split longer sentences so we have at least ``target`` items."""

    if len(sentences) >= target:
        return sentences

    splitter = re.compile(r",\s+(?:and|but|so|then)\s+", flags=re.IGNORECASE)
    while len(sentences) < target:
        if not sentences:
            break
        longest_idx = max(range(len(sentences)), key=lambda idx: len(sentences[idx]))
        candidate = sentences.pop(longest_idx)
        parts = splitter.split(candidate)
        if len(parts) < 2:
            midpoint = max(len(candidate) // 2, 1)
            split_at = candidate.rfind(" ", 0, midpoint)
            if split_at == -1:
                split_at = midpoint
            parts = [candidate[:split_at], candidate[split_at:]]
        for part in parts[::-1]:
            cleaned = normalise_whitespace(part)
            if cleaned:
                sentences.insert(longest_idx, cleaned)
        if len(sentences) >= target:
            break
    return sentences


def distribute_sentences(sentences: List[str], target_groups: int) -> List[List[str]]:
    """Evenly distribute sentences into ``target_groups`` sequential groups."""

    groups: List[List[str]] = []
    total = len(sentences)
    base = total // target_groups
    remainder = total % target_groups
    index = 0
    for group_idx in range(target_groups):
        count = base + (1 if group_idx < remainder else 0)
        if count == 0:
            groups.append([])
            continue
        groups.append(sentences[index : index + count])
        index += count
    return groups


def detect_dialogues(text: str) -> List[str]:
    """Return phrases enclosed in single or double quotes."""

    matches = re.findall(r'"([^\"]+)"', text)
    matches.extend(re.findall(r"'([^']+)'", text))
    cleaned = [normalise_whitespace(m) for m in matches if normalise_whitespace(m)]
    unique_dialogues: List[str] = []
    for item in cleaned:
        if item not in unique_dialogues:
            unique_dialogues.append(item)
    return unique_dialogues


def detect_emotions(text: str) -> List[str]:
    """Detect basic emotion keywords inside the text."""

    emotion_keywords = {
        "happy": {"happy", "joy", "joyful", "cheerful", "delighted", "excited"},
        "sad": {"sad", "downcast", "gloomy", "tearful", "melancholy"},
        "angry": {"angry", "furious", "irate", "rage", "annoyed"},
        "scared": {"scared", "afraid", "fearful", "terrified", "anxious"},
        "surprised": {"surprised", "shocked", "astonished", "startled"},
        "calm": {"calm", "serene", "peaceful", "relaxed"},
    }
    words = {w.lower() for w in re.findall(r"[A-Za-z']+", text)}
    detected: List[str] = []
    for emotion, keywords in emotion_keywords.items():
        if words.intersection(keywords):
            detected.append(emotion)
    return detected


def extract_keywords(text: str, *, max_keywords: int = 6) -> List[str]:
    """Lightweight keyword extractor using frequency filtering."""

    words = re.findall(r"[A-Za-z'][A-Za-z'-]*", text.lower())
    stopwords = {
        "the",
        "and",
        "a",
        "an",
        "in",
        "on",
        "with",
        "under",
        "when",
        "while",
        "for",
        "to",
        "of",
        "at",
        "from",
        "that",
        "it",
        "was",
        "were",
        "is",
        "are",
    }
    counts: dict[str, int] = {}
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        counts[word] = counts.get(word, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in sorted_words[:max_keywords]]


def build_prompt(scene: Scene) -> str:
    """Construct a comic-friendly text-to-image prompt."""

    keywords = extract_keywords(scene.scene_text)
    keyword_fragment = ", ".join(keywords) if keywords else "important story elements"
    if scene.dialogues:
        dialogue_fragment = "; ".join(f'"{dlg}"' for dlg in scene.dialogues)
    else:
        dialogue_fragment = "no dialogue, focus on expressions"
    mood = ", ".join(scene.emotions) if scene.emotions else "neutral mood"
    prompt = (
        f"Scene {scene.scene_id} – {scene.scene_summary}. "
        "Comic panel illustration with bold ink outlines, vibrant colors, dynamic lighting. "
        f"Setting and action: {scene.scene_text}. "
        f"Key visual elements: {keyword_fragment}. "
        f"Character mood: {mood}. "
        f"Dialogue cues: {dialogue_fragment}."
    )
    return normalise_whitespace(prompt)


def placeholder_image(scene: Scene, *, size: int = 768) -> Image.Image:
    """Generate a coloured placeholder image containing the scene summary."""

    background_palette = [
        (246, 174, 45),
        (120, 200, 255),
        (189, 147, 249),
        (255, 121, 198),
        (85, 239, 196),
        (250, 177, 160),
    ]
    image = Image.new("RGB", (size, size), background_palette[(scene.scene_id - 1) % len(background_palette)])
    draw = ImageDraw.Draw(image)
    font_size = 28
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except (IOError, OSError):  # pragma: no cover - fallback when font missing
        font = ImageFont.load_default()
    text = f"Scene {scene.scene_id}\n{scene.scene_summary}"
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:  # pragma: no cover - fallback for older Pillow
        text_width, text_height = draw.multiline_textsize(text, font=font, spacing=6)
    position = ((size - text_width) / 2, (size - text_height) / 2)
    draw.multiline_text(position, text, fill="black", font=font, align="center", spacing=6)
    return image


def extract_scenes(paragraph: str) -> List[Scene]:
    sentences = split_into_sentences(paragraph)
    target_scene_count = min(max(len(sentences), MIN_SCENES), MAX_SCENES)
    sentences = ensure_min_sentences(sentences, target_scene_count)
    if not sentences:
        return []
    if len(sentences) < target_scene_count:
        target_scene_count = len(sentences)
    sentence_groups = distribute_sentences(sentences, target_scene_count)

    scenes: List[Scene] = []
    for idx, group in enumerate(sentence_groups, start=1):
        group_text = normalise_whitespace(" ".join(group))
        if not group_text:
            continue
        summary = group[0]
        if len(summary) > 120:
            truncated = summary[:117]
            if " " in truncated:
                truncated = truncated.rsplit(" ", 1)[0]
            summary = truncated.rstrip(" ,.;:") + "..."
        dialogues = detect_dialogues(group_text)
        emotions = detect_emotions(group_text)
        scenes.append(
            Scene(
                scene_id=idx,
                scene_summary=summary,
                scene_text=group_text,
                dialogues=dialogues,
                emotions=emotions,
            )
        )
    return scenes


def generate_storyboard(paragraph: str) -> List[Scene]:
    scenes = extract_scenes(paragraph)
    for scene in scenes:
        scene.prompt = build_prompt(scene)
    return scenes


def render_storyboard(scenes: List[Scene]) -> None:
    st.markdown("### Storyboard")
    for scene in scenes:
        with st.expander(f"Scene {scene.scene_id}: {scene.scene_summary}"):
            st.write(scene.scene_text)
            if scene.dialogues:
                st.markdown("**Dialogues detected:**")
                for dialogue in scene.dialogues:
                    st.markdown(f"- “{dialogue}”")
            if scene.emotions:
                st.markdown(
                    "**Emotions detected:** " + ", ".join(scene.emotions)
                )
            st.markdown("**Image prompt:**")
            st.code(scene.prompt or "No prompt generated.")


def render_panels(scenes: List[Scene]) -> None:
    st.markdown("### Comic Panels")
    columns_per_row = 2
    cols = st.columns(columns_per_row)
    for idx, scene in enumerate(scenes):
        column = cols[idx % columns_per_row]
        with column:
            st.subheader(f"Scene {scene.scene_id}")
            if scene.image is not None:
                st.image(scene.image, caption=scene.scene_summary, use_container_width=True)
            else:
                st.image(placeholder_image(scene), caption=scene.scene_summary, use_container_width=True)
            mood_caption = ", ".join(scene.emotions) if scene.emotions else "neutral mood"
            st.caption(f"Mood: {mood_caption}\nPrompt: {scene.prompt}")
        if (idx + 1) % columns_per_row == 0 and idx != len(scenes) - 1:
            cols = st.columns(columns_per_row)


def generate_images(scenes: List[Scene], openai_client: Optional[OpenAI]) -> None:
    if not openai_client:
        return
    for scene in scenes:
        if not scene.prompt:
            continue
        try:
            response = openai_client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=scene.prompt,
                size="768x768",
            )
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            scene.image = Image.open(BytesIO(image_bytes))
        except Exception as exc:  # pragma: no cover - fallback for unexpected issues
            st.warning(
                f"OpenAI image generation failed for Scene {scene.scene_id}: {exc}"
            )


def main() -> None:
    st.set_page_config(page_title="AI Comic Creator", layout="wide")
    st.title("AI Comic Creator")
    st.write(
        "Provide a short paragraph (2–8 sentences) and watch it turn into a storyboard and illustrated comic panels."
    )

    with st.form("comic-form"):
        paragraph = st.text_area(
            "Story paragraph",
            height=180,
            placeholder="It was raining when John met Lily under the old bridge...",
        )
        submitted = st.form_submit_button("Generate Comic")

    if not submitted:
        st.info("Describe your story above and click **Generate Comic** to begin.")
        return

    if not paragraph or not paragraph.strip():
        st.warning("Please provide a story paragraph before generating the comic.")
        return

    with st.spinner("Extracting scenes and building prompts..."):
        scenes = generate_storyboard(paragraph)

    if not scenes:
        st.error("Could not derive any scenes from the supplied paragraph. Try adding more detail.")
        return

    render_storyboard(scenes)

    openai_client = get_openai_client()
    if openai_client:
        with st.spinner("Generating illustrated panels with OpenAI..."):
            generate_images(scenes, openai_client)
    else:
        st.info(
            "OpenAI credentials were not detected. Displaying placeholder panels instead."
        )

    render_panels(scenes)


if __name__ == "__main__":
    main()
