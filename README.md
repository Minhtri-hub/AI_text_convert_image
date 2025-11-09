# AI_text_convert_image
# AI Comic Creator

AI Comic Creator is a Streamlit web application that turns a short story paragraph into a comic storyboard and illustrated panels. The app analyses your text, splits it into scenes, generates descriptive prompts, tags basic emotions, and optionally calls the OpenAI Images API to render each panel.

## Features

- **Scene extraction** – Breaks a paragraph into 3–6 sequential scenes with summaries, detailed descriptions, detected emotion keywords, and dialogue lines.
- **Prompt generation** – Builds comic-style text-to-image prompts for every scene using lightweight keyword extraction and emotion cues.
- **Storyboard view** – Shows expandable scene cards with summaries, full text, dialogues, and the generated prompt.
- **Panel rendering** – Displays either OpenAI-generated illustrations or colourful placeholder panels when no API credentials are provided.

## Prerequisites

- Python 3.10 or newer
- An OpenAI API key with access to the Images API (optional – the app still runs with placeholder images).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

To enable AI-generated artwork, export your OpenAI credentials before running the app:

```bash
export OPENAI_API_KEY=sk-your-key
```

Windows PowerShell equivalents:

```powershell
setx OPENAI_API_KEY sk-your-key
```

If this variable is missing, the app continues to work with locally generated placeholder panels so you can still iterate on the storyboard and prompts.

## Usage

```bash
streamlit run streamlit_app.py
```

Once the Streamlit dashboard opens:

1. Paste a short paragraph (2–8 sentences work best) into the **Story paragraph** box.
2. Click **Generate Comic**.
3. Review the storyboard expanders to understand how the text was split into scenes.
4. Scroll down to the **Comic Panels** section to view the illustrations (or placeholders) along with their prompts and mood tags.

## Vietnamese guide / Hướng dẫn tiếng Việt

1. Cài Python 3.10 trở lên và tạo môi trường ảo:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows dùng: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. (Tùy chọn) Thiết lập biến môi trường cho OpenAI trước khi chạy ứng dụng:
   ```bash
   export OPENAI_API_KEY=sk-your-key
   ```
3. Khởi động ứng dụng:
   ```bash
   streamlit run streamlit_app.py
   ```
4. Nhập một đoạn mô tả ngắn (2–8 câu) vào ô **Story paragraph** rồi bấm **Generate Comic**. Ứng dụng sẽ tự động chia câu chuyện thành các cảnh, tạo lời gợi ý vẽ, gắn nhãn cảm xúc và hiển thị khung truyện tranh tương ứng. Nếu có API key OpenAI, ứng dụng còn tạo hình minh họa cho từng cảnh.

## Development notes

- The scene extraction uses deterministic heuristics so it works offline.
- Prompts and placeholder art are shown even when OpenAI is unavailable, which makes it easy to debug prompt design.
- If you integrate other image APIs, adjust `generate_images` with the provider of your choice.
requirements.txt
New
