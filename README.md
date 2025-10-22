# AI Comic Strip Prototype

This project contains a simple Streamlit application that turns a short idea into a four-panel comic strip using OpenAI's text and image generation APIs.

## Getting Started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
4. Run the app: `streamlit run streamlit_app.py`.

Once Streamlit launches in your browser, paste a short story idea, then click
**Generate comic** to see up to four panels.

## Hướng dẫn sử dụng (Vietnamese usage guide)

1. Cài Python 3.10 trở lên và tạo môi trường ảo:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows dùng: .venv\Scripts\activate
   ```
2. Cài thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Thiết lập khóa OpenAI API trước khi chạy ứng dụng:
   ```bash
   export OPENAI_API_KEY=sk-...       # macOS/Linux
   setx OPENAI_API_KEY sk-...         # Windows PowerShell (sau đó mở cửa sổ mới)
   ```
4. Khởi chạy ứng dụng Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```
5. Truy cập địa chỉ mà Streamlit hiển thị (mặc định là http://localhost:8501),
   nhập mô tả ngắn (1–2 câu) về câu chuyện và bấm **Generate comic** để tạo
   dải truyện tranh 4 khung.

## How it works

1. The app expands the user's idea into short panel descriptions with `gpt-4o-mini`.
2. Each panel description is rendered as an image with `gpt-image-1`.
3. Streamlit displays the panels in a simple 2x2 grid.

If image generation fails, the app falls back to showing the text prompt so you can still iterate on your story.
