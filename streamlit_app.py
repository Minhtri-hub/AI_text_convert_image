import streamlit as st
import requests
import pandas as pd
import zipfile, io

st.set_page_config(page_title="AI Comic Factory Việt Hóa", layout="wide")

st.title("🎨 AI Comic Factory Việt Hóa")

style = st.selectbox("Phong cách vẽ:", ["manga cổ điển", "anime", "realistic", "sketch"])
ratio = st.radio("Tỷ lệ khung:", ["16:9", "1:1", "9:16"], horizontal=True)
book_id = st.text_input("Book ID", "B001")
chapter_id = st.text_input("Chapter ID", "CH01")
raw_text = st.text_area("Raw Text", height=300)

if st.button("🚀 Sinh truyện tranh"):
    with st.spinner("Đang xử lý..."):
        resp = requests.post("https://n8n.n2nai.io/webhook-test/56ffc891-809f-4bd8-b74d-a8a9185a5cba",
                             json={"Book_ID": book_id, "Chapter_ID": chapter_id,
                                   "Raw_Text": raw_text, "Style": style, "Ratio": ratio},
                             timeout=600)
        if resp.status_code != 200:
            st.error(resp.text)
        else:
            data = resp.json()
            st.success(f"✅ Đã tạo {len(data['scenes'])} cảnh!")
            for scene in data["scenes"]:
                st.subheader(f"{scene['scene_id']} — {scene['scene_summary']}")
                st.image(scene["image_url"], use_container_width=True)
                st.caption(f"🎭 {scene['emotion']} | 🗣 {scene['dialogue'] or 'Không có thoại'}")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for sc in data["scenes"]:
                    zf.writestr(f"{sc['scene_id']}.txt", sc["image_url"])
            st.download_button("📦 Tải ZIP metadata", buf.getvalue(), file_name="comic_metadata.zip")
